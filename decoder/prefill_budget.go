package decoder

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// prefillEnters counts every call to (*Model).NewCache — the actual KV allocation a generation
// makes. Mirrors fitguard.go's weightAllocs: it exists so "AdmitPrefillMemory is a pure check,
// never itself the thing that allocates" is an assertion a test can make, not a deduction from
// the absence of a crash.
var prefillEnters atomic.Int64

// prefillAvailFraction is the share of currently available memory a request's own KV+scratch may consume. It has the same value as
// fitguard.go's fitMemFraction, inherited rather than independently measured for this context (available-RAM headroom, not a
// fraction of total RAM); it has no measurement of its own.
const prefillAvailFraction = fitMemFraction

// availProbeTTL rate-limits the currently-available memory probe: hostRAMAvailable shells out (vm_stat on darwin, a /proc/meminfo
// read on linux), and AdmitPrefillMemory runs it for every request that reaches prefill, a fork+exec per chat request on darwin.
// Available memory changes on the timescale of other processes starting and exiting, not per request, so 250 ms of staleness is
// acceptable.
const availProbeTTL = 250 * time.Millisecond

var (
	availProbeMu    sync.Mutex
	availProbeValue int64
	availProbeAt    time.Time
)

// cachedHostRAMAvailable is hostRAMAvailable (the raw, uncached, test-injectable indirection) behind a short TTL cache. It is a
// separate wrapper so a test overriding hostRAMAvailable still sees its own value immediately: resetAvailProbeCache, called from
// those tests' inject helper, clears the cache so a fresh override is never masked by a prior test's cached read.
func cachedHostRAMAvailable() int64 {
	availProbeMu.Lock()
	defer availProbeMu.Unlock()
	if now := time.Now(); now.Sub(availProbeAt) < availProbeTTL {
		return availProbeValue
	}
	v := hostRAMAvailable()
	availProbeValue = v
	availProbeAt = time.Now()
	return v
}

// resetAvailProbeCache invalidates the cache immediately — test-only, called when a test swaps
// hostRAMAvailable to a new fake value so the swap takes effect on the very next read.
func resetAvailProbeCache() {
	availProbeMu.Lock()
	availProbeAt = time.Time{}
	availProbeMu.Unlock()
}

// AdmitPrefillMemory is the request-time counterpart to fitguard.go's load-time guard. The load-time guard prices the worst case a
// request could reach (the model's own maximum context) once, at load; an unpinned load can still be handed a request whose actual
// prompt is enormous (a real agent's system prompt plus its tool schema runs to tens of thousands of tokens). This runs before
// prefill begins (internal/serveapp calls it right after `prepare` resolves the prompt length and clamped max_tokens, for every
// endpoint that reaches prefill) and prices KV(promptTokens+maxTokens) plus prefill scratch against what remains of currently
// available memory, never starting a prefill that would page. It returns a descriptive error when the request would not fit, and
// nil when it fits, when availability is unknown, or when GOINFER_NO_FIT_GUARD disables the check. residentPath says the request
// runs the stateless GPU-resident path, which allocates no host KV.
//
// It is priced against currently available memory, not a fraction of total RAM: a fixed fraction of total RAM assumes nothing else
// on the machine needs more than the remainder, which is false on a shared machine. Weights are not subtracted (unlike the
// load-time guard): at request time the model is resident, so the available figure already excludes its footprint and subtracting
// again would double-count. Available memory is read through cachedHostRAMAvailable (a short TTL cache).
//
// It runs for CPU and Metal-resident alike: Metal's own residency guard prices weights only and has no per-request check, so this
// is additive to it.
func (m *Model) AdmitPrefillMemory(promptTokens, maxTokens int, residentPath bool) error {
	return m.AdmitPrefillMemoryShare(promptTokens, maxTokens, residentPath, 1)
}

// AdmitPrefillMemoryShare is AdmitPrefillMemory for a server running up to share generations of this model at once (serve
// -max-concurrent): each request must fit in 1/share of the safety margin, because concurrent prefills all draw on the same
// available memory, which each one's own check reads before any has allocated. share <= 1 is AdmitPrefillMemory exactly.
func (m *Model) AdmitPrefillMemoryShare(promptTokens, maxTokens int, residentPath bool, share int) error {
	if m == nil || m.w == nil || m.knobs.get(knobNoFitGuard) != "" {
		return nil
	}
	avail := cachedHostRAMAvailable()
	if avail <= 0 {
		return nil // unknown ⇒ proceed, the same principle the load-time guard uses
	}
	remaining := int64(float64(avail) * prefillAvailFraction / float64(max(1, share)))

	cfg := m.Config()
	positions := promptTokens + maxTokens
	// A request on the stateless GPU-resident path (internal/serveapp's residentPath: false for vision and adapter requests, which
	// stay on the CPU/staged session path and need this term) never allocates the host KV priced here: generateInto allocates one
	// lazily, only on the CPU fallback. Pricing it anyway could reject a request an 8 GB CUDA box would have served entirely in VRAM.
	var kv int64
	if !(residentPath && m.ResidentActive()) {
		kv = estimateKVBytes(cfg, positions, m.kvF16, m.kvI8)
	}
	scratch := prefillScratchBytes(cfg, promptTokens)
	need := kv + scratch
	if need <= remaining {
		return nil
	}
	return fmt.Errorf(
		"decoder: this request needs ~%.2f GB (KV %.2f GB + prefill scratch %.2f GB for %d prompt + %d max_tokens positions) "+
			"but only %.2f GB of this machine's %.2f GB currently-available memory would be left as a safety margin — "+
			"rejected before prefill rather than paging.\n"+
			"  Send a smaller prompt, lower max_tokens, pass -ctx to cap the context, close other "+
			"applications to free memory, or -stream-weights to free more of the budget for KV.\n"+
			"  Set GOINFER_NO_FIT_GUARD=1 to allow it anyway if this machine really has the room",
		float64(need)/fitGB, float64(kv)/fitGB, float64(scratch)/fitGB, promptTokens, maxTokens,
		float64(remaining)/fitGB, float64(avail)/fitGB)
}

// FitBudgetSummary reports the numbers the load banner states: the context KV is priced at (what the load-time guard actually
// used: the pin, the auto-pinned cap, or the model's own maximum), KV bytes at that context, the resident weight bytes, and the
// memory budget. known=false when availability or the model's config was not readable, matching the guard's "unknown means say
// nothing" rule (a banner line with half its numbers missing is worse than none).
//
// budgetBytes is priced against currently available memory, read at call time. By then the model is loaded, so weightBytes is
// already excluded from availability by the OS's own accounting: the caller (internal/serveapp/banner.go) must not subtract
// weightBytes from budgetBytes again. It is returned for display only.
func (m *Model) FitBudgetSummary() (ctx int, kvBytes, weightBytes, budgetBytes int64, known bool) {
	if m == nil || m.w == nil {
		return 0, 0, 0, 0, false
	}
	avail := cachedHostRAMAvailable()
	weightBytes = m.ResidentWeightBytes()
	if avail <= 0 || weightBytes <= 0 {
		return 0, 0, 0, 0, false
	}
	cfg := m.Config()
	budgetBytes = int64(float64(avail) * fitMemFraction)
	// A resident holds its own KV: its capacity, at the precision its backend allocates (Metal: f16 whatever -kv says), not the CPU's
	// per-request ceiling over the model's whole window, which badly overstates it.
	if capper, ok := m.resident.(ResidentCapped); ok && m.be != nil {
		if c := capper.ContextCap(); c > 0 {
			return c, m.ResidentKVBytes(m.be.Name(), c, m.kvF16, m.kvI8), weightBytes, budgetBytes, true
		}
	}
	ctx = m.resCtxReq
	if ctx <= 0 {
		if cfg == nil || cfg.MaxPositions <= 0 {
			return 0, 0, 0, 0, false
		}
		ctx = cfg.MaxPositions
	}
	kvBytes = estimateKVBytes(cfg, ctx, m.kvF16, m.kvI8)
	return ctx, kvBytes, weightBytes, budgetBytes, true
}

// prefillScratchBytes is a stated approximation, not a full accounting of every backend's prefill scratch. Two terms:
//
//  1. Attention scratch: prefillAttnScratchBudget (scratch.go, 256 MiB) is a real, already-enforced hard cap (the batched-prefill
//     worker pool throttles its slot count to stay inside it, prefillAttnWorkersK), so this term is never more than the prefill
//     path already allows itself.
//  2. MLP/batched-matmul scratch, the dominant term that budget does not cover: gate/up activations for a batched (M-token)
//     sweep, 2×IntermediateDim×M float32 (f32 regardless of weight quant). The int8 activation buffer the matmul path pools and
//     reuses (weightmat.go's matmulWSPool) is not counted, being pooled rather than sized per request. This term is a derivable
//     upper bound from the model's dimensions, not a profiled number.
func prefillScratchBytes(cfg *Config, promptTokens int) int64 {
	if cfg == nil || cfg.IntermediateDim <= 0 || promptTokens <= 0 {
		return prefillAttnScratchBudget
	}
	mlp := int64(2*4) * int64(cfg.IntermediateDim) * int64(promptTokens) // gate + up, f32
	return prefillAttnScratchBudget + mlp
}
