package decoder

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// prefillEnters counts every call to (*Model).NewCache — the actual KV allocation a generation
// makes. Mirrors fitguard.go's weightAllocs: it exists so "AdmitPrefillMemory is a pure check,
// never itself the thing that allocates" is an assertion a test can make, not a deduction from
// the absence of a crash.
var prefillEnters atomic.Int64

// prefillAvailFraction is the share of CURRENTLY AVAILABLE memory a request's own KV+scratch may
// consume. Same numeric value as fitguard.go's fitMemFraction, and inherited from it rather than
// independently measured for this specific context (available-RAM headroom, not a fraction of
// total RAM) — stated rather than hidden, pending a real measurement of its own.
const prefillAvailFraction = fitMemFraction

// availProbeTTL rate-limits the CURRENTLY-AVAILABLE memory probe (P-13, audit-2026-09-10):
// hostRAMAvailable shells out (vm_stat on darwin, a /proc/meminfo read on linux) on every call,
// and AdmitPrefillMemory runs it on every request that reaches prefill — a fork+exec per chat
// request on darwin. Available memory does not need sub-250ms freshness for an admission check;
// it changes on the timescale of other processes starting/exiting, not per-request.
const availProbeTTL = 250 * time.Millisecond

var (
	availProbeMu    sync.Mutex
	availProbeValue int64
	availProbeAt    time.Time
)

// cachedHostRAMAvailable is hostRAMAvailable (the raw, uncached, test-injectable indirection)
// behind a short TTL cache. Kept as a separate wrapper rather than folded into hostRAMAvailable
// itself so a test overriding hostRAMAvailable directly (fitguard_test.go, prefill_budget_test.go)
// still sees its own value immediately — resetAvailProbeCache (called from those tests' inject
// helper) clears the cache so a fresh override is never masked by a stale cached read from a
// PRIOR test's value within the same 250ms window.
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

// AdmitPrefillMemory is the request-time counterpart to fitguard.go's load-time guard (R13,
// docs/measurements/cold-user-2026-09-07-macbook-arm64.md). The load-time guard prices the WORST
// CASE a request could reach — the model's own maximum context — once, at load, and either caps
// or refuses on that basis. But an unpinned load whose auto-pinned (or never-needed-a-pin) context
// leaves real headroom can still be handed a request whose actual prompt is enormous: a real
// agent's system prompt plus its full tool schema, tens of thousands of tokens. That is exactly
// what happened on the run that found this — a 7B int4 model the load-time guard rated "79% of
// budget" (KV priced at 0, since nothing was pinned) reached 14 GB RSS and swapped a 16 GB Mac
// hard on its first opencode request, not on load.
//
// PRICED AGAINST CURRENTLY-AVAILABLE MEMORY, NOT A FRACTION OF TOTAL RAM (R13-follow-on, the same
// report's live re-run of this fix). The first version of this function repeated fitguard.go's
// load-time shape — a fixed fraction of TOTAL RAM, minus resident weights — which implicitly
// assumes nothing else running on the machine ever needs more than the remaining fraction. On the
// live re-run, the load-time guard correctly auto-pinned a smaller context and this function
// correctly reported every check as fitting — and `serve check`'s own requests still drove 9.7 GB
// of swap, because "70% of 16 GB total" was never actually free: other processes on a real,
// shared machine were already using more than the remaining 30% assumed available. Weights are
// NOT subtracted here (unlike the load-time guard): at request time the model is already
// resident, so `HostRAMAvailableBytes` already excludes its footprint by construction — subtracting
// it again would double-count. Reads live available memory through cachedHostRAMAvailable (P-13,
// audit-2026-09-10: a 250ms TTL cache, not read fresh on every call as this comment used to say —
// the raw probe forks+execs on darwin, and every request that reaches prefill was paying for one).
//
// This runs BEFORE prefill begins (internal/serveapp calls it right after `prepare` resolves the
// prompt length and clamped max_tokens, for every endpoint that reaches prefill), and prices
// KV(promptTokens+maxTokens) plus prefill scratch against what remains of currently-available
// memory — never starting a prefill that would page. It runs for CPU and Metal-resident alike:
// Metal's own residency guard (metal/backend.go's residentFitsMemory) prices weights only,
// against host RAM (Metal's unified memory IS host RAM), and has no per-request check at all — this
// is additive to it, not a replacement.
func (m *Model) AdmitPrefillMemory(promptTokens, maxTokens int, residentPath bool) error {
	if os.Getenv("GOINFER_NO_FIT_GUARD") != "" {
		return nil
	}
	if m == nil || m.w == nil {
		return nil
	}
	avail := cachedHostRAMAvailable()
	if avail <= 0 {
		return nil // unknown ⇒ proceed, the same principle the load-time guard uses
	}
	remaining := int64(float64(avail) * prefillAvailFraction)

	cfg := m.Config()
	positions := promptTokens + maxTokens
	// P-01 (audit-2026-09-10): a request that will actually run the stateless GPU-resident path
	// (internal/serveapp/openai.go's own residentPath — false for vision and adapter requests,
	// which stay on the CPU/staged session path and DO need this term) never allocates the host
	// KV this prices: generateInto's lazy allocation (P-01's other half, model.go) only
	// constructs one on the CPU fallback. Pricing it here anyway is what could 413 a request an
	// 8 GB CUDA box would have served entirely in VRAM.
	var kv int64
	if !(residentPath && m.ResidentActive()) {
		kv = kvBytesPerPosition(cfg, m.kvF16, m.kvI8) * int64(positions)
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

// FitBudgetSummary reports the numbers R13's banner line states at every load: the context KV is
// priced at (whatever the load-time guard actually used — the pin, the auto-pinned cap, or the
// model's own maximum), KV at that context, the resident weight bytes, and the memory budget.
// known=false when availability or the model's config was not readable, matching the guard's own
// "unknown ⇒ say nothing" rule — a banner line with half its numbers missing is worse than no
// line.
//
// budgetBytes is priced against CURRENTLY AVAILABLE memory (R13-follow-on), read at call time —
// by the time this runs the model is already loaded, so weightBytes is ALREADY excluded from
// availability by the OS's own accounting. The caller (internal/serveapp/banner.go) must NOT
// subtract weightBytes from budgetBytes again when computing what remains — weightBytes is
// returned for DISPLAY only, the same "no double-count" rule prefill_budget.go's
// AdmitPrefillMemory applies at request time.
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
	ctx = m.resCtxReq
	if ctx <= 0 {
		if cfg == nil || cfg.MaxPositions <= 0 {
			return 0, 0, 0, 0, false
		}
		ctx = cfg.MaxPositions
	}
	budgetBytes = int64(float64(avail) * fitMemFraction)
	kvBytes = kvBytesPerPosition(cfg, m.kvF16, m.kvI8) * int64(ctx)
	return ctx, kvBytes, weightBytes, budgetBytes, true
}

// prefillScratchBytes is a stated approximation, not a full accounting — the honest scope cut
// R13 makes rather than block the admission check on a complete scratch-byte model of every
// backend's prefill path. Two terms:
//
//  1. Attention scratch: decoder/scratch.go's prefillAttnScratchBudget (256 MiB) is a REAL,
//     already-enforced hard cap — the batched-prefill worker pool throttles its slot count down
//     to stay inside it (prefillAttnWorkers), so this term can never be more than what the prefill
//     path already allows itself, on any prompt length.
//  2. MLP/batched-matmul scratch: the dominant term prefillAttnScratchBudget does NOT cover —
//     gate/up activations for a batched (M-token) prefill sweep, sized 2×IntermediateDim×M
//     float32 (both buffers, f32 regardless of weight quant — activations are quantized into a
//     separate, smaller int8 buffer the matmul path pools and reuses, decoder/weightmat.go's
//     matmulWSPool, not counted here because it is pooled/reused rather than sized per-request).
//     This is the term this function is honest about NOT having measured: it is a real, derivable
//     upper bound from the model's own dimensions, not a number read off a profiler.
func prefillScratchBytes(cfg *Config, promptTokens int) int64 {
	if cfg == nil || cfg.IntermediateDim <= 0 || promptTokens <= 0 {
		return prefillAttnScratchBudget
	}
	mlp := int64(2*4) * int64(cfg.IntermediateDim) * int64(promptTokens) // gate + up, f32
	return prefillAttnScratchBudget + mlp
}
