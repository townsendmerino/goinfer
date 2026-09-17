//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package (under a build tag) from a main to enable `--backend metal`. Dense residency only
// (Qwen2/Llama, DecodeRunnerEligible); declines gracefully to the staged/CPU path otherwise.

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"golang.org/x/sys/unix"
)

func init() {
	decoder.RegisterBackend("metal", func() (decoder.Backend, error) {
		return &metalBackend{}, nil
	})
	// residentFitsMemory's own budget, exposed to decoder.Model.Plan (docs/tasks/task-fit-to-hardware.md
	// Phase 1) via the SAME arithmetic — not a live "available" query, deliberately: darwin's UBC
	// reclaim makes "available" report what survived rather than what can be asked for (the
	// residentMemFraction comment above this file's own guard). Plan sees exactly the number the
	// real guard would judge it against, so the two can never disagree.
	decoder.RegisterMemoryProbe("metal", func() (int64, bool) {
		ram, err := unix.SysctlUint64("hw.memsize")
		if err != nil || ram == 0 {
			return 0, false
		}
		return int64(float64(ram) * residentMemFraction), true
	})
}

// Compile-time seams: catch signature drift against decoder/residency.go + decoder/backend.go.
var (
	_ decoder.Backend             = (*metalBackend)(nil)
	_ decoder.ResidencyBackend    = (*metalBackend)(nil)
	_ decoder.ResidentForward     = (*metalResident)(nil)
	_ decoder.ResidentMRoPE       = (*metalResident)(nil)
	_ decoder.ResidentAdapter     = (*metalResident)(nil)
	_ decoder.Prefiller           = (*metalResident)(nil)
	_ decoder.PrefillPathReporter = (*metalResident)(nil)
	_ decoder.VerifyPathReporter  = (*metalResident)(nil)
)

// metalBackend implements decoder.Backend + decoder.ResidencyBackend.
type metalBackend struct {
	resident *metalResident // set by BuildResident; released in Close
}

func (b *metalBackend) Name() string { return "metal" }

// MatmulBT is the staged (non-resident) path — prefill matmuls, non-dense families, or a
// no-GPU fallback — dispatched to the shared SIMD linalg kernels (same as the CPU backend),
// so `--backend metal` stays correct even when the resident GPU path declines.
func (b *metalBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, bmat, dst, M, K, N)
}

// BuildResident builds the resident Metal decoder from a loaded dense Model and wraps it in
// an adapter satisfying decoder.ResidentForward. Never crashes the process: BuildResident
// compiles MSL / creates the device and panics on failure — recover → decline (ok=false) →
// the decoder falls back to the staged/CPU path. Callers gate on DecodeRunnerEligible first;
// weights load as either int4 or int8 (Options{Quant:"int4"/"int8int8"}) — an f32 projection
// declines. G10 (docs/tasks/task-gpu-paths-2026-09.md): this backend has NO int8 GEMV kernel at all —
// int4Buf (model.go) re-quantizes an int8-loaded weight through the SAME W4A8 packer an int4
// load uses, so `--quant int8int8` on Metal runs int4 numerics on the GPU while holding the int8
// host copy (more RAM, not more precision). decoder.Model.DecodePath() reports this honestly
// rather than repeating the requested quant string as if it were what actually executes.
func (b *metalBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	defer func() {
		if p := recover(); p != nil {
			fmt.Fprintf(os.Stderr, "[metal] BuildResident declined: %v\n", p)
			rf, ok, err = nil, false, nil
		}
	}()
	// Admission check: DecodeRunnerEligible was scoped to the richer WebGPU/CUDA runner, which
	// admits QK-norm / sliding-window / partial-rotary and more. The Metal kernel set implements
	// only a subset (decoder.ResidentBackendFeatures("metal")); anything it doesn't implement must
	// DECLINE (→ correct CPU fallback) rather than run with the feature silently dropped. The
	// subset check uses the shared taxonomy (one source of truth; a new arch classifies itself).
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("metal")); len(missing) > 0 {
		fmt.Fprintf(os.Stderr, "[metal] declined — unimplemented features: %v\n", missing)
		return nil, false, nil
	}
	// FITS-IN-MEMORY GUARD. Metal's unified memory IS host RAM, and it WIRES the mmap pages a
	// command buffer touches, so a model whose weights exceed RAM does not merely run slowly —
	// it pages until swap is exhausted. Measured 2026-08-31 on a 16 GB MacBook with gpt-oss-20b
	// (11.28 GB of weights): swap went to 35.98 GB of 36 GB, the process sat in uninterruptible
	// I/O wait with RSS creeping 1.8 -> 2.0 GB over 12 minutes, and the load NEVER completed and
	// never declined. Declining is strictly better than that, and there was no check at all: the
	// only size guard here caps the KV CONTEXT (checkCap, above), not the weights.
	//
	// Keyed on two quantities WE compute — the model's own weight bytes and the machine's
	// physical RAM — never on the OS's account of what is free. Darwin's UBC reclaims under
	// pressure, so "available" reports what survived rather than what can be asked for; an
	// RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline,
	// which is a guard that inverts exactly when it is needed.
	// G6 (docs/tasks/task-gpu-paths-2026-09.md — "honoured or refused with numbers"): an explicit -ctx
	// above metalCtxCapMax cannot be honoured (a fixed-size kernel score buffer, not a tunable
	// budget) and must be a NAMED refusal, not folded into the generic "BuildResident declined"
	// swallow below (buildResident itself also calls resolveMetalCtxCap and would hit the exact
	// same error, but by then it's just another opaque build failure) — mirrors CUDA's own
	// errKVWontFit precedent (cuda/backend.go): an operator's explicit request that cannot be
	// honoured gets a specific, propagated error naming the numbers, not a silent CPU fallback
	// indistinguishable from "this arch doesn't fit here at all".
	if _, cerr := resolveMetalCtxCap(m); cerr != nil {
		fmt.Fprintf(os.Stderr, "[metal] %v\n", cerr)
		return nil, false, cerr
	}
	if !residentFitsMemory(m) {
		return nil, false, nil
	}
	res, e := buildResident(m)
	if e != nil {
		fmt.Fprintf(os.Stderr, "[metal] BuildResident declined: %v\n", e)
		return nil, false, nil
	}
	b.resident = &metalResident{r: res, hidden: res.H}
	return b.resident, true, nil
}

// residentMemFraction is the share of physical RAM the WEIGHTS alone may occupy. The remainder
// is not slack: the KV cache, per-layer scratch, the command buffers, and the rest of the system
// all live in the same unified memory.
//
// 0.70 is set from ONE measured failure (11.28 GB of 16 GB = 70.5% thrashed to swap exhaustion)
// and is therefore a threshold, not a curve — it is honestly a single point, and a machine that
// would in fact have fit can override with GOINFER_NO_RESIDENT_MEM_GUARD=1 rather than be told
// no by a number nobody has swept. What it must not do is silently pass the case it was written
// for, which is why the bar sits just below that measurement rather than at a rounder 0.75.
const residentMemFraction = 0.70

// fitsResidentBudget is the arithmetic alone, split out so it can be driven with the numbers
// from the measurement instead of requiring a 12 GB checkpoint to exercise the guard.
func fitsResidentBudget(need int64, ram uint64) bool {
	if need <= 0 || ram == 0 {
		return true // unknown ⇒ do not refuse
	}
	return uint64(need) <= uint64(float64(ram)*residentMemFraction)
}

// metalMoESlotsRequest is the resolved expert-slot request as a string, ready for the SAME
// strconv.Atoi + validation metal/moe.go and metal/gemma4_moe.go already do at their real
// dispatch-building call sites: `--moe-cache-slots` / decoder.Options.MoECacheSlots
// (m.MoECacheSlotsRequest(), the SAME flag CUDA's own auto-cap-to-VRAM already reads) wins when
// set (Phase 2, docs/tasks/task-gpu-paths-2026-09.md — "Metal slots become an Option and a flag");
// GOINFER_METAL_MOE_SLOTS is kept as a deprecated fallback for anyone still setting it directly.
// "" means unset either way — n==0/unset ⇒ every expert resident, today's behavior, unchanged.
func metalMoESlotsRequest(m *decoder.Model) string {
	if n := m.MoECacheSlotsRequest(); n > 0 {
		return strconv.Itoa(n)
	}
	// M-13 (audit-metal-2026-09-12.md): --moe-cache-experts alone (no explicit --moe-cache-slots)
	// used to read as "0 ⇒ unpaged" here, so a user following the CLI's own advice for a
	// bigger-than-RAM MoE (26B/35B/gpt-oss-20b) got every-expert-resident anyway, which the memory
	// guard then declined to the CPU-staged path the peer-matrix row actually measured. CUDA
	// already auto-caps to free VRAM in this exact shape (MoECacheExperts set, slots unset); this
	// is Metal's twin.
	if m.MoECacheExperts() {
		return strconv.Itoa(autoMoESlots(m))
	}
	return os.Getenv("GOINFER_METAL_MOE_SLOTS")
}

// moeTopK is this model's own top-k routed-experts-per-token count — the floor autoMoESlots must
// clamp above, since fewer slots than top-k cannot hold one token's own routed set simultaneously
// (the same floor buildResident enforces on an explicit --moe-cache-slots, gated by
// TestMoESlotsViaOptions_belowTopKRefusesWithNumbers). Gemma-4's MoE and the generic MoE families
// store their top-k under different Config fields (registry.go's own per-architecture TopK:
// assignments), so this branches on which shape the model actually is rather than guessing one
// field name works for both.
func moeTopK(m *decoder.Model) int {
	if m.HasGemma4MoEResident() {
		if k := m.Config().TopKExperts; k > 0 {
			return k
		}
		return 1
	}
	if _, k, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams(); ok && k > 0 {
		return k
	}
	return 1
}

// autoMoESlotsMax is the per-layer expert-slot ceiling this auto-sizer will request even when
// memory allows more — task-metal-expert-streaming-at-scale.md's own measurement run settled on
// N=64 as the recommended default; raise it only if a future measurement moves that number.
const autoMoESlotsMax = 64

// autoMoESlotsFor is autoMoESlots' pure formula, split out (fitsResidentBudget's own pattern, and
// residentNeedBytes' own reason for existing) so it can be unit-tested directly against a
// deliberately small ram value — a real machine holding less RAM than a real MoE model's dense
// term is not available to test against otherwise.
//
// Solves the same inequality residentFitsMemory's guard checks (0.7·RAM ≥ need) for N instead of a
// pass/fail: need(N) = needFixed + N·perSlot, where needFixed is everything that does NOT scale
// with the slot count (dense weights doubled by Metal's host-copy-plus-device-buffer footprint,
// plus KV) and perSlot is one additional expert slot's marginal bytes. Clamped to
// [topK, autoMoESlotsMax]: below topK a token's own routed set cannot fit simultaneously (the same
// floor buildResident enforces on an explicit --moe-cache-slots); autoMoESlotsMax is
// task-metal-expert-streaming-at-scale.md's own measured recommendation. perSlot<=0 or a
// non-positive remaining budget both fall back to a safe default (autoMoESlotsMax or topK
// respectively) rather than a fabricated small number — the memory guard downstream still has the
// final say either way.
func autoMoESlotsFor(topK int, perSlot, needFixed int64, ram uint64) int {
	if perSlot <= 0 {
		return max(topK, autoMoESlotsMax)
	}
	budget := int64(float64(ram) * residentMemFraction)
	remaining := budget - needFixed
	if remaining <= 0 {
		return topK // the fixed part alone doesn't fit; buildResident/the guard will refuse either way
	}
	n := int(remaining / perSlot)
	return max(topK, min(n, autoMoESlotsMax))
}

// autoMoESlots derives a per-layer expert-slot count from live memory for a model that asked to
// stream MoE experts (MoECacheExperts) but gave no explicit slot count (M-13). perSlotBytes is
// read from the SAME byte accessor residentNeedBytes already calls, at the marginal cost of one
// additional slot (ResidentWeightBytesPaged(2)-ResidentWeightBytesPaged(1)) rather than re-derived
// by hand — a layer's experts are uniform in shape (that accessor's own doc comment), so the
// marginal slot cost is exact, not approximated, and stays correct if the byte accounting itself
// ever changes. An unreadable hw.memsize falls back to autoMoESlotsMax (see autoMoESlotsFor).
func autoMoESlots(m *decoder.Model) int {
	topK := moeTopK(m)
	perSlot := m.ResidentWeightBytesPaged(2) - m.ResidentWeightBytesPaged(1)
	ram, err := unix.SysctlUint64("hw.memsize")
	if err != nil || ram == 0 {
		return max(topK, autoMoESlotsMax)
	}
	needFixed := m.ResidentDenseWeightBytes() + m.ResidentHostCopyBytes(1) + residentKVBytes(m) // ResidentHostCopyBytes(>0) is dense-only (paged)
	return autoMoESlotsFor(topK, perSlot, needFixed, ram)
}

// metalMoESlotsFromEnv is the guard's own reader (residentNeedBytes, below) — it needs an int,
// not a validated dispatch-ready string, and unlike the two real call sites an invalid or unset
// value is not an error here: it just means "assume unpaged" (the guard's existing, safe
// behavior); buildResident itself still validates and declines on a bad value. Despite the name
// (kept for now — see metalMoESlotsRequest's own doc comment on why the underlying knob is no
// longer env-only), this reads the SAME resolved request metalMoESlotsRequest does.
func metalMoESlotsFromEnv(m *decoder.Model) int {
	n, err := strconv.Atoi(metalMoESlotsRequest(m))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// residentKVBytes is the resident KV cache's footprint: the resolved ctx cap × kvDim × 2 bytes
// (f16 KV — the only path this backend ships; the f32 KV kernels exist but are compiled out,
// metal/model.go's kvF32 hardcoded false since the Gemma "crater" traced to BOS K/V, not
// precision) × 2 buffers (K and V), summed per layer since a per-layer geometry family (Gemma 4)
// varies kvDim between local and global layers — a single model-level figure would misprice
// whichever shape isn't the majority. This is the dominant "scratch" term M-02 named as missing;
// the many small per-model buffers (r.mq/r.gu/r.logits/etc, metal/model.go) are each at most a
// few hundred KB — elementwise or vocab/hidden-sized — and round to nothing beside it, the same
// exemption ResidentWeightBytes' own doc comment already gives norms/biases.
//
// Runs BEFORE buildResident (this is the pre-build guard), so there is no real *resident yet to
// read ctxCap off of — resolveMetalCtxCap(m) is called independently here. An error (an explicit
// ctx above metalCtxCapMax) is NOT this function's problem to report: residentFitsMemory's caller
// (metalBackend.BuildResident) checks that specific case on its own, earlier and more clearly, so
// this just falls back to metalCtxCapMax on error — a safe over-estimate for a guard whose whole
// job is "don't under-count", never the thing that actually explains the refusal to the user.
func residentKVBytes(m *decoder.Model) int64 {
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		ctxCap = metalCtxCapMax
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	bytesPerElem := int64(2) // f16; see comment above
	if m.KVCacheI8() {
		bytesPerElem = 1 // int8 KV: 1 byte/elem + per-head scales
	}
	// N-36 (audit-metal-2026-09-12.md): a Gated-DeltaNet layer (qwen3_5/qwen3_5_moe/qwen3_next's
	// linear-attention layers) has no attention geometry at all — no q/k/v/o, no KV cache;
	// metal/model.go's own buildResident comment is explicit that r.kc[l]/r.vc[l] stay zero-value
	// for these layers. Charging them the model's default kvDim anyway overstates this guard's
	// estimate on every DeltaNet-hybrid model (qwen3_5_moe is 3:1 linear:softmax), in the
	// conservative direction (a guard that overcounts can only decline early, never admit a model
	// that doesn't fit) but still wrong — the same chokepoint metal/model.go's own layer-build loop
	// skips past.
	_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
	var total int64
	for l := 0; l < nLayers; l++ {
		if dnetOK && m.Qwen35LinearLayer(l) {
			continue
		}
		kvDim := int64(m.KVHeadsAtResident(l)) * int64(m.HeadDimAtResident(l))
		total += 2 * int64(ctxCap) * kvDim * bytesPerElem // ×2 for K and V
		if m.KVCacheI8() {
			total += 2 * int64(ctxCap) * int64(m.KVHeadsAtResident(l)) * 4 // ×2 for K scale and V scale (f32)
		}
	}
	return total
}

// residentNeedBytes is the byte count residentFitsMemory judges against — split out from
// residentFitsMemory so it can be unit-tested directly, without real RAM or a checkpoint large
// enough to swing the guard's verdict.
//
// M-02: this used to always be m.ResidentWeightBytes() — the UNPAGED number — even when the
// caller had set GOINFER_METAL_MOE_SLOTS, so a model that fits fine under paging (a few GB) was
// declined on the number it would need with every expert resident (tens of GB for a large MoE).
// Reading the same knob buildResident is about to honor and asking for the PAGED estimate instead
// fixes the audit's Qwen3.5-35B-A3B example without moving the guard itself — it still runs
// before buildResident, on the byte count that will actually apply once paging is resolved.
//
// M-02 (continued, 2026-09-09): the weight term alone under-counted by ~2x on a real GGUF/
// safetensors load (docs/audit-2026-09-02.md's "~9 GB Q4_K_M GGUF... lands at ~18 GB anonymous"
// example) because Metal's unified memory holds the quantized HOST WeightMat AND a freshly
// re-packed device buffer for the same weights (decoder.Model.ResidentHostCopyBytes' own doc
// comment). Genuinely paged experts are exempt (they stream, no host copy), which is why this
// asks for the host-copy addend at the SAME slot count rather than assuming it doubles the whole
// weight term. KV was entirely absent; residentKVBytes above closes that.
func residentNeedBytes(m *decoder.Model) int64 {
	slots := metalMoESlotsFromEnv(m)
	return m.ResidentWeightBytesPaged(slots) + m.ResidentHostCopyBytes(slots) + residentKVBytes(m)
}

// residentFitsMemory reports whether this model's weights fit the machine, declining loudly when
// they do not. True (proceed) whenever the answer is unknown — an unreadable hw.memsize or a
// model reporting zero bytes must not silently disable residency for everyone.
func residentFitsMemory(m *decoder.Model) bool {
	if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		return true
	}
	need := residentNeedBytes(m)
	if need <= 0 {
		return true // nothing to compare against; not a reason to refuse
	}
	ram, err := unix.SysctlUint64("hw.memsize")
	if err != nil || ram == 0 {
		return true
	}
	if fitsResidentBudget(need, ram) {
		return true
	}
	budget := uint64(float64(ram) * residentMemFraction)
	const gb = 1 << 30
	// M-13: name the actual escape hatch for an MoE model, not just the guard override — a model
	// with routed experts that doesn't fit resident may still fit PAGED, and the decline line
	// used to say nothing about how to reach that path.
	moeHint := ""
	if m.HasGemma4MoEResident() {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	} else if _, _, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams(); ok {
		moeHint = " This model routes MoE experts — try --moe-cache-experts (Metal pages them; auto-sizes the slot count from free RAM) or --moe-cache-slots N to pick one yourself."
	}
	fmt.Fprintf(os.Stderr, "[metal] declined — weights %.2f GB exceed %.0f%% of %.1f GB RAM "+
		"(budget %.2f GB). Metal wires the pages it touches, so loading this would page to swap "+
		"exhaustion rather than run; continuing on the CPU/staged path. Override with "+
		"GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.%s\n",
		float64(need)/gb, residentMemFraction*100, float64(ram)/gb, float64(budget)/gb, moeHint)
	return false
}

func (b *metalBackend) Close() error {
	if b.resident != nil {
		return b.resident.Close()
	}
	return nil
}

// metalResident adapts *resident (whose Forward takes a token id and returns logits, no error)
// to decoder.ResidentForward (Forward takes a precomputed embedding and returns logits+error).
type metalResident struct {
	r      *resident
	hidden int
}

// ctxCap is this resident's resolved KV capacity — a.r.ctxCap when a real *resident exists, else
// metalCtxCapMax. The fallback matters for TestMetalResidentCheckCap/TestMetalCtxCapWithinKernelBound
// (metal/resident_cap_test.go), which deliberately construct a zero-value &metalResident{} (r ==
// nil) to test checkCap/ContextCap as pure logic with no Metal device — those tests predate G6's
// per-build ctxCap and are meant to keep working unmodified against "the historical constant"
// semantics, so a nil/zero r reads as "no explicit request was ever resolved here", not as 0.
func (a *metalResident) ctxCap() int {
	if a.r == nil || a.r.ctxCap == 0 {
		return metalCtxCapDefault
	}
	return a.r.ctxCap
}

// checkCap guards the resident KV allocation (C3). Every layer's cache is r.kc[l]/r.vc[l], sized
// ctxCap()*kvDim, so kv_store writes absolute position p at kc[p*kvDim ...]; valid positions
// are [0, ctxCap()). Writing past it is an out-of-bounds device write — on Metal's UNIFIED
// memory that silently corrupts adjacent MTLBuffers (other models' resident weights), and once
// nKeys > 4096 the attention kernel's `threadgroup float sc[4096]` overflows too. The decode loop
// increments pos unbounded (a ≤cap prompt + a large max_tokens is enough), so refuse here; the
// decode loop surfaces the error (model.go) and the caller can fall back to the staged path.
func (a *metalResident) checkCap(pos, n int) error {
	c := a.ctxCap()
	if pos < 0 || pos+n > c {
		return fmt.Errorf("metal: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, c)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions. Implementing it makes metalResident satisfy
// decoder.ResidentCapped, so generateInto clamps maxTokens to the cap UP FRONT (stops cleanly at
// the cap instead of erroring mid-decode). Queryable so callers clamp rather than discover mid-run.
func (a *metalResident) ContextCap() int { return a.ctxCap() }

// ForwardMRoPE is decoder.ResidentMRoPE: like Forward, but the rotation angle (ropePos) and the
// KV-cache/attention position (pos) are supplied separately — Qwen2.5-VL decode past an image
// block needs them to differ. Forward itself is ForwardMRoPE(pos, pos).
func (a *metalResident) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if len(embedding) != a.hidden {
		return nil, fmt.Errorf("metal: embedding len %d != hidden %d", len(embedding), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero any Gated-DeltaNet state before it carries over from a prior
		// Generate on this resident (audit C-01's CUDA analogue). No-op for every other family.
		a.Reset()
	}
	logits := a.r.ForwardEmbMRoPEPipe(embedding, pos, ropePos)
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09: a command buffer aborted — surface it, do NOT return stale logits
	}
	return logits, nil
}

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	return a.ForwardMRoPE(embedding, pos, pos)
}

// ForwardNoLogits (decoder.ResidentPrefillKV) runs the token's forward to build ONLY its
// resident K/V — skipping the final norm's LM-head dispatch, the ~1 MB logits readback, and any
// softcap. residentPrefillSeed calls this for every prompt token but the last (audit-
// metal-2026-09-12.md M-01): before this existed every sequential-prefill token on Metal paid the
// full int8 LM head + a 608 KB readback for logits nobody read (233 MB/token on a 1.5B, 671 MB on
// a Gemma-class vocabulary). The layer chain — hence the K/V written at pos — is identical to
// Forward, so decode from the last prompt token is byte-identical.
//
// Pipelined through the encode-ahead executor (ForwardEmbNoLogitsPipe) via a noHead bit on execJob:
// overlaps token t+1's trunk encode with token t's GPU execution while skipping the LM head
// dispatch and logits readback (~0.9 ms/token prefill latency recovery).
//
// On a paged MoE (g4moe or generic moe), forwardHiddenNoHead's trunk encoder has no paged branch
// — only Forward → forwardLogitsPaged/forwardLogitsMoEPaged does (the same gap C-02 found at
// HiddenLast/ForwardArgmax) — so this falls back to the full head-bearing Forward and discards
// the logits: correct K/V either way, just without the head-skip win on that family.
func (a *metalResident) ForwardNoLogits(embedding []float32, pos int) error {
	if len(embedding) != a.hidden {
		return fmt.Errorf("metal: embedding len %d != hidden %d", len(embedding), a.hidden)
	}
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		_, err := a.Forward(embedding, pos)
		return err
	}
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	if pos == 0 {
		a.Reset() // fresh sequence — same reason as Forward
	}
	a.r.ForwardEmbNoLogitsPipe(embedding, pos)
	return a.r.takeExecErr() // C-09: a command buffer aborted — surface it
}

// metalFastPrefillFloor is the PROMPT-LENGTH floor (whole prompt = startPos+M) below which
// the f16-MMA batched pass declines and the sequential per-token loop runs instead. Lowered to
// 256 (2026-09-12, audit-metal-2026-09-12.md M-02): the K=256 "expected to fail §3.2" this floor
// was originally set against is CUDA's own combined L2+L3 result, not Metal's — Metal's K=256
// decision cell has PASSED on both the ref-B gate (docs/measurements/prefill-gate-l1-ref-
// b-2026-09-09.md) and the fused-attention gate (prefill-l2-metal-fused-attn-2026-09-09.md).
// Going lower needs one more passing decision cell at the new depth (the harness already
// parameterises FLOOR via GOINFER_METAL_FAST_PREFILL_FLOOR, 0 = no floor).
const metalFastPrefillFloor = 256

// metalFastPrefillEnabled reports whether the batched f16-MMA prefill path is selected.
//
// Default ON above metalFastPrefillFloor (256 tokens, M-02) since §3.2 gate (TestPrefillGateVsReference)
// passed 2026-09-09 (S model, K=256/512/1024). GOINFER_METAL_FAST_PREFILL=0/false/off or
// --exact-prefill to opt out.
//
//	GOINFER_METAL_FAST_PREFILL  1 | true | on   on (even below the floor — for tests)
//	                            0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
//
// The old GOINFER_METAL_BATCHED_PREFILL continues to work: =1 forces on, =0 forces off, unset defers to the default.
func metalFastPrefillEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_METAL_FAST_PREFILL"))) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	// Unset: honour the old var for backward compat (=1 on, =0 off, unset → new default).
	if v := strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_METAL_BATCHED_PREFILL"))); v != "" {
		return v == "1"
	}
	return true // §3.2 gate passed 2026-09-09 (S cells K=256/512/1024; D7 skipped — fit-guard on 16GB)
}

// metalFastPrefillFloorFor returns the prompt-length floor, allowing experiment or escape.
// Set GOINFER_METAL_FAST_PREFILL_FLOOR to override; 0 disables the floor entirely.
func metalFastPrefillFloorFor() int {
	if v := os.Getenv("GOINFER_METAL_FAST_PREFILL_FLOOR"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return metalFastPrefillFloor
}

// metalFusedAttentionEnabled reports whether attention_prefill_fused (the simdgroup_matrix
// flash-attention twin of attention_prefill, L2-Metal — docs/completed/task-prefill-gap.md §4) runs in
// place of the exact scalar kernel. Default ON since §3 gate passed 2026-09-10 (S model, set B
// decision cells K=256/512/1024 + K=3900 confirm; fused beat exact on all three pooled criteria —
// see docs/measurements/prefill-l2-metal-fused-attn-2026-09-09.md §5). GOINFER_METAL_FUSED_ATTENTION=0
// or --exact-prefill (which also covers metalFastPrefillEnabled) opts back to the exact kernel.
// attention_prefill_fused also requires hd%8==0 && hd<=128 (ATTN_MAXHD in metal/prefill.go) —
// PrefillLast falls back to the exact kernel outside that range regardless of this flag.
//
//	GOINFER_METAL_FUSED_ATTENTION  1 | true | on   on
//	                               0 | false | off  off (explicit opt-out; use --exact-prefill on the server)
func metalFusedAttentionEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_METAL_FUSED_ATTENTION"))) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	return true // §3 gate passed 2026-09-10 (S set B decision cells K=256/512/1024 + K=3900 confirm)
}

// PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident will use
// the batched f16-MMA path. Default ON above metalFastPrefillFloor (256 tokens, M-02) since §3.2
// gate passed 2026-09-09. The floor applies per-call; PrefillPath reports true iff the enabled
// state AND arch both allow batching.
func (a *metalResident) PrefillPath() (bool, string) {
	if !a.r.prefillOK {
		return false, "sequential — arch/geometry not supported by f16 MMA prefill kernel"
	}
	if !metalFastPrefillEnabled() {
		return false, "sequential — fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill)"
	}
	floor := metalFastPrefillFloorFor()
	if floor > 0 {
		return true, fmt.Sprintf("batched f16-MMA above %d prompt tokens; sequential below (§3 floor)", floor)
	}
	return true, "batched f16-MMA (GOINFER_METAL_FAST_PREFILL_FLOOR=0; §3.2 gate passed 2026-09-09)"
}

// PrefillLast (decoder.Prefiller) ingests the whole prompt in one batched f16-MMA pass and
// returns the last token's logits, populating the resident KV. Falls back (declines) for prompts
// shorter than the fast-prefill floor or longer than the resident KV/attention cap.
func (a *metalResident) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	// One pass, so one check: this backend ingests the whole prompt in a single command buffer and
	// has no inner loop to interrupt. Checking at entry is therefore the ONLY granularity available
	// here, and it is honest about that rather than pretending finer. A Metal prefill that wants
	// mid-pass cancellation needs the chunking cuda has (prefillChunked), which is its own change.
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	// DEFAULT ON above metalFastPrefillFloor (256 tokens, M-02) since §3.2 gate passed 2026-09-09
	// (S cells K=256/512/1024). GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill to opt out.
	if !metalFastPrefillEnabled() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill / GOINFER_METAL_BATCHED_PREFILL=0); using sequential path")
	}
	// FLOOR: below metalFastPrefillFloor no decision cell has passed yet. Sequential path there;
	// fast path only where the gate cleared (K=256 itself passed §3.2 on Metal — see the floor's
	// own doc comment).
	promptLen := startPos + len(embeddings)
	if floor := metalFastPrefillFloorFor(); floor > 0 && promptLen < floor {
		return nil, fmt.Errorf("metal: prompt too short (%d tokens) for fast prefill (floor=%d; §3 floor); using sequential path", promptLen, floor)
	}
	// The f16 MMA prefill kernels implement a dense gated FFN (SiLU or GeGLU, G8) out of
	// L.guW/L.dW with per-layer rope/window, per-head QK-norm, and Gemma's sandwich norms, and
	// (G8 MoE half) a generically-shaped gated-SwiGLU MoE FFN — run row by row off the batched
	// residual, reusing the unchanged per-token decode MoE dispatch chain (metal/moe.go) — but
	// NOT per-layer-varying attention geometry (dense Gemma 4's local/global head_dim split;
	// prefillOK's own per-layer-geometry guard, metal/model.go) or Gemma 4's enable_moe_block
	// variant (residLayer.g4moe, a third FFN shape this path never reads; explicitly excluded
	// via HasGemma4MoEResident regardless of per-layer geometry). Any of these declines here,
	// and the caller re-runs the prompt through the (correct) sequential Forward loop.
	if !a.r.prefillOK {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	// startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds — on
	// UMA that silently corrupts adjacent buffers (audit R-27). Unreachable today (the decoder always
	// passes 0) but cheap to guard.
	if startPos < 0 || len(embeddings) == 0 || startPos+len(embeddings) > a.ctxCap() {
		return nil, fmt.Errorf("metal: prompt len %d at startPos %d out of resident cap %d", len(embeddings), startPos, a.ctxCap())
	}
	// ensurePrefill's compile panic and the ~24 per-call MustBuf OOM panics fire HERE, at request
	// time, with no recover of their own (buildResident's is build-scoped). A transient OOM would kill
	// the server; recover into an error so the request fails and the caller falls back to sequential
	// decode (audit R-23; B-10 class).
	var logits []float32
	if err := func() (err error) {
		defer func() {
			if p := recover(); p != nil {
				err = fmt.Errorf("metal: batched prefill aborted: %v", p)
			}
		}()
		logits = a.r.PrefillLast(embeddings, startPos)
		return nil
	}(); err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09
	}
	return logits, nil
}

// HiddenLast (decoder.ResidentHiddenLast) ingests a whole sequence starting at startPos and
// returns the LAST position's hidden state after the model's final norm — the resident twin of
// PrefillLast, but for embedding requests (G4, docs/tasks/task-gpu-paths-2026-09.md) instead of
// generation: it never runs the LM head. This runs the SAME per-token sequential kernels decode
// uses — one forwardHiddenNoHead call per position — which is bit-identical to the CPU reference
// by construction, at the cost of one command-buffer submit per token (≈K × 13-18ms) instead of a
// single batched pass (≈1.8s for K=512).
//
// N-25 (audit-metal-2026-09-12.md): unlike when this doc comment was first written, Metal's
// batched (f16-MMA) PrefillLast is NOT declined by default for generation anymore —
// metalFastPrefillEnabled() defaults true (§3.2 gate passed 2026-09-09) — so the fidelity bar
// that justifies this sequential loop's cost for embeddings is already accepted for decode's own
// output. The audit's suggested fix (PrefillLast's dispatch graph minus its last two dispatches —
// the LM head + softcap — reused here) is a real, scoped lever, not a design question, but it is
// its own parity-gated engineering task (a HiddenLast-batched path needs the same kind of S-cell
// tolerance gate PrefillLast itself passed, verified against THIS function as the oracle) rather
// than a same-sitting fix; left as follow-up work.
func (a *metalResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: HiddenLast called with no embeddings")
	}
	// forwardHiddenNoHead's trunk encoder (encodeTrunkInto → encodeLayer) has no paged branch — only
	// Forward's dispatch to forwardLogitsPaged/forwardLogitsMoEPaged does — so on a paged MoE it would
	// bind the zero-value stacked expert buffers and return a finite garbage hidden state with no
	// error (audit-metal-2026-09-12.md C-02). Decline instead: decoder.Model.HiddenLast treats a
	// resident decline as "no resident backend for this request" and falls through to the CPU path,
	// exactly like an OOM or cap decline does (decoder/embed.go:81-83).
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return nil, fmt.Errorf("metal: HiddenLast not implemented for paged MoE (no headless paged forward); use the CPU path")
	}
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	var out []float32
	for i, emb := range embeddings {
		if len(emb) != a.hidden {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != hidden %d", i, len(emb), a.hidden)
		}
		// G18: an abandoned client otherwise leaves the whole sequence streaming through the
		// device with nothing watching — same discipline as residentPrefillSeed's sequential loop.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		want := i == len(embeddings)-1
		h, err := a.r.forwardHiddenNoHead(emb, startPos+i, want)
		if err != nil {
			return nil, err
		}
		if err := a.r.takeExecErr(); err != nil {
			return nil, err // C-09: a command buffer aborted — surface it, do NOT return a stale/zero vector
		}
		if want {
			out = h
		}
	}
	return out, nil
}

// ForwardN runs a batch of embeddings at consecutive positions (prefill/verify).
// It sequences all N token forward steps inside a SINGLE Metal command buffer
// and single compute encoder (one commit, one wait), returning all N logits vectors.
// For paged MoE (which requires mid-layer host interaction), it falls back to the
// sequential loop.
func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	// Fail-fast before any write: checking the whole batch up front refuses an over-cap run
	// without partial KV writes.
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if len(embeddings) == 0 {
		return nil, nil
	}
	// Fall back to sequential loop for paged MoE where per-layer host interaction is required.
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		out := make([][]float32, len(embeddings))
		for i, emb := range embeddings {
			l, err := a.Forward(emb, startPos+i)
			if err != nil {
				return nil, err
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, nil
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	out, err := a.r.ForwardBatch(embeddings, startPos)
	if err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09: a command buffer aborted — surface it
	}
	return out, nil
}

// VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes
// a single-command-buffer batched pass or falls back to a sequential loop.
func (a *metalResident) VerifyPath() (bool, string) {
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return false, "sequential — paged MoE requires per-layer host staging"
	}
	return true, "batched layer-major single-command-buffer"
}

// UploadKV (prefix-reuse bridge) is not supported: the resident decoder owns its KV writes
// per Forward, and the stateless Generate path re-runs the prompt through Forward instead.
// SetAdapter implements decoder.ResidentAdapter (G3, docs/tasks/task-gpu-paths-2026-09.md) —
// generateInto calls this to bind/clear a compute-time LoRA adapter for an admitted session.
func (a *metalResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return a.r.SetAdapter(layers)
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches at absolute
// position base..base+n-1, where n = len(keys)/kvDim. Used by decoder.residentUploadPrefill
// to push a CPU-computed prefill's KV into the resident GPU cache, letting GenerateVL and
// GenerateQwenVL run text decode on resident GPU rather than dropping to CPU.
func (a *metalResident) UploadKV(layer, base int, keys, vals []float32) error {
	if a.r == nil {
		return fmt.Errorf("metal: resident is nil")
	}
	if layer < 0 || layer >= len(a.r.layers) {
		return fmt.Errorf("metal: UploadKV layer %d out of range", layer)
	}
	if a.r.kc[layer] == (Buffer{}) {
		return nil // recurrent DeltaNet layer with no attention KV cache
	}
	g := a.r.layers[layer].geom
	if g == nil {
		return fmt.Errorf("metal: layer %d has no attention geometry", layer)
	}
	kvDim := g.kvDim
	if kvDim == 0 {
		return fmt.Errorf("metal: layer %d has zero kvDim", layer)
	}
	if len(keys) != len(vals) || len(keys)%kvDim != 0 {
		return fmt.Errorf("metal: UploadKV len(keys)=%d len(vals)=%d not aligned to kvDim=%d", len(keys), len(vals), kvDim)
	}
	n := len(keys) / kvDim
	if e := a.checkCap(base, n); e != nil {
		return e
	}
	off := base * kvDim
	if a.r.kvI8 {
		nKV := g.nKV
		hd := g.hd
		kc := a.r.kc[layer].Int8s()
		vc := a.r.vc[layer].Int8s()
		ks := a.r.ks[layer].Floats()
		vs := a.r.vs[layer].Floats()
		for p := 0; p < n; p++ {
			pos := base + p
			for h := 0; h < nKV; h++ {
				kHead := keys[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				vHead := vals[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				var amaxK, amaxV float32
				for _, v := range kHead {
					if a := float32(math.Abs(float64(v))); a > amaxK {
						amaxK = a
					}
				}
				for _, v := range vHead {
					if a := float32(math.Abs(float64(v))); a > amaxV {
						amaxV = a
					}
				}
				scK := amaxK / 127.0
				if scK == 0 {
					scK = 1.0
				}
				scV := amaxV / 127.0
				if scV == 0 {
					scV = 1.0
				}
				ks[pos*nKV+h] = scK
				vs[pos*nKV+h] = scV
				invK := 1.0 / scK
				invV := 1.0 / scV
				for d := 0; d < hd; d++ {
					kc[pos*kvDim+h*hd+d] = int8(math.Round(float64(kHead[d] * invK)))
					vc[pos*kvDim+h*hd+d] = int8(math.Round(float64(vHead[d] * invV)))
				}
			}
		}
	} else if a.r.kvF32 {
		copy(a.r.kc[layer].Floats()[off:off+len(keys)], keys)
		copy(a.r.vc[layer].Floats()[off:off+len(vals)], vals)
	} else {
		kc := a.r.kc[layer].U16s()[off : off+len(keys)]
		vc := a.r.vc[layer].U16s()[off : off+len(vals)]
		parallelF32ToF16(kc, keys)
		parallelF32ToF16(vc, vals)
	}
	return nil
}

// TruncateTo is a no-op: KV positions are overwritten on write, and attention only reads
// keys[0..pos], so stale positions past the current one are never observed.
func (a *metalResident) TruncateTo(pos int) {}

// Reset zeroes every Gated-DeltaNet layer's causal-conv ring and recurrent matrix state (no-op
// for every other family — a.r.dnet is nil). Unlike KV positions, this state COMPOUNDS: a fresh
// Generate on the same resident without this would continue decaying stale state from the PRIOR
// sequence (audit C-01's CUDA analogue). See resetDeltaNet (deltanet.go).
func (a *metalResident) Reset() { a.r.resetDeltaNet() }

// Close stops the pipelined executor (waiting for it) and frees every MTLBuffer this resident
// allocated. Metal buffers are unified/system memory and purego has no ARC, so without this a
// multi-model serve (or /admin/models/unload) leaks the whole model per load.
func (a *metalResident) Close() error { return a.r.Close() }
