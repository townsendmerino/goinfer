//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package (under a build tag) from a main to enable `--backend metal`. Dense residency only
// (Qwen2/Llama, DecodeRunnerEligible); declines gracefully to the staged/CPU path otherwise.

import (
	"context"
	"fmt"
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
	// residentFitsMemory's own budget, exposed to decoder.Model.Plan (docs/task-fit-to-hardware.md
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
	_ decoder.ResidentAdapter     = (*metalResident)(nil)
	_ decoder.Prefiller           = (*metalResident)(nil)
	_ decoder.PrefillPathReporter = (*metalResident)(nil)
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
// declines. G10 (docs/task-gpu-paths-2026-09.md): this backend has NO int8 GEMV kernel at all —
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
	// G6 (docs/task-gpu-paths-2026-09.md — "honoured or refused with numbers"): an explicit -ctx
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
// set (Phase 2, docs/task-gpu-paths-2026-09.md — "Metal slots become an Option and a flag");
// GOINFER_METAL_MOE_SLOTS is kept as a deprecated fallback for anyone still setting it directly.
// "" means unset either way — n==0/unset ⇒ every expert resident, today's behavior, unchanged.
func metalMoESlotsRequest(m *decoder.Model) string {
	if n := m.MoECacheSlotsRequest(); n > 0 {
		return strconv.Itoa(n)
	}
	return os.Getenv("GOINFER_METAL_MOE_SLOTS")
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
	const bytesPerElem = 2 // f16; see comment above
	var total int64
	for l := 0; l < nLayers; l++ {
		kvDim := int64(m.KVHeadsAtResident(l)) * int64(m.HeadDimAtResident(l))
		total += 2 * int64(ctxCap) * kvDim * bytesPerElem // ×2 for K and V
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
	fmt.Fprintf(os.Stderr, "[metal] declined — weights %.2f GB exceed %.0f%% of %.1f GB RAM "+
		"(budget %.2f GB). Metal wires the pages it touches, so loading this would page to swap "+
		"exhaustion rather than run; continuing on the CPU/staged path. Override with "+
		"GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.\n",
		float64(need)/gb, residentMemFraction*100, float64(ram)/gb, float64(budget)/gb)
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
		return metalCtxCapMax
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

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if len(embedding) != a.hidden {
		return nil, fmt.Errorf("metal: embedding len %d != hidden %d", len(embedding), a.hidden)
	}
	// Guard before the pipelined executor enqueues the job: the KV write happens at commit with
	// r.uPos=pos, so refusing here (pre-enqueue) prevents any OOB device write. This path also
	// covers PrefillLast's >cap decline, which falls back to sequential Forward(emb, i).
	if e := a.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero any Gated-DeltaNet state before it carries over from a prior
		// Generate on this resident (audit C-01's CUDA analogue). No-op for every other family.
		a.Reset()
	}
	logits := a.r.ForwardEmbPipe(embedding, pos) // pipelined executor (encode-ahead)
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // C-09: a command buffer aborted — surface it, do NOT return stale logits
	}
	return logits, nil
}

// metalFastPrefillFloor is the PROMPT-LENGTH floor (whole prompt = startPos+M) below which
// the f16-MMA batched pass declines and the sequential per-token loop runs instead. Set to 512
// to match CUDA's measured floor (K=256 is expected to fail §3.2; the gate test disables this
// floor via GOINFER_METAL_FAST_PREFILL_FLOOR=0 to confirm). On Phase B pass, if K=256 fails as
// expected, this floor stands. Override with GOINFER_METAL_FAST_PREFILL_FLOOR (0 = no floor).
const metalFastPrefillFloor = 512

// metalFastPrefillEnabled reports whether the batched f16-MMA prefill path is selected.
//
// CURRENTLY OPT-IN (default off) pending Phase B (TestPrefillGateVsReference, 2026-09-09). When
// Phase B passes under §3.2's pooled form, the default flips to ON above metalFastPrefillFloor
// and this comment is updated to name the measurement doc. The infrastructure is here now so the
// flip is a one-line change in this function.
//
//	GOINFER_METAL_FAST_PREFILL  1 | true | on   on (even below the floor — for tests)
//	                            0 | false | off  off (the current default; will become explicit opt-out on gate pass)
//
// The old GOINFER_METAL_BATCHED_PREFILL=1 opt-in continues to work as an alias. --exact-prefill
// (internal/serveapp) sets GOINFER_METAL_FAST_PREFILL=0 to suppress the batched path.
func metalFastPrefillEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_METAL_FAST_PREFILL"))) {
	case "0", "false", "off":
		return false
	case "1", "true", "on":
		return true
	}
	// Unset: honour the old opt-in var for backward compat (=1 enables, =0 disables).
	if strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_METAL_BATCHED_PREFILL"))) == "1" {
		return true
	}
	return false // flip to `true` when Phase B passes (§3.2 gate, 2026-09-09)
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

// PrefillPath (decoder.PrefillPathReporter) reports at load time whether this resident will use
// the batched f16-MMA path. Currently opt-in (default off; flip on Phase B gate pass). The floor
// applies per-call; PrefillPath reports true iff the enabled state AND arch both allow batching.
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
	return true, "batched f16-MMA (GOINFER_METAL_FAST_PREFILL_FLOOR=0; §3 gate pending)"
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
	// OPT-IN until Phase B (TestPrefillGateVsReference, 2026-09-09) passes. On pass the default flips
	// to ON above the 512-token floor. GOINFER_METAL_FAST_PREFILL=1 or GOINFER_METAL_BATCHED_PREFILL=1
	// opt in now; GOINFER_METAL_FAST_PREFILL=0 or --exact-prefill suppresses it once it is default-on.
	if !metalFastPrefillEnabled() {
		return nil, fmt.Errorf("metal: fast prefill disabled (GOINFER_METAL_FAST_PREFILL=0 / --exact-prefill / GOINFER_METAL_BATCHED_PREFILL=0); using sequential path")
	}
	// FLOOR: below metalFastPrefillFloor the fidelity gate has NOT passed (K=256 cell failed §3.2).
	// Sequential path there; fast path only where the gate cleared.
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
// PrefillLast, but for embedding requests (G4, docs/task-gpu-paths-2026-09.md) instead of
// generation: it never runs the LM head. Metal's batched (f16-MMA) PrefillLast is declined by
// default because it is not bit-identical to decode (§A2-Metal); rather than reuse that
// divergent path, this runs the SAME per-token sequential kernels decode uses — one
// forwardHiddenNoHead call per position — which is bit-identical to the CPU reference by
// construction, at the cost of one command-buffer submit per token instead of Prefiller's one
// pass (the same TTFT trade PrefillLast's decline already makes for generation).
func (a *metalResident) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("metal: HiddenLast called with no embeddings")
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

// ForwardN runs a batch of embeddings at consecutive positions (prefill). Each row is copied
// off the reused host logits buffer so all survive.
func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	// Fail-fast before any write: the loop's Forward calls each guard their own pos, but checking
	// the whole batch up front refuses an over-cap run without partial KV writes.
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
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

// UploadKV (prefix-reuse bridge) is not supported: the resident decoder owns its KV writes
// per Forward, and the stateless Generate path re-runs the prompt through Forward instead.
// SetAdapter implements decoder.ResidentAdapter (G3, docs/task-gpu-paths-2026-09.md) —
// generateInto calls this to bind/clear a compute-time LoRA adapter for an admitted session.
func (a *metalResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return a.r.SetAdapter(layers)
}

func (a *metalResident) UploadKV(layer, base int, keys, vals []float32) error {
	return fmt.Errorf("metal: UploadKV not supported (re-run the prefix through Forward)")
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
