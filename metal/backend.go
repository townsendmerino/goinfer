//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package (under a build tag) from a main to enable `--backend metal`. Dense residency only
// (Qwen2/Llama, DecodeRunnerEligible); declines gracefully to the staged/CPU path otherwise.

import (
	"fmt"
	"os"
	"strconv"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"golang.org/x/sys/unix"
)

func init() {
	decoder.RegisterBackend("metal", func() (decoder.Backend, error) {
		return &metalBackend{}, nil
	})
}

// Compile-time seams: catch signature drift against decoder/residency.go + decoder/backend.go.
var (
	_ decoder.Backend          = (*metalBackend)(nil)
	_ decoder.ResidencyBackend = (*metalBackend)(nil)
	_ decoder.ResidentForward  = (*metalResident)(nil)
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
// weights must be int8-loaded (Options{Quant:"int8int8"}) or the pack step declines.
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

// metalMoESlotsFromEnv parses GOINFER_METAL_MOE_SLOTS the same way metal/moe.go and
// metal/gemma4_moe.go do (the shared paging knob), for the guard's own use. Unlike those two, an
// invalid or unset value is not an error here — it just means "assume unpaged" (the guard's
// existing, safe behavior); buildResident itself still validates and declines on a bad value.
func metalMoESlotsFromEnv() int {
	n, err := strconv.Atoi(os.Getenv("GOINFER_METAL_MOE_SLOTS"))
	if err != nil || n <= 0 {
		return 0
	}
	return n
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
func residentNeedBytes(m *decoder.Model) int64 {
	return m.ResidentWeightBytesPaged(metalMoESlotsFromEnv())
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

// checkCap guards the resident KV allocation (C3). Every layer's cache is r.kc[l]/r.vc[l], sized
// metalCtxCap*kvDim, so kv_store writes absolute position p at kc[p*kvDim ...]; valid positions
// are [0, metalCtxCap). Writing past it is an out-of-bounds device write — on Metal's UNIFIED
// memory that silently corrupts adjacent MTLBuffers (other models' resident weights), and once
// nKeys > 4096 the attention kernel's `threadgroup float sc[4096]` overflows too. The decode loop
// increments pos unbounded (a ≤cap prompt + a large max_tokens is enough), so refuse here; the
// decode loop surfaces the error (model.go) and the caller can fall back to the staged path.
func (a *metalResident) checkCap(pos, n int) error {
	if pos < 0 || pos+n > metalCtxCap {
		return fmt.Errorf("metal: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, metalCtxCap)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions. Implementing it makes metalResident satisfy
// decoder.ResidentCapped, so generateInto clamps maxTokens to the cap UP FRONT (stops cleanly at
// the cap instead of erroring mid-decode). Queryable so callers clamp rather than discover mid-run.
func (a *metalResident) ContextCap() int { return metalCtxCap }

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

// PrefillLast (decoder.Prefiller) ingests the whole prompt in one batched f16-MMA pass and
// returns the last token's logits, populating the resident KV. Falls back (declines) for prompts
// longer than the resident KV/attention cap, so the caller uses the sequential loop.
func (a *metalResident) PrefillLast(embeddings [][]float32, startPos int) ([]float32, error) {
	// DECLINE BY DEFAULT — Metal's batched prefill is NOT bit-identical to the sequential decode path.
	// It runs f16-activation MMA vs decode's int8 activations (+ fast-math contraction/reassociation),
	// which diverges the greedy stream on real weights (54% — TestMetalPrefillDivergenceRate; §A2-Metal).
	// The decoder's shared batched-prefill gate is default-ON (correct for CUDA, which was made
	// fma-bit-identical), so it would otherwise pull this divergent path in by default. Decline → the
	// caller falls back to the sequential Forward loop (decode kernels ⇒ bit-identical).
	//
	// Opt in — a real, disclosed feature since P11 (measured 3.9-4.6x TTFT at real prompt lengths,
	// docs/ollama-chase.md "Metal batched prefill as a TTFT lever"), not just an internal measurement
	// knob: `--metal-fast-prefill` on the server binary (internal/serveapp) sets this same env var at
	// startup, with the tradeoff spelled out in --help. GOINFER_METAL_BATCHED_PREFILL=1 also works
	// directly for anyone driving decoder.Load without the CLI (tests, embedders).
	if os.Getenv("GOINFER_METAL_BATCHED_PREFILL") != "1" {
		return nil, fmt.Errorf("metal: batched prefill declined — not bit-identical to decode (54%% stream divergence, §A2-Metal); using sequential. Pass --metal-fast-prefill (or set GOINFER_METAL_BATCHED_PREFILL=1) to force")
	}
	// The f16 MMA prefill kernels implement only the plain dense shape (a dense FFN out of
	// L.guW/L.dW, model-level rope/window, SiLU-only swiglu). A MoE model never packs those
	// dense FFN buffers at all, so prefilling it would bind zero buffers — decline instead,
	// and the caller re-runs the prompt through the (correct) sequential Forward loop.
	if !a.r.prefillOK {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	// startPos < 0 would wrap to a huge uint32 and make kv_store_f16 write far out of bounds — on
	// UMA that silently corrupts adjacent buffers (audit R-27). Unreachable today (the decoder always
	// passes 0) but cheap to guard.
	if startPos < 0 || len(embeddings) == 0 || startPos+len(embeddings) > metalCtxCap {
		return nil, fmt.Errorf("metal: prompt len %d at startPos %d out of resident cap %d", len(embeddings), startPos, metalCtxCap)
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
func (a *metalResident) UploadKV(layer int, keys, vals []float32) error {
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
