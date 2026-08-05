//go:build darwin

package metal

// Metal decoder backend — plugs the cgo-free native-Metal resident decoder into the goinfer
// decode loop via decoder.RegisterBackend, mirroring the cuda backend. Blank-import this
// package (under a build tag) from a main to enable `--backend metal`. Dense residency only
// (Qwen2/Llama, DecodeRunnerEligible); declines gracefully to the staged/CPU path otherwise.

import (
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
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
			if os.Getenv("GOINFER_RESIDENT_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "[metal] BuildResident declined: %v\n", p)
			}
			rf, ok, err = nil, false, nil
		}
	}()
	// Admission check: DecodeRunnerEligible was scoped to the richer WebGPU/CUDA runner, which
	// admits QK-norm / sliding-window / partial-rotary and more. The Metal kernel set implements
	// only a subset (decoder.ResidentBackendFeatures("metal")); anything it doesn't implement must
	// DECLINE (→ correct CPU fallback) rather than run with the feature silently dropped. The
	// subset check uses the shared taxonomy (one source of truth; a new arch classifies itself).
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("metal")); len(missing) > 0 {
		if os.Getenv("GOINFER_RESIDENT_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[metal] declined — unimplemented features: %v\n", missing)
		}
		return nil, false, nil
	}
	res, e := buildResident(m)
	if e != nil {
		if os.Getenv("GOINFER_RESIDENT_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[metal] BuildResident declined: %v\n", e)
		}
		return nil, false, nil
	}
	b.resident = &metalResident{r: res, hidden: res.H}
	return b.resident, true, nil
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
	return a.r.ForwardEmbPipe(embedding, pos), nil // pipelined executor (encode-ahead)
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
	// caller falls back to the sequential Forward loop (decode kernels ⇒ bit-identical). Opt in only for
	// measurement/TTFT-at-the-cost-of-exactness: GOINFER_METAL_BATCHED_PREFILL=1.
	if os.Getenv("GOINFER_METAL_BATCHED_PREFILL") != "1" {
		return nil, fmt.Errorf("metal: batched prefill declined — not bit-identical to decode (54%% stream divergence, §A2-Metal); using sequential. Set GOINFER_METAL_BATCHED_PREFILL=1 to force")
	}
	// The f16 MMA prefill kernels implement only the plain dense shape (a dense FFN out of
	// L.guW/L.dW, model-level rope/window, SiLU-only swiglu). A MoE model never packs those
	// dense FFN buffers at all, so prefilling it would bind zero buffers — decline instead,
	// and the caller re-runs the prompt through the (correct) sequential Forward loop.
	if !a.r.prefillOK {
		return nil, fmt.Errorf("metal: prefill not implemented for this arch's FFN shape (use the sequential path)")
	}
	if len(embeddings) == 0 || startPos+len(embeddings) > metalCtxCap {
		return nil, fmt.Errorf("metal: prompt %d exceeds resident cap %d", len(embeddings), metalCtxCap)
	}
	return a.r.PrefillLast(embeddings, startPos), nil
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

// TruncateTo / Reset are no-ops: KV positions are overwritten on write, and attention only
// reads keys[0..pos], so stale positions past the current one are never observed.
func (a *metalResident) TruncateTo(pos int) {}
func (a *metalResident) Reset()             {}

// Close stops the pipelined executor (waiting for it) and frees every MTLBuffer this resident
// allocated. Metal buffers are unified/system memory and purego has no ARC, so without this a
// multi-model serve (or /admin/models/unload) leaks the whole model per load.
func (a *metalResident) Close() error { a.r.Close(); return nil }
