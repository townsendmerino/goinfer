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
	res, e := BuildResident(m)
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

// metalResident adapts *Resident (whose Forward takes a token id and returns logits, no error)
// to decoder.ResidentForward (Forward takes a precomputed embedding and returns logits+error).
type metalResident struct {
	r      *Resident
	hidden int
}

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if len(embedding) != a.hidden {
		return nil, fmt.Errorf("metal: embedding len %d != hidden %d", len(embedding), a.hidden)
	}
	return a.r.ForwardEmbPipe(embedding, pos), nil // pipelined executor (encode-ahead)
}

// PrefillLast (decoder.Prefiller) ingests the whole prompt in one batched f16-MMA pass and
// returns the last token's logits, populating the resident KV. Falls back (declines) for prompts
// longer than the resident KV/attention cap, so the caller uses the sequential loop.
func (a *metalResident) PrefillLast(embeddings [][]float32, startPos int) ([]float32, error) {
	if len(embeddings) == 0 || startPos+len(embeddings) > metalCtxCap {
		return nil, fmt.Errorf("metal: prompt %d exceeds resident cap %d", len(embeddings), metalCtxCap)
	}
	return a.r.PrefillLast(embeddings, startPos), nil
}

// ForwardN runs a batch of embeddings at consecutive positions (prefill). Each row is copied
// off the reused host logits buffer so all survive.
func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
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

// Close stops the pipelined executor goroutine (if started). Metal buffers are freed at process
// exit (single-model lifetime).
func (a *metalResident) Close() error { a.r.stopExec(); return nil }
