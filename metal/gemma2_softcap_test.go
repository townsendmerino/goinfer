//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma2Softcap_lanesAndPrefillMetal builds a resident for each Gemma 2 tiny fixture directly (Metal does not
// declare the attention softcap, so a load declines: TestGemma2DeclinesResidentMetal) and checks two things about a
// model with a cap. It is not the Gemma 2 parity gate; that is registered in
// docs/tasks/task-forward-architecture-2026-10.md §4.6 and runs with the declaration.
//
//   - The lanes: a capped model takes the exact attention kernels only, the ones the softcap is gated on
//     (TestAttnSoftcap_decodeKernels, TestAttnSoftcap_prefillKernel). The fixtures' head dimension is 32, which without a
//     cap takes the fused prefill kernel.
//   - The batched prefill agrees with the token-by-token path on the last token, at TestPrefillParityGemma's bar.
func TestGemma2Softcap_lanesAndPrefillMetal(t *testing.T) {
	for _, ckpt := range []string{"../testdata/gemma2-tiny", "../testdata/gemma2-hd-tiny"} {
		t.Run(filepath.Base(ckpt), func(t *testing.T) {
			m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, err := buildResident(m)
			if err != nil {
				t.Fatalf("resident: %v", err)
			}
			defer r.Close()
			if r.attnSoftcap == 0 || r.attnSoftcap != m.AttnLogitSoftcapResident() {
				t.Fatalf("resident cap %v, model cap %v: the fixture no longer has a cap, or the resident did not take it", r.attnSoftcap, m.AttnLogitSoftcapResident())
			}
			if fused, steel := r.prefillAttnKernels(); fused || steel {
				t.Errorf("a capped model would take the fused (%v) or steel (%v) prefill kernel, which the softcap is not gated on", fused, steel)
			}
			if !r.prefillExactAttn() {
				t.Error("prefillExactAttn is false for a capped model: the exact kernel's 4096-key bound would not be applied")
			}
			if r.batch != nil || r.batchIneligible() == "" {
				t.Errorf("a capped model built the batched step (batchIneligible %q)", r.batchIneligible())
			}
			for l := range r.layers {
				if r.canUseAttnFA(l) {
					t.Errorf("layer %d of a capped model may take the split-KV kernels", l)
				}
			}
			if !r.prefillOK {
				t.Fatal("prefillOK is false: the batched prefill would not run, and the comparison below would not gate it")
			}

			_, _, _, _, _, _, vocab := m.Dims()
			prompt := gemmaPrefillPrompt
			var seq []float32
			for i, id := range prompt {
				seq = r.Forward(id%vocab, i)
			}
			seq = append([]float32(nil), seq...)
			embs := make([][]float32, len(prompt))
			for i, id := range prompt {
				e := make([]float32, r.H)
				r.embed.Row(id%vocab, e)
				if r.embedScale > 1 {
					for j := range e {
						e[j] *= r.embedScale
					}
				}
				embs[i] = e
			}
			pre := r.PrefillLast(embs, 0)
			cos := cosF(seq, pre)
			t.Logf("batched prefill vs token by token, last of %d tokens: argmax %d / %d, cosine %.5f", len(prompt), argmaxF(pre), argmaxF(seq), cos)
			if math.IsNaN(float64(cos)) || argmaxF(pre) != argmaxF(seq) || cos < 0.95 {
				t.Errorf("batched prefill disagrees with the token-by-token path: argmax %d / %d, cosine %v", argmaxF(pre), argmaxF(seq), cos)
			}
		})
	}
}
