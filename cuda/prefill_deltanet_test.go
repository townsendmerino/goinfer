//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCUDADeltaNetPrefill_matchesPerToken is gate 1 of docs/tasks/task-cuda-deltanet-prefill-2026-09.md: the batched
// prefill of a Gated-DeltaNet model against the per-token path on the same resident.
//   - Below 16 rows the batched GEMVs are decode's own kernel: the last-token logits must be bit-identical.
//   - At 16 rows and above the projections take the batched GEMM: argmax equal and cosine >= 0.9999.
//   - State continuity: 16 greedy tokens after the prompt equal the per-token path's, and a prompt prefilled in two
//     chunks reads the same last-token argmax (cosine >= 0.9999).
//
// Tiny fixtures by default; GOINFER_CUDA_DNET_MODEL points it at a real checkpoint (Qwen3.5-9B Q4_K_M, lengths
// 145 / 561 / 621), which also reports the speed gate's timing.
func TestCUDADeltaNetPrefill_matchesPerToken(t *testing.T) {
	type cfg struct {
		path    string
		lengths []int
		split   int
	}
	cfgs := []cfg{
		{filepath.Join("..", "testdata", "qwen35-tiny"), []int{3, 8, 15, 40}, 25},
		{filepath.Join("..", "testdata", "qwen3next-tiny"), []int{3, 8, 15, 40}, 25},
		// Post-only norm placement (Olmo 3's): the DeltaNet layers' FFN must still take the pre-MLP norm, as decode's
		// segBFFN does. The first cut applied the model-level placement there and read cosine 0.888 at 24 rows.
		{filepath.Join("..", "testdata", "olmo_hybrid-tiny"), []int{3, 8, 15, 40}, 25},
	}
	if p := os.Getenv("GOINFER_CUDA_DNET_MODEL"); p != "" {
		cfgs = []cfg{{p, []int{145, 561, 621}, 300}}
	}
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path), func(t *testing.T) {
			requireCUDADevice(t)
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no fixture at %s", c.path)
			}
			m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				// qwen3next-tiny's MoE shape (moeInter 16, hidden 64) is below the resident's int4 multiple-of-32
				// requirement, so the whole resident declines before this path is reached. A skip, not a pass:
				// the task doc records which fixtures exercised the gate.
				t.Skipf("not CUDA-resident, so the batched prefill is unreachable here: %s", m.ResidentDecline())
			}
			if r.dnet == nil {
				t.Fatal("fixture has no Gated-DeltaNet layers — the gate would test nothing")
			}
			_, _, _, _, _, _, vocab := m.Dims()
			emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
			argmax := func(v []float32) int {
				b := 0
				for i := range v {
					if v[i] > v[b] {
						b = i
					}
				}
				return b
			}
			cos := func(a, b []float32) float64 {
				var ab, aa, bb float64
				for i := range a {
					ab += float64(a[i]) * float64(b[i])
					aa += float64(a[i]) * float64(a[i])
					bb += float64(b[i]) * float64(b[i])
				}
				return ab / math.Sqrt(aa*bb)
			}
			perToken := func(prompt [][]float32) ([]float32, time.Duration) {
				r.Reset()
				t0 := time.Now()
				var last []float32
				for i, e := range prompt {
					l, err := r.Forward(e, i)
					if err != nil {
						t.Fatalf("per-token forward %d: %v", i, err)
					}
					last = l
				}
				return append([]float32(nil), last...), time.Since(t0)
			}
			batched := func(prompt [][]float32) ([]float32, time.Duration) {
				r.Reset()
				t0 := time.Now()
				l, err := r.PrefillLast(context.Background(), prompt, 0)
				if err != nil {
					t.Fatalf("batched prefill: %v", err)
				}
				return append([]float32(nil), l...), time.Since(t0)
			}
			greedy := func(from []float32, pos, n int) []int {
				ids := make([]int, 0, n)
				l := from
				for i := 0; i < n; i++ {
					id := argmax(l)
					ids = append(ids, id)
					nl, err := r.Forward(emb(id), pos+i)
					if err != nil {
						t.Fatalf("greedy step %d: %v", i, err)
					}
					l = append([]float32(nil), nl...)
				}
				return ids
			}
			if os.Getenv("GOINFER_CUDA_DNET_PROF") != "" { // diagnostic: where a prefill's time goes
				L := c.lengths[len(c.lengths)-1]
				prompt := make([][]float32, L)
				for i := range prompt {
					prompt[i] = emb(i*131 + 7)
				}
				r.Reset()
				r.prof = &prefillProf{}
				if _, err := r.PrefillLast(context.Background(), prompt, 0); err != nil {
					t.Fatal(err)
				}
				p := r.prof
				r.prof = nil
				t.Logf("profile L=%d: gemv %v, attention %v, DeltaNet recurrence %v, glue %v", L, p.gemv, p.attn, p.rec, p.glue)
			}
			for _, L := range c.lengths {
				prompt := make([][]float32, L)
				for i := range prompt {
					prompt[i] = emb(i*131 + 7)
				}
				ref, tRef := perToken(prompt)
				refCont := greedy(ref, L, 16)
				got, tGot := batched(prompt)
				gotCont := greedy(got, L, 16)
				cs0 := cos(ref, got)
				diff := 0
				for i := range ref {
					if ref[i] != got[i] {
						diff++
					}
				}
				t.Logf("L=%d: %d of %d logits differ, cosine %.7f, argmax per-token %d batched %d; per-token %v, batched %v (%.1fx)",
					L, diff, len(ref), cs0, argmax(ref), argmax(got), tRef.Round(time.Millisecond), tGot.Round(time.Millisecond),
					float64(tRef)/float64(tGot))
				if L < 16 && diff != 0 {
					t.Errorf("L=%d (<16 rows, decode's own kernels): %d logits differ — must be bit-identical", L, diff)
				}
				if argmax(ref) != argmax(got) || cs0 < 0.9999 {
					t.Errorf("L=%d: argmax %d vs %d, cosine %.7f (bar: equal, >= 0.9999)", L, argmax(ref), argmax(got), cs0)
				}
				if fmt.Sprint(refCont) != fmt.Sprint(gotCont) {
					t.Errorf("L=%d: 16 greedy tokens after the prompt differ:\n per-token %v\n batched   %v", L, refCont, gotCont)
				}
				if L > c.split {
					r.Reset()
					if _, err := r.PrefillLast(context.Background(), prompt[:c.split], 0); err != nil {
						t.Fatalf("chunk 1: %v", err)
					}
					two, err := r.PrefillLast(context.Background(), prompt[c.split:], c.split)
					if err != nil {
						t.Fatalf("chunk 2: %v", err)
					}
					if cs := cos(ref, two); argmax(two) != argmax(ref) || cs < 0.9999 {
						t.Errorf("L=%d split at %d: argmax %d vs %d, cosine %.7f — the second chunk did not continue the state",
							L, c.split, argmax(two), argmax(ref), cs)
					} else {
						t.Logf("L=%d split at %d: argmax equal, cosine %.7f", L, c.split, cs)
					}
				}
			}
		})
	}
}
