//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// S9 on CUDA, part A: the batched text-prompt prefill for Gemma 4 E-models. The plan, the gates and the bars:
// docs/tasks/task-multimodal-support-2026-10.md ("S9 on CUDA, part A").

// eModelRows returns n resident rows (hidden ‖ PLE tails) for the golden prompt's ids cycled, so every row has a real, distinct token-identity term.
func eModelRows(t *testing.T, m *decoder.Model, n int) [][]float32 {
	t.Helper()
	ids := eModelPrompt(t)
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest(ids[(i*5+i/len(ids))%len(ids)])
	}
	return rows
}

func loadEModelResident(t *testing.T) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireEModelFixture(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda int4): %v", err)
	}
	t.Cleanup(func() { mg.Close() })
	r := mustResident(t, mg)
	// Non-vacuity: the fixture must present the E-model shape, or every comparison below passes by exercising none of it.
	shared := 0
	for _, L := range r.layers {
		if L.kvShared {
			shared++
		}
	}
	if r.pleP != 32 || shared != 2 || r.layers[0].ffnI != 256 || r.layers[4].ffnI != 512 {
		t.Fatalf("resident did not take the E-model shape: P=%d shared=%d ffn[0]=%d ffn[4]=%d", r.pleP, shared, r.layers[0].ffnI, r.layers[4].ffnI)
	}
	if ok, why := r.PrefillPath(); !ok {
		t.Fatalf("the E-model still declines the batched prefill: %s", why)
	}
	return mg, r
}

// seqLogits runs rows one token at a time on r (Reset first) and returns every position's logits, copied.
func seqLogits(t *testing.T, r *cudaResident, rows [][]float32) [][]float32 {
	t.Helper()
	r.Reset()
	out := make([][]float32, len(rows))
	for i, e := range rows {
		l, err := r.Forward(e, i)
		if err != nil {
			t.Fatalf("sequential Forward(%d): %v", i, err)
		}
		out[i] = append([]float32(nil), l...)
	}
	return out
}

func countDiff(a, b []float32) (n, first int) {
	first = -1
	if len(a) != len(b) {
		return len(a) + len(b), 0
	}
	for i := range a {
		if a[i] != b[i] {
			if first < 0 {
				first = i
			}
			n++
		}
	}
	return n, first
}

// TestGemma4EModelPrefill_batchedBitIdentical is G1p: the batched pass against the sequential per-token path on the SAME resident, bit-identical. Four
// cuts, all on the tiny E-model (sliding window 4, so the shared layers attend past it; 2 KV-shared layers; FFN 256 and 512):
//   - every row of PrefillLastN (the exact-kernel tail) against Forward at that position, over 96 rows and over the golden's 18;
//   - the last row of PrefillLast (the fast-lever tail) over the same;
//   - the chunked driver at chunk 5 and 7 (divisors of neither 18 nor 96);
//   - decode after the batched pass: 6 more tokens against the same 6 after a sequential prefill, which is what a wrongly stored or missing K/V on a shared
//     layer, or a PLE tail left stale in r.x, would break.
func TestGemma4EModelPrefill_batchedBitIdentical(t *testing.T) {
	ctx := context.Background()
	for _, n := range []int{18, 96} {
		t.Run(strconv.Itoa(n)+"rows", func(t *testing.T) {
			mg, r := loadEModelResident(t)
			rows := eModelRows(t, mg, n+6) // the last 6 are the decode-after tokens
			prompt, after := rows[:n], rows[n:]

			seq := seqLogits(t, r, rows) // positions 0..n+5, sequential throughout
			wantLast := seq[n-1]
			wantAfter := seq[n:]

			check := func(name string, run func() ([][]float32, []float32)) {
				t.Helper()
				r.Reset()
				r.passPromptLen = 0
				all, last := run()
				if r.passPromptLen != n {
					t.Fatalf("%s: passPromptLen = %d, want %d: the batched pass did not run (vacuous)", name, r.passPromptLen, n)
				}
				if all != nil {
					if len(all) != n {
						t.Fatalf("%s: %d rows of logits, want %d", name, len(all), n)
					}
					for i := range all {
						if d, f := countDiff(all[i], seq[i]); d != 0 {
							t.Errorf("%s: position %d: %d logits differ from sequential (first at %d: %v vs %v)", name, i, d, f, all[i][f], seq[i][f])
							break
						}
					}
					last = all[n-1]
				}
				if d, f := countDiff(last, wantLast); d != 0 {
					t.Errorf("%s: last-row logits: %d differ from sequential (first at %d: %v vs %v)", name, d, f, last[f], wantLast[f])
				}
				for i, e := range after {
					l, err := r.Forward(e, n+i)
					if err != nil {
						t.Fatalf("%s: decode after prefill, token %d: %v", name, i, err)
					}
					if d, f := countDiff(l, wantAfter[i]); d != 0 {
						t.Errorf("%s: decode token %d after the batched pass: %d logits differ from the sequential run (first at %d)", name, i, d, f)
						break
					}
				}
			}
			check("PrefillLastN (exact tail, every row)", func() ([][]float32, []float32) {
				all, err := r.PrefillLastN(prompt, 0)
				if err != nil {
					t.Fatalf("PrefillLastN: %v", err)
				}
				return all, nil
			})
			check("PrefillLast (fast-lever tail)", func() ([][]float32, []float32) {
				l, err := r.PrefillLast(ctx, prompt, 0)
				if err != nil {
					t.Fatalf("PrefillLast: %v", err)
				}
				return nil, l
			})
			for _, chunk := range []string{"5", "7"} {
				decoder.SetKnobEnvForTest(t, mg, "GOINFER_PREFILL_CHUNK", chunk)
				check("PrefillLast chunk "+chunk, func() ([][]float32, []float32) {
					l, err := r.PrefillLast(ctx, prompt, 0)
					if err != nil {
						t.Fatalf("PrefillLast chunk %s: %v", chunk, err)
					}
					return nil, l
				})
			}
		})
	}
}

// TestGemma4EModelPrefill_plantedDefects is G2p: S1's eight planted defects, each through the batched path. The batched all-rows logits are graded against
// the CPU int4 reference with G1c's rule; unplanted it must pass, and each defect alone must turn it red. Both halves matter: a batched path that did not
// implement the thing a seam perturbs would leave that defect green, and one that lacked the E-model machinery altogether would fail the unplanted run.
func TestGemma4EModelPrefill_plantedDefects(t *testing.T) {
	g := runEModelG1Mode(t, true)
	if ok, why := g.pass(); !ok {
		t.Fatalf("G2p unplanted: the batched path is RED against the CPU: %s (first hard divergence at pos %d; mean %.6f vs CPU envelope %.6f)", why, g.firstHardPos, g.meanCUDA, g.meanCPU)
	}
	t.Logf("G2p unplanted: batched GREEN (mean CUDA-vs-CPUint4 %.6f, envelope %.6f, exact argmax %d/%d)", g.meanCUDA, g.meanCPU, g.exact, g.n)
	for _, d := range eModelDefects() {
		t.Run(d.name, func(t *testing.T) {
			d.set(true)
			defer d.set(false)
			g := runEModelG1Mode(t, true)
			ok, why := g.pass()
			t.Logf("G2p %s: batched %s %s (mean cosine %.6f, exact argmax %d/%d, gaps>3%% %d)", d.name, map[bool]string{true: "GREEN", false: "RED"}[ok], why, g.meanCUDA, g.exact, g.n, g.gaps3)
			if ok {
				t.Errorf("planted defect %s left the batched path green — it does not implement what the seam perturbs", d.name)
			}
		})
	}
}

// TestGemma4EModelPrefill_exactKernelsAboveTheFloor guards that the fast prefill levers, which engage above a 512-row prompt floor, are not bit-identical to
// decode and have no fidelity evidence on an E-model (docs/tasks/task-multimodal-support-2026-10.md). With the floor moved to 0 ("fast at any length",
// GOINFER_CUDA_FAST_PREFILL_FLOOR) an E-model's PrefillLast must therefore still be bit-identical to the sequential path, and must launch none of the fast
// kernels. The tiny gates above never see this: their prompts are far under the floor.
func TestGemma4EModelPrefill_exactKernelsAboveTheFloor(t *testing.T) {
	mg, r := loadEModelResident(t)
	decoder.SetKnobEnvForTest(t, mg, "GOINFER_CUDA_FAST_PREFILL_FLOOR", "0")
	if !r.fastAttn && !r.fastGemm {
		t.Skip("neither fast lever is enabled on this resident: nothing to keep off")
	}
	const n = 96
	rows := eModelRows(t, mg, n)
	seq := seqLogits(t, r, rows)
	r.Reset()
	before := r.fastAttnLaunches + r.fastGemmLaunches
	l, err := r.PrefillLast(context.Background(), rows, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	if got := r.fastAttnLaunches + r.fastGemmLaunches - before; got != 0 {
		t.Errorf("%d fast-kernel launches on an E-model's batched prefill, want 0", got)
	}
	if d, f := countDiff(l, seq[n-1]); d != 0 {
		t.Errorf("last-row logits with the floor at 0: %d differ from sequential (first at %d: %v vs %v)", d, f, l[f], seq[n-1][f])
	}
}
