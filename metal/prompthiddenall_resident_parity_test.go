//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// D11's follow-up on Metal (docs/measurements/decisions-d11-resident-hidden-2026-10-03.md): decoder.Model.PromptHiddenAll on a Metal-resident Qwen3.5 answers from
// the device through the ResidentResidualAll seam, and every row stays within promptHiddenAllResidentBarMetal of the CPU's PromptHiddenAll on the same weights. The
// Metal twin of cuda/prompthiddenall_resident_parity_test.go. Two checks:
//
//   - the answer is the runner's own pre-norm residual rows with the decoder's f32 final norm applied (so the dispatch took the resident, and the norm is the host's,
//     not the int8 vector Metal's fused final norm leaves in r.aq/r.aSc);
//   - it tracks the CPU path per row (the device kernels, DeltaNet included, compute the right thing at EVERY position, not only the last). Three prompt lengths run
//     one after another, so a DeltaNet state left over from one call would show in the next.
//
// qwen3_5-tiny-normw has a random final-norm weight, so a missing or doubled norm shows. Both sides run int4: Metal has no int8 GEMV (int8int8 reads as int4 on its
// decode path), so an int8int8 CPU side would compare two quantizations, not two executions (see TestPromptHiddenResidentMetal). No chunked case: Metal's
// ResidualAll is per token, with no chunk loop to lose rows in.
const promptHiddenAllResidentBarMetal = 0.998

func TestPromptHiddenAllResidentMetal(t *testing.T) {
	for _, ckpt := range []string{"../decoder/testdata/qwen3_5-tiny-normw", "../decoder/testdata/qwen3_5_moe-tiny"} {
		t.Run(ckpt, func(t *testing.T) {
			moe := strings.Contains(ckpt, "moe")
			opts := decoder.Options{Backend: "metal", Quant: "int4"}
			mRes, err := decoder.Load(ckpt, opts)
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			defer mRes.Close()
			rf := mRes.ResidentForwardForTest()
			if rf == nil {
				// The dense fixture is the gate, so it failing to go resident is a failure, not a skip.
				if moe {
					t.Skipf("%s did not go resident — decode path %q; decline: %s", ckpt, mRes.DecodePath(), mRes.ResidentDecline())
				}
				t.Fatalf("%s did not go resident — decode path %q; decline: %s", ckpt, mRes.DecodePath(), mRes.ResidentDecline())
			}
			rr, ok := rf.(decoder.ResidentResidualAll)
			if !ok {
				t.Fatal("the metal resident runner does not implement decoder.ResidentResidualAll")
			}
			opts.Backend = "cpu"
			mCPU, err := decoder.Load(ckpt, opts)
			if err != nil {
				t.Fatalf("load cpu: %v", err)
			}
			defer mCPU.Close()
			_, _, _, _, _, _, vocab := mCPU.Dims()
			for _, n := range []int{40, 9, 2} {
				ids := make([]int, n)
				for i := range ids {
					ids[i] = (i*37 + 11) % vocab
				}
				got, err := mRes.PromptHiddenAll(context.Background(), ids)
				if err != nil {
					t.Fatalf("%d tokens: resident PromptHiddenAll: %v", n, err)
				}
				embs := make([][]float32, n)
				for i, id := range ids {
					embs[i] = mRes.EmbedResidentForTest(id)
				}
				raw, err := rr.ResidualAll(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("%d tokens: runner ResidualAll: %v", n, err)
				}
				if len(got) != n || len(raw) != n {
					t.Fatalf("%d tokens: %d rows from PromptHiddenAll and %d from the runner", n, len(got), len(raw))
				}
				for i := range got {
					want := mRes.FinalNormForTest(raw[i])
					for j := range want {
						if got[i][j] != want[j] {
							t.Fatalf("%d tokens, row %d: PromptHiddenAll differs from the runner's residual with the host's final norm at %d (%v vs %v): it did not take the resident", n, i, j, got[i][j], want[j])
						}
					}
				}
				cpu, err := mCPU.PromptHiddenAll(context.Background(), ids)
				if err != nil {
					t.Fatalf("%d tokens: cpu PromptHiddenAll: %v", n, err)
				}
				worstCos, worstAbs, sumCos := 1.0, 0.0, 0.0
				for i := range cpu {
					cos, maxAbs := cosF32(cpu[i], got[i])
					sumCos += cos
					if cos < worstCos {
						worstCos = cos
					}
					if maxAbs > worstAbs {
						worstAbs = maxAbs
					}
				}
				meanCos := sumCos / float64(n)
				lastCos, _ := cosF32(cpu[n-1], got[n-1])
				// For context: the existing last-row resident path (HiddenLast, the int8 final-norm output) against the CPU on the same prompt.
				if lastRes, err1 := mRes.PromptHidden(context.Background(), ids); err1 == nil {
					if lastCPU, err2 := mCPU.PromptHidden(context.Background(), ids); err2 == nil {
						c, _ := cosF32(lastCPU, lastRes)
						t.Logf("%d tokens: the existing last-row path (HiddenLast, int8) vs CPU: cosine %.8f; the new path's last row: %.8f", n, c, lastCos)
					}
				}
				t.Logf("%d tokens: resident vs CPU, per-row cosine: worst %.8f, mean %.8f, last %.8f; worst |diff| %.3g", n, worstCos, meanCos, lastCos, worstAbs)
				// DENSE: every row within the bar. MoE: a random-init router has near-tied top-k scores, so a few interior positions pick another expert on the device
				// than on the CPU and only those rows move; the mean and the last row are held to the bar and the worst is reported, not asserted (the CUDA twin's note).
				if moe {
					if meanCos < promptHiddenAllResidentBarMetal || lastCos < promptHiddenAllResidentBarMetal {
						t.Errorf("%d tokens: MoE mean row cosine %.8f or last-row cosine %.8f < %.4f against the CPU", n, meanCos, lastCos, promptHiddenAllResidentBarMetal)
					}
				} else if worstCos < promptHiddenAllResidentBarMetal {
					t.Errorf("%d tokens: worst row cosine %.8f < %.4f against the CPU", n, worstCos, promptHiddenAllResidentBarMetal)
				}
			}
		})
	}
}

// TestPromptHiddenAllResidentMetal_realWeights is the same comparison on real weights, EXPLORATORY: Qwen3.5-0.8B (the only real
// Qwen3.5 this 16 GB Mac can hold resident; the 9B Clef-flash cannot), resident against the CPU at int4, every row of a few hundred
// tokens. It asserts only that every row is finite and the comparison ran; the cosines are reported for the record, not graded.
// Set GOINFER_METAL_QWEN35_REAL to the checkpoint directory (from ~/models, never the archive) to run it.
func TestPromptHiddenAllResidentMetal_realWeights(t *testing.T) {
	ckpt := os.Getenv("GOINFER_METAL_QWEN35_REAL")
	if ckpt == "" {
		t.Skip("set GOINFER_METAL_QWEN35_REAL=<a Qwen3.5 checkpoint dir in ~/models> for the real-weights comparison")
	}
	opts := decoder.Options{Backend: "metal", Quant: "int4"}
	mRes, err := decoder.Load(ckpt, opts)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	if mRes.ResidentForwardForTest() == nil {
		t.Fatalf("did not go resident — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	opts.Backend = "cpu"
	mCPU, err := decoder.Load(ckpt, opts)
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()
	for _, n := range []int{300, 33} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 101) % vocab
		}
		t0 := time.Now()
		got, err := mRes.PromptHiddenAll(context.Background(), ids)
		if err != nil {
			t.Fatalf("%d tokens: resident: %v", n, err)
		}
		resT := time.Since(t0)
		t0 = time.Now()
		cpu, err := mCPU.PromptHiddenAll(context.Background(), ids)
		if err != nil {
			t.Fatalf("%d tokens: cpu: %v", n, err)
		}
		cpuT := time.Since(t0)
		worst, sum, worstAt, worstRest := 1.0, 0.0, 0, 1.0
		for i := range cpu {
			if firstNonFinite(got[i]) >= 0 {
				t.Fatalf("%d tokens: row %d is not finite", n, i)
			}
			c, _ := cosF32(cpu[i], got[i])
			sum += c
			if c < worst {
				worst, worstAt = c, i
			}
			if i > 0 {
				worstRest = min(worstRest, c)
			}
		}
		last, _ := cosF32(cpu[n-1], got[n-1])
		t.Logf("EXPLORATORY, %d tokens: resident vs CPU per-row cosine worst %.6f (row %d), worst after row 0 %.6f, mean %.6f, last %.6f; resident %s (%.1f ms/token), cpu %s",
			n, worst, worstAt, worstRest, sum/float64(n), last, resT.Round(time.Millisecond), float64(resT.Milliseconds())/float64(n), cpuT.Round(time.Millisecond))
		// The existing last-row path (HiddenLast: the same kernels, its int8 final-norm output) on the same prompt, as the baseline for what Metal's decode kernels
		// agree with the CPU to on these weights.
		if lr, e1 := mRes.PromptHidden(context.Background(), ids); e1 == nil {
			if lc, e2 := mCPU.PromptHidden(context.Background(), ids); e2 == nil {
				c, _ := cosF32(lc, lr)
				t.Logf("EXPLORATORY, %d tokens: the existing last-row path (HiddenLast) vs CPU: %.6f", n, c)
			}
		}
	}
}
