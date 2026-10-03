//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// D11's follow-up (docs/tasks/task-constrained-confidence.md, Route C): decoder.Model.PromptHiddenAll on a CUDA-resident Qwen3.5 answers from the device through the
// ResidentResidualAll seam, and every row stays within promptHiddenAllResidentBarCUDA of the CPU's PromptHiddenAll on the same weights. Three checks:
//
//   - the answer is the runner's own pre-norm residual rows with the decoder's f32 final norm applied (so the dispatch took the resident, and the norm is the host's, not the
//     int8-requantized vector ResidentHiddenLast returns);
//   - it tracks the CPU path per row (the device kernels, DeltaNet included, compute the right thing at EVERY position, not only the last);
//   - a prompt split into passes of 16 rows (GOINFER_PREFILL_CHUNK) gives the rows of the unchunked one: the chunk loop keeps every chunk's rows, not just the last.
//
// qwen3_5-tiny-normw has a random final-norm weight, so a missing or doubled norm shows. int4, as the other resident parity tests run.
const promptHiddenAllResidentBarCUDA = 0.998

func TestPromptHiddenAllResidentCUDA(t *testing.T) {
	requireCUDADevice(t)
	for _, ckpt := range []string{"../decoder/testdata/qwen3_5-tiny-normw", "../decoder/testdata/qwen3_5_moe-tiny"} {
		t.Run(ckpt, func(t *testing.T) {
			opts := decoder.Options{Backend: "cuda", Quant: "int4"}
			mRes, err := decoder.Load(ckpt, opts)
			if err != nil {
				t.Fatalf("load cuda: %v", err)
			}
			defer mRes.Close()
			rf := mRes.ResidentForwardForTest()
			if rf == nil {
				t.Skipf("%s did not go resident on this box — decode path %q; decline: %s", ckpt, mRes.DecodePath(), mRes.ResidentDecline())
			}
			rr, ok := rf.(decoder.ResidentResidualAll)
			if !ok {
				t.Fatal("the cuda resident runner does not implement decoder.ResidentResidualAll")
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
				// For context: the EXISTING last-row resident path (int8-requantized hidden) against the CPU on the same prompt.
				lastRes, err1 := mRes.PromptHidden(context.Background(), ids)
				lastCPU, err2 := mCPU.PromptHidden(context.Background(), ids)
				if err1 == nil && err2 == nil {
					c, _ := cosF32(lastCPU, lastRes)
					t.Logf("%d tokens: the existing last-row path (ResidentHiddenLast, int8-requantized) vs CPU: cosine %.8f; the new path's last row: %.8f", n, c, func() float64 { c2, _ := cosF32(cpu[n-1], got[n-1]); return c2 }())
				}
				t.Logf("%d tokens: resident vs CPU, per-row cosine: worst %.8f, mean %.8f, last %.8f; worst |diff| %.3g", n, worstCos, meanCos, lastCos, worstAbs)
				// DENSE: every row within the bar. MoE: a random-init router has near-tied top-k scores, so a few positions choose a different expert on the device
				// than on the CPU and ONLY those rows move (measured here: the LAST row stays at 0.9999 while a few interior rows dip to 0.993). The right statistic for
				// that is the mean, with the last row held to the bar and the worst reported, not asserted (the lesson of the MoE router-flip noise floor: floor the mean,
				// not the min over rows). A defect that moved EVERY row, or the last, still fails; one confined to a single MoE position would not, and the dense
				// case (same DeltaNet and attention kernels) is what pins per-position correctness.
				moe := strings.Contains(ckpt, "moe")
				if moe {
					if meanCos < promptHiddenAllResidentBarCUDA || lastCos < promptHiddenAllResidentBarCUDA {
						t.Errorf("%d tokens: MoE mean row cosine %.8f or last-row cosine %.8f < %.4f against the CPU", n, meanCos, lastCos, promptHiddenAllResidentBarCUDA)
					}
				} else if worstCos < promptHiddenAllResidentBarCUDA {
					t.Errorf("%d tokens: worst row cosine %.8f < %.4f against the CPU", n, worstCos, promptHiddenAllResidentBarCUDA)
				}
			}
		})
	}
}

// A prompt that runs in several passes returns every pass's rows, in order, identical to the one-pass run.
func TestPromptHiddenAllResidentCUDA_chunked(t *testing.T) {
	requireCUDADevice(t)
	const ckpt = "../decoder/testdata/qwen3_5-tiny-normw"
	load := func(chunk string) *decoder.Model {
		t.Helper()
		opts := decoder.Options{Backend: "cuda", Quant: "int4"}
		if chunk != "" {
			k := decoder.Knobs{"GOINFER_PREFILL_CHUNK": chunk}
			opts.Knobs = &k
		}
		m, err := decoder.Load(ckpt, opts)
		if err != nil {
			t.Fatal(err)
		}
		if m.ResidentForwardForTest() == nil {
			m.Close()
			t.Skipf("did not go resident: %s", m.ResidentDecline())
		}
		return m
	}
	one, chunked := load(""), load("16")
	defer one.Close()
	defer chunked.Close()
	_, _, _, _, _, _, vocab := one.Dims()
	ids := make([]int, 53)
	for i := range ids {
		ids[i] = (i*37 + 11) % vocab
	}
	a, err := one.PromptHiddenAll(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	b, err := chunked.PromptHiddenAll(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 53 || len(b) != 53 {
		t.Fatalf("%d and %d rows for 53 tokens", len(a), len(b))
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("row %d differs at %d between one pass and passes of 16 (%v vs %v): a chunk's rows were lost or reordered", i, j, a[i][j], b[i][j])
			}
		}
	}
}
