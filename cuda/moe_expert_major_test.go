//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEExpertMajorCUDA_bitIdentical is R11/P20's gate (docs/measurements/p20-expert-locality-2026-09-21.md):
// the expert-major MoE prefill restructuring (cuda/moe_expert_major.go), GOINFER_CUDA_MOE_EXPERT_MAJOR, on vs
// off, must produce byte-identical prefill logits on a real (tiny) MoE checkpoint — mirroring
// decoder/mlp.go's own TestMoEExpertMajor_bitIdentical (P18, CPU) both in what it asserts and in how: full
// logits, not just argmax, plus a non-vacuity counter so a silent decline cannot pass as a green.
//
//	go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestMoEExpertMajorCUDA_bitIdentical -v
func TestMoEExpertMajorCUDA_bitIdentical(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	// k=2 (qwen3moe-tiny) covers a real, different geometry; k=3 (qwen3moe-tiny-k3, same pin script with
	// num_experts_per_tok=3) is the one that can actually FAIL on an accumulation-order bug — float addition of
	// exactly 2 terms is commutative, so a rank-order regression is invisible at k=2 (checked directly: reversing
	// the fold order below passes cleanly at k=2 and fails 482/512 logits at k=3). Both are run for coverage.
	for _, dir := range []string{"../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3"} {
		t.Run(dir, func(t *testing.T) {
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture at %s: %v", dir, err)
			}
			t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
			t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3") // < nE(8) and < topK*rows, so real LRU eviction pressure is exercised

			run := func(t *testing.T, on bool) []float32 {
				if on {
					t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", "1")
				} else {
					t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", "")
				}
				mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
				if err != nil {
					t.Skipf("cuda load: %v", err)
				}
				defer mc.Close()
				r, ok := mc.ResidentForwardForTest().(*cudaResident)
				if !ok {
					t.Skip("resident did not engage")
				}
				if !r.cacheExperts {
					t.Skip("C′ expert staging did not engage")
				}
				if batched, why := r.PrefillPath(); !batched {
					t.Skipf("batched prefill declined: %s", why)
				}
				_, _, _, _, _, _, vocab := mc.Dims()
				const M = 40
				embs := make([][]float32, M)
				var s uint32 = 4242
				for i := range embs {
					s = s*1664525 + 1013904223
					embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
				}
				before := cudaMoeExpertMajorRuns
				out, err := r.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("PrefillLast: %v", err)
				}
				if on && cudaMoeExpertMajorRuns == before {
					t.Fatal("GOINFER_CUDA_MOE_EXPERT_MAJOR=1 but expert-major never ran on any layer — non-vacuity check failed (every layer declined eligibility)")
				}
				if !on && cudaMoeExpertMajorRuns != before {
					t.Fatal("expert-major ran with the flag OFF — the env gate did not reach the resident")
				}
				return out
			}

			off := run(t, false)
			on := run(t, true)
			if len(off) != len(on) {
				t.Fatalf("logit length off=%d on=%d", len(off), len(on))
			}
			diff := 0
			for i := range off {
				if math.Float32bits(off[i]) != math.Float32bits(on[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("expert-major diverges from the per-row path on %d/%d logits", diff, len(off))
			} else {
				t.Logf("expert-major bit-identical to the per-row path on all %d logits", len(off))
			}
		})
	}
}
