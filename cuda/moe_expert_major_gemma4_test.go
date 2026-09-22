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

// TestMoEExpertMajorGemma4CUDA_bitIdentical is the gemma4 extension's gate, mirroring
// TestMoEExpertMajorCUDA_bitIdentical exactly (cuda/moe_expert_major_test.go's own header explains
// why topK must be >= 3 to catch an accumulation-order regression at all — 2-term float addition is
// exactly commutative). testdata/gemma4-moe-tiny-k3 is the same pin_gemma4_moe_forward.py fixture
// family (PIN_TOPK=3), built for this test.
func TestMoEExpertMajorGemma4CUDA_bitIdentical(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny-k3"
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture at %s: %v", dir, err)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3") // == topK, minimum honourable (G-07); < nE(4) so LRU eviction pressure is exercised

	run := func(t *testing.T, on bool) []float32 {
		if on {
			t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", "1")
		} else {
			t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", "0")
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
		const M = 24
		embs := make([][]float32, M)
		var s uint32 = 7373
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
			t.Fatal("GOINFER_CUDA_MOE_EXPERT_MAJOR=1 but expert-major never ran on any layer — non-vacuity check failed")
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
		t.Errorf("gemma4 expert-major diverges from the per-row path on %d/%d logits", diff, len(off))
	} else {
		t.Logf("gemma4 expert-major bit-identical to the per-row path on all %d logits", len(off))
	}
}
