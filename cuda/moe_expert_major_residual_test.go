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

// emResidualRows loads dir on CUDA and returns every prompt row's f32 residual after the last layer (ResidualAll). slots != ""
// turns C' expert staging on with that many slots (fewer than the expert count, so the cache evicts) and on selects the
// expert-major MoE prefill; slots == "" is the fully resident build, whose prefill is the per-row loop.
func emResidualRows(t *testing.T, dir string, on bool, slots string) [][]float32 {
	t.Helper()
	if slots != "" {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", slots)
	} else {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", "")
	}
	t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", map[bool]string{true: "1", false: "0"}[on])
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("cuda load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || r.cacheExperts != (slots != "") {
		t.Skipf("the resident did not engage as asked (C' staging wanted: %v)", slots != "")
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
	rows, err := r.ResidualAll(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("ResidualAll: %v", err)
	}
	if ran := cudaMoeExpertMajorRuns != before; ran != on {
		t.Fatalf("expert-major ran = %v with the flag %v: the arm is not the one named", ran, on)
	}
	return rows
}

// TestMoEExpertMajorCUDA_residualIdentical holds the expert-major MoE prefill to the per-row path on the f32 residual of
// EVERY prompt row, not on the last row's logits. The residual is where the two paths' arithmetic lands; the logits pass
// through an int8 requantization that hides a last-bit difference in it (TestMoEExpertMajorCUDA_bitIdentical compares
// logits and passed while the residuals differed). It also holds C' to the fully resident build's residuals.
func TestMoEExpertMajorCUDA_residualIdentical(t *testing.T) {
	// The W4A8 decode, which this test holds to the W4A8 CPU or prefill path; the float-activation decode a Gemma 4 MoE
	// takes by default has its own gate (TestGemma4MoE_floatParity).
	floatActOffForTest = true
	t.Cleanup(func() { floatActOffForTest = false })
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	for _, dir := range []string{"../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3", "../testdata/gemma4-moe-tiny", "../testdata/gemma4-moe-tiny-k3"} {
		t.Run(dir, func(t *testing.T) {
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture at %s: %v", dir, err)
			}
			off := emResidualRows(t, dir, false, "3")
			on := emResidualRows(t, dir, true, "3")
			if len(off) != len(on) || len(off) == 0 {
				t.Fatalf("rows off=%d on=%d", len(off), len(on))
			}
			diff, total, rowsDiff := 0, 0, 0
			for m := range off {
				d := 0
				for i := range off[m] {
					if math.Float32bits(off[m][i]) != math.Float32bits(on[m][i]) {
						d++
					}
				}
				diff += d
				total += len(off[m])
				if d > 0 {
					rowsDiff++
				}
			}
			if diff != 0 {
				t.Errorf("expert-major differs from the per-row path on %d of %d residual elements (%d of %d rows)", diff, total, rowsDiff, len(off))
			} else {
				t.Logf("expert-major equals the per-row path on all %d residual elements of %d rows", total, len(off))
			}
			// And the fully resident build, whose prefill never takes expert-major: C' must give its residuals.
			res := emResidualRows(t, dir, false, "")
			d2 := 0
			for m := range res {
				for i := range res[m] {
					if math.Float32bits(res[m][i]) != math.Float32bits(on[m][i]) {
						d2++
					}
				}
			}
			if len(res) != len(on) || d2 != 0 {
				t.Errorf("C' with expert-major differs from the fully resident build on %d of %d residual elements", d2, total)
			}
		})
	}
}
