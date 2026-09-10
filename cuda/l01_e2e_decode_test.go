//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestL01_e2eDecode_matchesBaseline is the wiring-level correctness gate
// docs/task-l01-hybrid-moe-cpu-gpu.md §9 calls for before the freshly-wired
// loadRoutedExperts/moeMLPPost changes (cuda/resident.go) are trusted: does turning
// GOINFER_CUDA_L01_CPU_OFFLOAD on change what the model actually decodes?
//
// TestL01_cpuExtraction_matchesModelOwnWeights (l01_cpu_offload_test.go) already proved the
// EXTRACTION is correct in isolation — right permutation, right fields, right scales — by calling
// l01ComputeExpert directly. It never touches loadRoutedExperts or moeMLPPost, so it cannot catch
// a wiring mistake in either (wrong mask indexing, a stale hostWgt read, a missing unadmit, a merge
// added twice, etc.). This test runs the actual decode path both ways and compares the two
// resulting logit streams — the same "independent-quantization noise vs a structural bug" gate the
// other resident-parity tests in this package use (see TestGemma4MoEScaled_residentParity), applied
// here to the ON/OFF axis instead of a CUDA/CPU axis.
//
// At pos 0 the per-layer expert LRU is empty, so with the flag on EVERY routed expert at every MoE
// layer is a miss and gets routed to CPU instead of DMA'd — this is deliberately the hardest case,
// maximum CPU-path exercise, not a corner nobody hits.
func TestL01_e2eDecode_matchesBaseline(t *testing.T) {
	const dir = "../testdata/qwen35-tiny"
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}

	run := func(t *testing.T, l01 bool) [][]float32 {
		if l01 {
			t.Setenv("GOINFER_CUDA_L01_CPU_OFFLOAD", "1")
		} else {
			t.Setenv("GOINFER_CUDA_L01_CPU_OFFLOAD", "")
		}
		mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Skipf("cuda load: %v", err)
		}
		defer mc.Close()
		rf, ok := mc.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Skip("not resident on cuda")
		}
		if !rf.cacheExperts {
			t.Fatal("TAUTOLOGY GUARD: cacheExperts did not engage — nothing routed through loadRoutedExperts/moeMLPPost and this run would prove nothing")
		}
		if l01 && !rf.l01Enabled {
			t.Fatal("TAUTOLOGY GUARD: l01Enabled did not engage even with the env var set — the ON arm would be identical to the OFF arm")
		}
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("pos %d: %v", i, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}

	var base, withL01 [][]float32
	t.Run("baseline_off", func(t *testing.T) { base = run(t, false) })
	t.Run("l01_on", func(t *testing.T) { withL01 = run(t, true) })
	if base == nil || withL01 == nil {
		t.Skip("one arm skipped (no cuda hardware) — nothing to compare")
	}

	for i := range prompt {
		cos, maxAbs := cosMaxAbs(base[i], withL01[i])
		am, aw := argmaxF(base[i]), argmaxF(withL01[i])
		t.Logf("pos %2d  cosine=%.8f maxAbsDiff=%.6g  argmax off=%d on=%d", i, cos, maxAbs, am, aw)
		// Same tolerance rationale as TestL01_cpuExtraction_matchesModelOwnWeights: the CPU
		// extraction path and the GPU DMA path read the identical pre-quantized pinned-host
		// bytes, so turning L-01 on should only perturb output by the same order of rounding
		// noise a single extra dequant/requant round trip introduces — not by a structural
		// amount. A wrong mask/index/merge bug would show as a large divergence (or an
		// outright NaN/inf), not a few ULPs.
		const wantCosine = 0.999
		const wantMaxAbs = 5e-2
		if cos < wantCosine || maxAbs > wantMaxAbs {
			t.Errorf("pos %d: L-01 ON diverges from OFF: cosine=%.8f (want >=%v) maxAbsDiff=%.6g (want <=%v) — wiring bug likely (mask indexing, stale hostWgt, missing unadmit, double merge)",
				i, cos, wantCosine, maxAbs, wantMaxAbs)
		}
		if am != aw {
			t.Errorf("pos %d: argmax differs off=%d on=%d", i, am, aw)
		}
	}
}
