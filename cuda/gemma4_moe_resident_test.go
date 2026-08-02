//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoE_residentParity is Split-B task 2c's END-TO-END gate. Every primitive underneath is
// pinned in isolation and the pos-0 per-branch parity (TestGemma4MoE_localize) proves the wiring, so
// this asserts kernel correctness at the logit level (pos 0) and then CHARACTERIZES the multi-position
// int4-vs-int4 drift with two calibrated instruments instead of a picked-below-observed floor:
//
//  1. ROUTING AGREEMENT at EVERY position, not just pos 0. rn derives from h derives from the KV
//     cache, so as attention drifts a top-k flip at position N becomes possible even though pos 0 is
//     clean — and a flipped expert reads IDENTICAL to accumulation in a cosine (same shape, same
//     "grows with position"). If resident idx == CPU idx at every position, accumulation is the only
//     explanation left; a flip means the 0.87 has a discrete component.
//  2. A CALIBRATED curve: CUDA-int4-vs-CPU-int4 vs CPU-int4-vs-CPU-f32 at the same positions. The
//     latter is "as well as int4 arithmetic can agree with f32". CUDA-vs-CPU-int4 (same weights, only
//     W4A8 activation rounding differs) should track it or sit ABOVE it. If CUDA drops FASTER than the
//     fixture's own quantization curve, that's a real divergence no conditioning explains.
func TestGemma4MoE_residentParity(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_G4_CAPTURE", "1")
	const dir = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED gemma4 MoE with env on — admission regressed")
	}
	r := rf.(*cudaResident)
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}

	// CPU int4 over the prompt, capturing its per-decision idx (the routing reference).
	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	cpu4 := make([][]float32, len(prompt))
	c4cache := mc4.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mc4.ForwardForTest(tok, c4cache)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpu4[i] = append([]float32(nil), l...)
	}
	idxCpu4, _ := decoder.RouterCaptureForTest()
	idxCpu4 = append([][]int(nil), idxCpu4...) // snapshot before the f32 run appends more

	// CPU f32 over the prompt (the conditioning reference; idx not needed).
	cpuF := make([][]float32, len(prompt))
	cFcache := mcF.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mcF.ForwardForTest(tok, cFcache)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		cpuF[i] = append([]float32(nil), l...)
	}

	// Resident int4 over the prompt (g4capIdx accumulates per decision).
	cuda := make([][]float32, len(prompt))
	for i, tok := range prompt {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		cuda[i] = append([]float32(nil), l...)
	}

	// ---- curves: CUDA-vs-CPU-int4 vs the fixture's own CPU-int4-vs-f32 quantization curve ----
	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	cVs4, c4VsF := make([]float64, len(prompt)), make([]float64, len(prompt))
	minCudaVs4, pos0 := 1.0, 0.0
	for i := range prompt {
		cVs4[i] = cos(cpu4[i], cuda[i])  // CUDA-int4 vs CPU-int4 (only W4A8 activation rounding + reduction differ)
		c4VsF[i] = cos(cpuF[i], cpu4[i]) // CPU-int4 vs CPU-f32 (the "as well as int4 arithmetic can agree" bound)
		if i == 0 {
			pos0 = cVs4[i]
		}
		if cVs4[i] < minCudaVs4 {
			minCudaVs4 = cVs4[i]
		}
		t.Logf("  pos %2d  CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cuda=%d cpu4=%d",
			i, cVs4[i], c4VsF[i], argmaxF(cuda[i]), argmaxF(cpu4[i]))
	}

	// ---- per-position routing agreement (resident idx vs CPU-int4 idx) ----
	nMoE := len(idxCpu4) / len(prompt)
	routeAgree := true
	if len(r.g4capIdx) != len(idxCpu4) {
		t.Fatalf("decision-count mismatch: resident %d vs cpu %d", len(r.g4capIdx), len(idxCpu4))
	}
	for k := range idxCpu4 {
		if !sameExpertSet(idxCpu4[k], r.g4capIdx[k]) {
			routeAgree = false
			t.Errorf("ROUTING FLIP at decision %d (pos %d, moe-layer %d): cpu idx %v vs resident %v — the 0.87 "+
				"has a DISCRETE component, not pure accumulation", k, k/nMoE, k%nMoE, idxCpu4[k], r.g4capIdx[k])
		}
	}
	if routeAgree {
		t.Logf("routing agreement: resident idx == CPU-int4 idx at ALL %d decisions — the multi-position drift is "+
			"pure accumulation, no expert flip", len(idxCpu4))
	}

	t.Logf("gemma4 MoE resident: pos0=%.6f | min CUDA-vs-CPUint4=%.6f", pos0, minCudaVs4)

	// ---- gates ----
	if pos0 < 0.97 {
		t.Errorf("pos-0 cosine %.6f < 0.97 — kernel divergence at the first token (GOINFER_G4_CAPTURE / "+
			"TestGemma4MoE_localize to localize)", pos0)
	}
	// CALIBRATED, per position (faithful to "does CUDA track the fixture's own quantization curve").
	// CUDA-vs-CPU-int4 differs from CPU only in W4A8 activation rounding — a SMALLER perturbation than
	// the full int4 weight quantization CPUint4-vs-f32 measures — so CUDA should sit at or above that
	// curve at every position. Slack absorbs the two perturbations compounding differently; a drop
	// beyond it is a real divergence conditioning can't explain (a picked-below-observed floor would
	// only mean "lower than today"). Here CUDA is well above at 7/8 positions and 0.018 below at pos 7.
	const slack = 0.05
	for i := range prompt {
		if cVs4[i] < c4VsF[i]-slack {
			t.Errorf("pos %d: CUDA-vs-CPUint4 %.6f dropped %.3f below the CPU-int4-vs-f32 curve (%.6f) — CUDA "+
				"diverges FASTER than the fixture's own int4 quantization, so this is a real divergence, NOT conditioning",
				i, cVs4[i], c4VsF[i]-cVs4[i], c4VsF[i])
		}
	}
}
