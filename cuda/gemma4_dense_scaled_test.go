//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4DenseScaled_residentParity exercises the CUDA resident bridge on a SCALED dense Gemma 4
// (hidden 1024, 12 layers, 5:1 sliding/full, REAL head dims 256 local / 512 global, K=V globals) —
// closing the 256-local geometry gap the tiny Split-A fixture (hd=16) left. Gated exactly as Split A
// / the MoE 2c gate: pos-0 kernel correctness + the calibrated per-position curve (568f292),
// CUDA-int4-vs-CPU-int4 measured against the fixture's own CPU-int4-vs-f32 quantization curve so a
// chaotic int4 floor doesn't masquerade as a kernel bug.
//
// FINDING baked into the gate: random weights over 12 layers are LESS int4-conditioned than the tiny
// 2-layer fixture (floor ~0.46 vs 0.79) — realistic geometry ≠ realistic conditioning (that needs
// trained weights). So the multi-position int4-vs-int4 drift is REPORTED, and the gate is the run
// mean vs that curve, not an absolute floor. No coherence gate: random weights → degenerate greedy.
func TestGemma4DenseScaled_residentParity(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	const dir = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_dense_scaled.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED scaled dense Gemma 4 with env on — admission regressed (256-local geometry?)")
	}
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

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200} // the golden's, len 16 > window 8

	cpu4 := make([][]float32, len(prompt))
	c4c := mc4.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mc4.ForwardForTest(tok, c4c)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpu4[i] = append([]float32(nil), l...)
	}
	cpuF := make([][]float32, len(prompt))
	cFc := mcF.NewCache(len(prompt))
	for i, tok := range prompt {
		l, err := mcF.ForwardForTest(tok, cFc)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		cpuF[i] = append([]float32(nil), l...)
	}
	cuda := make([][]float32, len(prompt))
	for i, tok := range prompt {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		cuda[i] = append([]float32(nil), l...)
	}

	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	cVs4, c4VsF := make([]float64, len(prompt)), make([]float64, len(prompt))
	pos0, exact := 0.0, 0
	sumCuda, sumCpu, minFloor := 0.0, 0.0, 1.0
	for i := range prompt {
		cVs4[i] = cos(cpu4[i], cuda[i])
		c4VsF[i] = cos(cpuF[i], cpu4[i])
		if i == 0 {
			pos0 = cVs4[i]
		}
		sumCuda += cVs4[i]
		sumCpu += c4VsF[i]
		if c4VsF[i] < minFloor {
			minFloor = c4VsF[i]
		}
		if argmaxF(cuda[i]) == argmaxF(cpu4[i]) {
			exact++
		}
		t.Logf("  pos %2d  CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cuda=%d cpu4=%d",
			i, cVs4[i], c4VsF[i], argmaxF(cuda[i]), argmaxF(cpu4[i]))
	}
	meanCuda, meanCpu := sumCuda/float64(len(prompt)), sumCpu/float64(len(prompt))
	t.Logf("scaled dense (256-local): pos0=%.6f exact-argmax %d/%d | mean CUDA-vs-CPUint4=%.6f  CPUint4-vs-f32=%.6f (min floor %.4f)",
		pos0, exact, len(prompt), meanCuda, meanCpu, minFloor)

	// pos-0 kernel correctness (no KV accumulation): the 256-local + 512-global geometry must compose
	// correctly. cuda-vs-cpu-int4 differs only in W4A8 activation rounding, so pos 0 is close even when
	// the int4-vs-f32 floor is chaotic.
	if pos0 < 0.97 {
		t.Errorf("pos-0 CUDA-vs-CPUint4 %.6f < 0.97 — the 256-local/512-global resident geometry diverges from CPU at the first token", pos0)
	}
	// CALIBRATED run-mean: CUDA must agree with CPU-int4 at least as well ON AVERAGE as int4 agrees
	// with f32. Holds by construction (activation perturbation < weight perturbation) regardless of how
	// chaotic the fixture's conditioning is. A real divergence sinks the mean below the baseline.
	if meanCuda < meanCpu {
		t.Errorf("mean CUDA-vs-CPUint4 %.6f < mean CPUint4-vs-f32 %.6f — CUDA diverges faster than the fixture's own int4 quantization: a real bug, not conditioning",
			meanCuda, meanCpu)
	}
}
