//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4DenseScaled_metalParity is the Metal mirror of the box's CUDA scaled-dense gate
// (cuda/gemma4_dense_scaled_test.go, f93bda1) — the DEPTH/REAL-GEOMETRY control the 26B needed and
// the tiny fixtures lacked. Scaled dense Gemma 4: hidden 1024, 12 layers, 5:1 sliding/full, REAL head
// dims 256 local / 512 global, K=V globals, sandwich, softcap 30 — fits non-paged (~55 MB int4).
//
// It settles whether the 26B secondary-gate divergence is a Metal composition BUG or int4 CONDITIONING
// at depth. The box found "conditioning ≠ geometry": goinfer's CPU forward is bit-right at this
// geometry (cosine 1.0 vs HF golden), and CUDA composes within the int4 envelope. The gate is the
// CALIBRATED envelope, not an absolute floor: Metal must agree with CPU-int4 at least as well ON
// AVERAGE as int4 agrees with f32 (holds by construction — activation perturbation < weight
// perturbation — unless a real kernel bug diverges faster than the fixture's own quantization).
// If Metal passes here at hd=256/512, the 26B "divergence" is int4 conditioning at 64-layer depth
// (worse with more layers + a specific prompt), NOT a Metal attention bug.
func TestGemma4DenseScaled_metalParity(t *testing.T) {
	const dir = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture (%s) — scp from the box / run scripts/pin_gemma4_dense_scaled.py", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	mg, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (metal int4): %v", err)
	}
	defer mg.Close()
	r, err := BuildResident(mg)
	if err != nil {
		t.Fatalf("BuildResident: %v — Metal declined the scaled dense geometry (256-local?)", err)
	}
	defer r.Close()
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

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200} // golden's, len 16 > window 8

	run := func(fwd func(pos, tok int) []float32) [][]float32 {
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			out[i] = append([]float32(nil), fwd(i, tok)...)
		}
		return out
	}
	c4c, cFc := mc4.NewCache(len(prompt)), mcF.NewCache(len(prompt))
	cpu4 := run(func(i, tok int) []float32 {
		l, err := mc4.ForwardForTest(tok, c4c)
		if err != nil {
			t.Fatalf("cpu int4 %d: %v", i, err)
		}
		return l
	})
	cpuF := run(func(i, tok int) []float32 {
		l, err := mcF.ForwardForTest(tok, cFc)
		if err != nil {
			t.Fatalf("cpu f32 %d: %v", i, err)
		}
		return l
	})
	metal := run(func(i, tok int) []float32 { return r.ForwardEmb(mg.EmbedResidentForTest(tok), i) })

	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	pos0, exact := 0.0, 0
	sumMetal, sumCpu, minFloor := 0.0, 0.0, 1.0
	for i := range prompt {
		mVs4 := cos(cpu4[i], metal[i])
		c4VsF := cos(cpuF[i], cpu4[i])
		if i == 0 {
			pos0 = mVs4
		}
		sumMetal += mVs4
		sumCpu += c4VsF
		if c4VsF < minFloor {
			minFloor = c4VsF
		}
		if argmaxF(metal[i]) == argmaxF(cpu4[i]) {
			exact++
		}
		t.Logf("  pos %2d  Metal-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax metal=%d cpu4=%d",
			i, mVs4, c4VsF, argmaxF(metal[i]), argmaxF(cpu4[i]))
	}
	meanMetal, meanCpu := sumMetal/float64(len(prompt)), sumCpu/float64(len(prompt))
	t.Logf("scaled dense (256-local, 12 layers): pos0=%.6f exact-argmax %d/%d | mean Metal-vs-CPUint4=%.6f  CPUint4-vs-f32=%.6f (min floor %.4f)",
		pos0, exact, len(prompt), meanMetal, meanCpu, minFloor)

	// pos-0 kernel correctness (no KV accumulation): the 256-local/512-global geometry must compose.
	if pos0 < 0.97 {
		t.Errorf("pos-0 Metal-vs-CPUint4 %.6f < 0.97 — the 256-local/512-global resident geometry diverges at the first token (a real bug)", pos0)
	}
	// CALIBRATED run-mean envelope: Metal must agree with CPU-int4 at least as well ON AVERAGE as int4
	// agrees with f32. A real kernel bug sinks the mean below the fixture's own quantization curve.
	if meanMetal < meanCpu {
		t.Errorf("mean Metal-vs-CPUint4 %.6f < mean CPUint4-vs-f32 %.6f — Metal diverges FASTER than the fixture's int4 quantization: a real bug, not conditioning",
			meanMetal, meanCpu)
	}
}
