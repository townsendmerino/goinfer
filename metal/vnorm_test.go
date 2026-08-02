//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestVNorm_scaleless proves the mechanism 9c Step 3 uses for Gemma 4's K=V layers: a scale-less
// per-head v_norm (V = v_norm(raw k), no learned weight) computed by REUSING the qk_norm kernel
// with nH=0 (every head takes the K branch), a UNIT-1.0 weight, and addOne=0. The trap this
// isolates (flagged on the CUDA port): qk_norm computes `x·rms·(addOne ? 1+w : w)`, so the
// scale-less identity requires w=1.0 AND addOne=0. Get the convention backwards — w=1.0 with
// addOne=1 — and you get x·rms·2, a silent 2× on V that reads as "cosine is slightly off" in a
// full forward but is instant and unambiguous here. So this asserts BOTH: (a) unit+addOne=0
// matches a CPU scale-less RMSNorm to f32 tolerance, and (b) unit+addOne=1 is exactly 2× that —
// pinning the convention so a future kernel edit can't flip it unnoticed.
func TestVNorm_scaleless(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pQK, err := d.NewComputePipeline(lib, "qk_norm")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const nKV, hd = 2, 512 // Gemma 4 global geometry (K=V lives on the globals)
	const eps = 1e-6
	rng := rand.New(rand.NewSource(3))
	in := make([]float32, nKV*hd)
	for i := range in {
		in[i] = float32(rng.NormFloat64()) * 0.5
	}
	// Outlier so a wrong (non-scale-less, or 2×) result is unmissable, not lost in the average.
	in[hd/2] = 30

	// CPU oracle: scale-less RMSNorm per head — rms = rsqrt(mean(x²)+eps); out = x·rms (weight 1).
	want := make([]float32, nKV*hd)
	for h := 0; h < nKV; h++ {
		var ss float64
		for i := 0; i < hd; i++ {
			v := float64(in[h*hd+i])
			ss += v * v
		}
		rms := 1.0 / math.Sqrt(ss/float64(hd)+eps)
		for i := 0; i < hd; i++ {
			want[h*hd+i] = float32(float64(in[h*hd+i]) * rms)
		}
	}

	unit := make([]float32, hd)
	for i := range unit {
		unit[i] = 1
	}
	run := func(addOne uint32) []float32 {
		buf := d.NewBufferFloats(append([]float32(nil), in...))
		uW := d.NewBufferFloats(unit)
		u0 := d.NewBufferU32(0) // nH=0 and nHhd=0 (V slot addressed at base 0+head*hd within buf)
		uNKV, uHd := d.NewBufferU32(nKV), d.NewBufferU32(hd)
		uEps := d.NewBufferFloats([]float32{eps})
		uAdd := d.NewBufferU32(addOne)
		cq := d.NewCommandQueue()
		enc := cq.Begin()
		enc.Dispatch(pQK, nKV*128, 128, buf, uW, uW, u0, uNKV, uHd, u0, uEps, uAdd)
		enc.End()
		return append([]float32(nil), buf.Floats()...)
	}

	// (a) unit weight + addOne=0 == CPU scale-less v_norm.
	got := run(0)
	var maxAbs float64
	for i := range want {
		if dd := math.Abs(float64(got[i] - want[i])); dd > maxAbs {
			maxAbs = dd
		}
	}
	if maxAbs > 1e-3 {
		t.Errorf("scale-less v_norm (unit,addOne=0) != CPU oracle: maxAbs=%.2e", maxAbs)
	}
	t.Logf("v_norm(unit, addOne=0) matches CPU scale-less RMSNorm: maxAbs=%.2e", maxAbs)

	// (b) the trap: unit weight + addOne=1 is x·rms·(1+1) = 2× the scale-less result. Pinning this
	// makes the "unit is 1.0 for x·w, not x·(1+w)" convention a test failure if anyone flips it.
	got2 := run(1)
	var maxRatioErr float64
	for i := range want {
		if math.Abs(float64(want[i])) < 1e-4 {
			continue
		}
		ratio := float64(got2[i]) / float64(want[i])
		if e := math.Abs(ratio - 2.0); e > maxRatioErr {
			maxRatioErr = e
		}
	}
	if maxRatioErr > 1e-3 {
		t.Errorf("addOne=1 with a unit weight should be exactly 2× scale-less (the convention trap); worst ratio err=%.2e", maxRatioErr)
	}
	t.Logf("convention pinned: unit weight + addOne=1 gives 2× (so v_norm MUST use addOne=0)")
}
