//go:build darwin

package metal

import (
	"runtime"
	"testing"
	"time"
)

// TestProfileKernels — per-kernel wall-time at the real qwen2.5-coder-1.5b dims, so we
// know WHERE the ~50 ms/token goes before optimizing (the profile the earlier guessing
// skipped). Each kernel is encoded R times into one command buffer (amortizing the
// ~161 µs commit/wait), best-of-N warm; reports µs/dispatch and the per-token contribution.
func TestProfileKernels(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	const H, I, nH, nKV, hd = 1536, 8960, 12, 2, 128
	kvDim := nKV * hd
	V := 151936
	// dummy buffers big enough for any kernel here.
	w := d.NewBufferUint32s(make([]uint32, V*(H/8))) // int4 weights (largest: lm head)
	sc := d.NewBufferFloats(make([]float32, V*(H/32)))
	wI := d.NewBufferUint32s(make([]uint32, (2*I)*(H/8))) // gate/up
	scI := d.NewBufferFloats(make([]float32, (2*I)*(H/32)))
	wD := d.NewBufferUint32s(make([]uint32, H*(I/8))) // down (K=I)
	scD := d.NewBufferFloats(make([]float32, H*(I/32)))
	aqI := byteBuf(d, I)
	aScB := d.NewBufferLen(1)
	bias := d.NewBufferFloats(make([]float32, V))
	outBig := d.NewBufferLen(V)
	uHb, uIb := d.NewBufferU32(H), d.NewBufferU32(I)
	fH, fH2 := d.NewBufferFloats(make([]float32, H)), d.NewBufferFloats(make([]float32, H))
	fI, fI2 := d.NewBufferFloats(make([]float32, I)), d.NewBufferFloats(make([]float32, I))
	kc := d.NewBufferLen(64 * kvDim)
	uNH, uNKV, uHd, uNK, uSc, uEps := d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd), d.NewBufferU32(32), d.NewBufferFloats([]float32{0.08}), d.NewBufferFloats([]float32{1e-6})

	q := d.NewCommandQueue()
	prof := func(name string, reps int, run func(reps int)) time.Duration {
		for range 4 {
			run(reps)
		}
		best := time.Hour
		for range 20 {
			t0 := time.Now()
			run(reps)
			if dt := time.Since(t0); dt < best {
				best = dt
			}
		}
		per := best / time.Duration(reps)
		t.Logf("%-22s %7.1f us/dispatch", name, float64(per.Nanoseconds())/1e3)
		return per
	}
	pGemvBias, pGemvC, pGemvR := pipe("gemv_w4a8_bias"), pipe("gemv_w4a8_coal"), pipe("gemv_w4a8_resid")
	pRms, pSw, pAttn := pipe("rmsnorm_quant"), pipe("swiglu_quant"), pipe("attention")
	pGemvSA := pipe("gemv_w4a8_sa")

	qkv := prof("qkv gemv (2048xK1536)", 200, func(r int) { q.Run1DBatch(pGemvBias, 2048*32, 32, r, w, sc, aqI, aScB, outBig, bias, uHb) })
	gu := prof("gate/up gemv (17920xK1536)", 200, func(r int) { q.Run1DBatch(pGemvC, (2*I)*32, 32, r, wI, scI, aqI, aScB, outBig, uHb) })
	guSA := prof("gate/up SA (17920xK1536)", 200, func(r int) { q.Run1DBatch(pGemvSA, (2*I)*32, 256, r, wI, scI, aqI, aScB, outBig, uHb) })
	_ = guSA
	down := prof("down gemv (1536xK8960)", 200, func(r int) { q.Run1DBatch(pGemvR, H*32, 32, r, wD, scD, aqI, aScB, fH, uIb) })
	oproj := prof("o gemv (1536xK1536)", 200, func(r int) { q.Run1DBatch(pGemvR, H*32, 32, r, w, sc, aqI, aScB, fH, uHb) })
	lm := prof("lm head gemv (151936xK1536)", 60, func(r int) { q.Run1DBatch(pGemvC, V*32, 32, r, w, sc, aqI, aScB, outBig, uHb) })
	rms := prof("rmsnorm_quant (H1536)", 400, func(r int) { q.Run1DBatch(pRms, 256, 256, r, fH, fH2, aqI, aScB, uHb, uEps) })
	sw := prof("swiglu_quant (I8960)", 400, func(r int) { q.Run1DBatch(pSw, 256, 256, r, fI, fI2, aqI, aScB, uIb) })
	at := prof("attention (nH12,nKeys32)", 400, func(r int) { q.Run1DBatch(pAttn, nH*128, 128, r, fH, kc, kc, fH2, uNH, uNKV, uHd, uNK, uSc) })

	// per-token budget: 28 layers × (qkv + gate/up + down + o + 2×rms + swiglu + attn) + lm head
	perLayer := qkv + gu + down + oproj + 2*rms + sw + at
	total := 28*perLayer + lm
	t.Logf("---- per-token estimate: 28×%.0fus + lm %.0fus = %.1f ms/token (%.0f tok/s) ----",
		float64(perLayer.Microseconds()), float64(lm.Microseconds()), float64(total.Microseconds())/1000, 1e6/float64(total.Microseconds()))
	t.Logf("  gemv share/layer: qkv %.0f + gate/up %.0f + down %.0f + o %.0f us (of %.0f)",
		float64(qkv.Microseconds()), float64(gu.Microseconds()), float64(down.Microseconds()), float64(oproj.Microseconds()), float64(perLayer.Microseconds()))
}
