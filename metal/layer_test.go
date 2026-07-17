//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// q8row quantizes a f32 row to per-row symmetric int8 (goinfer W8A8).
func q8row(row []float32) ([]int8, float32) {
	var mx float32
	for _, v := range row {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	sc := mx / 127
	if sc == 0 {
		sc = 1
	}
	q := make([]int8, len(row))
	for i, v := range row {
		r := math.Round(float64(v / sc))
		q[i] = int8(math.Max(-127, math.Min(127, r)))
	}
	return q, sc
}

// TestLayerB_fullLayerForward assembles ALL kernels into one dense decode layer, encoded
// into ONE command buffer (the tax requirement), and validates the whole layer's output
// vs a CPU reference — synthetic weights, so it proves the ASSEMBLY + inter-kernel data
// flow (not any single kernel, which are validated elsewhere).
func TestLayerB_fullLayerForward(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			t.Fatalf("pipeline %s: %v", name, e)
		}
		return p
	}
	pRms, pQv, pGemv := pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w8a8")
	pRope, pKv, pAttn := pipe("rope"), pipe("kv_store"), pipe("attention")
	pSw, pRes := pipe("swiglu_quant"), pipe("residual")
	// rmsnorm_quant/swiglu_quant are parameterized for Gemma: addOne selects the (1+w) RMS offset
	// and act selects SiLU vs GELU-tanh. This layer is a Llama/Qwen block — plain w, SwiGLU. These
	// MUST be bound: an unbound uniform reads garbage, and a nonzero act silently runs GeGLU.
	uAddOne0, uActSiLU := d.NewBufferU32(0), d.NewBufferU32(1)

	const H, nH, nKV, hd, I, pos = 256, 4, 2, 64, 512, 5
	const eps = 1e-6
	kvDim := nKV * hd
	half := hd / 2
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(42))
	rvec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	// weights (row-major [out,in]) + norms
	rmat := func(out, in int) []float32 { return rvec(out * in) }
	x0 := rvec(H)
	attnNorm, mlpNorm := rvec(H), rvec(H)
	Wq, Wk, Wv, Wo := rmat(nH*hd, H), rmat(kvDim, H), rmat(kvDim, H), rmat(H, nH*hd)
	Wg, Wu, Wd := rmat(I, H), rmat(I, H), rmat(H, I)
	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}
	// pre-existing KV history (pos positions already stored)
	kcHist, vcHist := rvec(pos*kvDim), rvec(pos*kvDim)

	// ---------- CPU reference ----------
	ref := cpuLayer(x0, attnNorm, mlpNorm, Wq, Wk, Wv, Wo, Wg, Wu, Wd, invf, kcHist, vcHist,
		H, nH, nKV, hd, I, pos, eps, scale)

	// ---------- GPU: pack weights, upload, encode one command buffer ----------
	packMat := func(w []float32, out, in int) (Buffer, Buffer) {
		bq := make([]int8, out*in)
		bs := make([]float32, out)
		for n := range out {
			q, s := q8row(w[n*in : (n+1)*in])
			copy(bq[n*in:(n+1)*in], q)
			bs[n] = s
		}
		return d.NewBufferInt8(bq), d.NewBufferFloats(bs)
	}
	qqW, qqS := packMat(Wq, nH*hd, H)
	kqW, kqS := packMat(Wk, kvDim, H)
	vqW, vqS := packMat(Wv, kvDim, H)
	oqW, oqS := packMat(Wo, H, nH*hd)
	gqW, gqS := packMat(Wg, I, H)
	uqW, uqS := packMat(Wu, I, H)
	dqW, dqS := packMat(Wd, H, I)

	byteBuf := func(n int) Buffer { return Buffer{id: d.id.Send(selNewBufferLen, uintptr(n), uintptr(0)), n: n} }
	x := d.NewBufferFloats(x0)
	aq, aSc := byteBuf(H), d.NewBufferLen(1)
	qB, kB, vB := d.NewBufferLen(nH*hd), d.NewBufferLen(kvDim), d.NewBufferLen(kvDim)
	// f16 KV cache: history uploaded as half, room for the new position (kv_store writes it).
	kh, vh := make([]uint16, (pos+1)*kvDim), make([]uint16, (pos+1)*kvDim)
	for i := 0; i < pos*kvDim; i++ {
		kh[i], vh[i] = f32ToF16(kcHist[i]), f32ToF16(vcHist[i])
	}
	kc, vc := d.NewBufferU16s(kh), d.NewBufferU16s(vh)
	ctx := d.NewBufferLen(nH * hd)
	cq, cSc := byteBuf(nH*hd), d.NewBufferLen(1)
	oO := d.NewBufferLen(H)
	mq, mSc := byteBuf(H), d.NewBufferLen(1)
	gO, uO := d.NewBufferLen(I), d.NewBufferLen(I)
	dq, dSc := byteBuf(I), d.NewBufferLen(1)
	dO := d.NewBufferLen(H)
	uHd, uKvDim := d.NewBufferU32(hd), d.NewBufferU32(uint32(kvDim))
	uPos, uH, uI := d.NewBufferU32(pos), d.NewBufferU32(H), d.NewBufferU32(I)
	uHH := d.NewBufferU32(uint32(nH * hd))
	uNH, uNKV, uNKeys := d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(pos+1)
	uWindow0 := d.NewBufferU32(0) // full causal — the attention kernel's window arg (buffer 9)
	uScale, uEps := d.NewBufferFloats([]float32{scale}), d.NewBufferFloats([]float32{eps})
	uInvf := d.NewBufferFloats(invf)
	uQtotal, uKtotal := d.NewBufferU32(uint32(nH*half)), d.NewBufferU32(uint32(nKV*half))
	uHalf := d.NewBufferU32(uint32(half)) // rope rhalf (buffer 5): rotaryDim/2, full rotary here

	q := d.NewCommandQueue()
	enc := q.begin()
	enc.dispatch(pRms, 256, 256, x, d.NewBufferFloats(attnNorm), aq, aSc, uH, uEps, uAddOne0)
	enc.dispatch(pGemv, nH*hd, 64, aq, aSc, qqW, qqS, qB, uH)
	enc.dispatch(pGemv, kvDim, 64, aq, aSc, kqW, kqS, kB, uH)
	enc.dispatch(pGemv, kvDim, 64, aq, aSc, vqW, vqS, vB, uH)
	enc.dispatch(pRope, nH*half, 64, qB, uInvf, uHd, uPos, uQtotal, uHalf)
	enc.dispatch(pRope, nKV*half, 64, kB, uInvf, uHd, uPos, uKtotal, uHalf)
	enc.dispatch(pKv, kvDim, 64, kB, vB, kc, vc, uKvDim, uPos)
	enc.dispatch(pAttn, nH*128, 128, qB, kc, vc, ctx, uNH, uNKV, uHd, uNKeys, uScale, uWindow0) // threadgroup-per-head
	enc.dispatch(pQv, 256, 256, ctx, cq, cSc, uHH)
	enc.dispatch(pGemv, H, 64, cq, cSc, oqW, oqS, oO, uHH)
	enc.dispatch(pRes, H, 64, x, oO)
	enc.dispatch(pRms, 256, 256, x, d.NewBufferFloats(mlpNorm), mq, mSc, uH, uEps, uAddOne0)
	enc.dispatch(pGemv, I, 64, mq, mSc, gqW, gqS, gO, uH)
	enc.dispatch(pGemv, I, 64, mq, mSc, uqW, uqS, uO, uH)
	enc.dispatch(pSw, 256, 256, gO, uO, dq, dSc, uI, uActSiLU)
	enc.dispatch(pGemv, H, 64, dq, dSc, dqW, dqS, dO, uI)
	enc.dispatch(pRes, H, 64, x, dO)
	enc.end()

	got := x.Floats()
	var dot, na, nb, maxabs float64
	for i := range H {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxabs {
			maxabs = dd
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.9999 || maxabs > 1e-2 {
		t.Fatalf("full-layer parity FAIL: cosine=%.7f maxAbs=%.2e (got[0]=%v ref[0]=%v)", cos, maxabs, got[0], ref[0])
	}
	t.Logf("FULL dense decode layer (H=%d nH=%d/%d hd=%d I=%d, 17 dispatches in ONE command buffer) "+
		"on Metal GPU (cgo-free) vs CPU: cosine=%.7f maxAbs=%.2e — PARITY", H, nH, nKV, hd, I, cos, maxabs)
}
