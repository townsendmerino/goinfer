//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// allKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path —
// W4A8 is validated separately; this proves ASSEMBLY, not the int4 packing again).
const allKernels = `
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_quant(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs(x[i]*rms*w[i]));
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round(x[i]*rms*w[i]*inv)),-127,127));
}
kernel void quant_vec(device const float* x[[buffer(0)]], device char* aq[[buffer(1)]],
    device float* asc[[buffer(2)]], constant uint& H[[buffer(3)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs(x[i]));
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round(x[i]*inv)),-127,127));
}
kernel void gemv_w8a8(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint n[[thread_position_in_grid]]) {
    int acc=0; device const char* brow=bq+(uint)n*K;
    for(uint k=0;k<K;k++) acc+=int(aq[k])*int(brow[k]);
    out[n]=float(acc)*asc[0]*bsc[n];
}
kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& total[[buffer(4)]],
    uint gid[[thread_position_in_grid]]) {
    if(gid>=total) return; uint hlf=hd/2; uint head=gid/hlf; uint dd=gid%hlf; uint base=head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th),s=sin(th);
    float x0=x[base+dd],x1=x[base+hlf+dd]; x[base+dd]=x0*c-x1*s; x[base+hlf+dd]=x0*s+x1*c;
}
kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device float* kc[[buffer(2)]], device float* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    kc[pos*kvDim+i]=k[i]; vc[pos*kvDim+i]=v[i];
}
kernel void attention(device const float* q[[buffer(0)]], device const float* kc[[buffer(1)]],
    device const float* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], uint qh[[thread_position_in_grid]]) {
    if(qh>=nH) return; uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV); uint qb=qh*hd; uint kb=kvh*hd;
    float acc[128]; for(uint d=0;d<hd;d++) acc[d]=0; float m=-INFINITY,l=0;
    for(uint s=0;s<nKeys;s++){ float sc=0; for(uint d=0;d<hd;d++) sc+=q[qb+d]*kc[s*kvDim+kb+d]; sc*=scale;
        float mn=max(m,sc); float a=exp(m-mn),p=exp(sc-mn); l=l*a+p;
        for(uint d=0;d<hd;d++) acc[d]=acc[d]*a+p*vc[s*kvDim+kb+d]; m=mn; }
    for(uint d=0;d<hd;d++) out[qb+d]=acc[d]/l;
}
kernel void swiglu_quant(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    for(uint i=tid;i<I;i+=tgs){ float s=(g[i]/(1+exp(-g[i])))*u[i]; mx=max(mx,fabs(s)); }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    for(uint i=tid;i<I;i+=tgs){ float s=(g[i]/(1+exp(-g[i])))*u[i]; dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
}
kernel void residual(device float* x[[buffer(0)]], device const float* y[[buffer(1)]], uint i[[thread_position_in_grid]]) { x[i]+=y[i]; }
`

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
		for n := 0; n < out; n++ {
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
	kc, vc := d.NewBufferLen((pos+1)*kvDim), d.NewBufferLen((pos+1)*kvDim)
	copy(kc.Floats()[:pos*kvDim], kcHist)
	copy(vc.Floats()[:pos*kvDim], vcHist)
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
	uScale, uEps := d.NewBufferFloats([]float32{scale}), d.NewBufferFloats([]float32{eps})
	uInvf := d.NewBufferFloats(invf)
	uQtotal, uKtotal := d.NewBufferU32(uint32(nH*half)), d.NewBufferU32(uint32(nKV*half))

	q := d.NewCommandQueue()
	enc := q.begin()
	enc.dispatch(pRms, 256, 256, x, d.NewBufferFloats(attnNorm), aq, aSc, uH, uEps)
	enc.dispatch(pGemv, nH*hd, 64, aq, aSc, qqW, qqS, qB, uH)
	enc.dispatch(pGemv, kvDim, 64, aq, aSc, kqW, kqS, kB, uH)
	enc.dispatch(pGemv, kvDim, 64, aq, aSc, vqW, vqS, vB, uH)
	enc.dispatch(pRope, nH*half, 64, qB, uInvf, uHd, uPos, uQtotal)
	enc.dispatch(pRope, nKV*half, 64, kB, uInvf, uHd, uPos, uKtotal)
	enc.dispatch(pKv, kvDim, 64, kB, vB, kc, vc, uKvDim, uPos)
	enc.dispatch(pAttn, nH, 32, qB, kc, vc, ctx, uNH, uNKV, uHd, uNKeys, uScale)
	enc.dispatch(pQv, 256, 256, ctx, cq, cSc, uHH)
	enc.dispatch(pGemv, H, 64, cq, cSc, oqW, oqS, oO, uHH)
	enc.dispatch(pRes, H, 64, x, oO)
	enc.dispatch(pRms, 256, 256, x, d.NewBufferFloats(mlpNorm), mq, mSc, uH, uEps)
	enc.dispatch(pGemv, I, 64, mq, mSc, gqW, gqS, gO, uH)
	enc.dispatch(pGemv, I, 64, mq, mSc, uqW, uqS, uO, uH)
	enc.dispatch(pSw, 256, 256, gO, uO, dq, dSc, uI)
	enc.dispatch(pGemv, H, 64, dq, dSc, dqW, dqS, dO, uI)
	enc.dispatch(pRes, H, 64, x, dO)
	enc.end()

	got := x.Floats()
	var dot, na, nb, maxabs float64
	for i := 0; i < H; i++ {
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
