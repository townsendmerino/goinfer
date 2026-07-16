//go:build darwin

package metal

import (
	"fmt"
	"runtime"
)

// Prefill kernels — the f16 simdgroup_matrix (MMA) path for fast prompt ingestion. Unlike the
// int4/scalar-MAC decode path (which can't amortize batching), an MMA GEMM reuses each weight
// across all M prompt rows → ~2.5× the per-token GEMV, flat with M. Activations flow in f16
// (no int8 quant); weights stay int4 and are dequanted to f16 in-kernel (no extra RAM). Kept in
// a SEPARATE library from allKernels so the decode path is unchanged and prefill is opt-in.
const prefillKernels = `
#include <metal_stdlib>
using namespace metal;

// Blocked int4→f16 MMA GEMM: C[M×N] = A[M×K](f16) · Wᵀ, W = resident int4/W4A8 (packed nibbles
// + f16 group scales), dequanted in-kernel. Each simdgroup owns RPS row-tiles × one 8-col output
// tile; dequants each 8×8 weight tile once (transposed to Wᵀ) and reuses across the RPS rows.
#define RPS 4
kernel void gemm_w4f16(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device float* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    threadgroup half wscr[8*64];
    uint sg = tgid*sgpt + sgid;
    uint tilesN = N/8u;
    uint rblk = sg / tilesN, tc = sg % tilesN;
    uint r0 = rblk*RPS;
    if (r0*8u >= M) return;
    uint n0 = tc*8u;
    threadgroup half* scr = wscr + sgid*64u;
    uint wpr = K/8u, gpr = K/32u;
    simdgroup_float8x8 acc[RPS];
    for (uint r=0;r<RPS;r++) acc[r]=make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        if (lane < 8u) {
            uint nl = lane;
            uint word = W[(n0+nl)*wpr + k/8u];
            float sc = float(WS[(n0+nl)*gpr + k/32u]);
            for (uint kl=0; kl<8u; kl++)
                scr[kl*8u + nl] = half(float(int((word >> (4u*kl)) & 0xF) - 8) * sc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 b; simdgroup_load(b, scr, 8);
        for (uint r=0;r<RPS;r++) {
            if ((r0+r)*8u >= M) break;
            simdgroup_half8x8 a; simdgroup_load(a, A + ((r0+r)*8u)*K + k, K);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<RPS;r++) {
        if ((r0+r)*8u >= M) break;
        simdgroup_store(acc[r], C + ((r0+r)*8u)*N + n0, N);
    }
}

// gemm_w4f16_bias / _resid — GEMM epilogues. bias adds a per-column bias (fused QKV); resid
// accumulates C into an existing f16 buffer (o-proj / down + residual). Both take the extra
// buffer at buffer(7). (Separate kernels keep the hot GEMM loop identical.)
kernel void gemm_w4f16_store(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    threadgroup half wscr[8*64];
    threadgroup float cscr[8*64];
    uint sg = tgid*sgpt + sgid;
    uint tilesN = N/8u;
    uint rblk = sg / tilesN, tc = sg % tilesN;
    uint r0 = rblk*RPS;
    if (r0*8u >= M) return;
    uint n0 = tc*8u;
    threadgroup half* scr = wscr + sgid*64u;
    threadgroup float* cs = cscr + sgid*64u;
    uint wpr = K/8u, gpr = K/32u;
    simdgroup_float8x8 acc[RPS];
    for (uint r=0;r<RPS;r++) acc[r]=make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        if (lane < 8u) {
            uint nl = lane;
            uint word = W[(n0+nl)*wpr + k/8u];
            float sc = float(WS[(n0+nl)*gpr + k/32u]);
            for (uint kl=0; kl<8u; kl++)
                scr[kl*8u + nl] = half(float(int((word >> (4u*kl)) & 0xF) - 8) * sc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 b; simdgroup_load(b, scr, 8);
        for (uint r=0;r<RPS;r++) {
            if ((r0+r)*8u >= M) break;
            simdgroup_half8x8 a; simdgroup_load(a, A + ((r0+r)*8u)*K + k, K);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: store via threadgroup scratch so lanes can apply bias/residual per element.
    for (uint r=0;r<RPS;r++) {
        uint mrow = r0*8u + r*8u;
        if (mrow >= M) break;
        simdgroup_store(acc[r], cs, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            for (uint ml=0; ml<8u; ml++) {
                uint m = mrow + ml, n = n0 + lane;
                float v = cs[ml*8u + lane];
                if (mode == 1u) v += bias[n];                 // fused bias (QKV)
                if (mode == 2u) v += float(C[m*N + n]);       // residual (o / down)
                C[m*N + n] = half(v);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// rmsnorm_f16: one threadgroup per row. out[m] = x[m]*rsqrt(mean(x[m]²)+eps)*w (w f32).
kernel void rmsnorm_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    uint row[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    device const half* xr = x + row*H; device half* orow = out + row*H;
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float v=float(xr[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs) orow[i]=half(float(xr[i])*rms*w[i]);
}

// residual_f16: x += y (element-wise, grid = M*H).
kernel void residual_f16(device half* x[[buffer(0)]], device const half* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])); }

// swiglu_f16: out[m][i] = silu(gu[m][i]) * gu[m][I+i], gu is [M×2I]. grid = M*I.
kernel void swiglu_f16(device const half* gu[[buffer(0)]], device half* out[[buffer(1)]],
    constant uint& I[[buffer(2)]], uint gid[[thread_position_in_grid]]) {
    uint m = gid / I, i = gid % I;
    float g=float(gu[m*2u*I + i]), u=float(gu[m*2u*I + I + i]);
    out[gid]=half((g/(1.0f+exp(-g)))*u);
}

// rope_f16: NeoX half-split, per-row position. Rotates a [M × total] region with row stride
// 'stride' starting at byte-free offset 'base0' (element offset into each row). positions[m].
kernel void rope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = total/2u;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint hlf = hd/2u; uint head = p/hlf, dd = p%hlf;
    uint base = m*stride + base0 + head*hd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th), s=sin(th);
    float x0=float(x[base+dd]), x1=float(x[base+hlf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+hlf+dd]=half(x0*s+x1*c);
}

// attention_prefill: one threadgroup per (row m, query head qh). Row m attends CAUSALLY to
// keys[0..startPos+m] over the shared f16 KV cache (GQA via kvh). q read from the fused qkv
// buffer (row stride qStride, q section at offset 0). Reuses the decode attention math (dot →
// threadgroup softmax → value accum), per row with per-row nKeys. O(M²) but that's inherent to
// prefill attention; an MMA/flash version is a later throughput lever.
kernel void attention_prefill(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint nKeys = startPos + m + 1u;
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+s*kvDim;
        for (uint d=0; d<hd; d++) a += float(qr[d])*float(k[d]);
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mmax=-INFINITY; for (uint s=tid;s<nKeys;s+=tgs) mmax=max(mmax,sc[s]);
    red[tid]=mmax; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=0;s<nKeys;s++) a += sc[s]*float(vb[s*kvDim+d]); out[m*qDim + qh*hd + d]=half(a/sum); }
}

// kv_store_f16: scatter M rows' K,V (slices of the fused qkv[M×stride]) into the f16 KV cache
// at positions[m]. grid = M*kvDim.
kernel void kv_store_f16(device const half* qkv[[buffer(0)]], device half* kc[[buffer(1)]],
    device half* vc[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& kvDim[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& kOff[[buffer(6)]], constant uint& vOff[[buffer(7)]],
    uint gid[[thread_position_in_grid]]) {
    uint m = gid / kvDim, i = gid % kvDim;
    uint pos = positions[m];
    kc[pos*kvDim + i] = qkv[m*stride + kOff + i];
    vc[pos*kvDim + i] = qkv[m*stride + vOff + i];
}
`

// prefillState holds the lazily-compiled prefill pipelines (opt-in; decode-only builds skip it).
type prefillState struct {
	pGemm, pGemmStore, pRms, pRes, pSw, pRope, pKv, pAttn Pipeline
}

func (r *Resident) ensurePrefill() {
	if r.pf != nil {
		return
	}
	lib, err := r.d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		panic(fmt.Sprintf("metal prefill compile: %v", err))
	}
	p := func(n string) Pipeline {
		pp, e := r.d.NewComputePipeline(lib, n)
		if e != nil {
			panic(fmt.Sprintf("metal prefill pipeline %s: %v", n, e))
		}
		return pp
	}
	r.pf = &prefillState{
		pGemm: p("gemm_w4f16"), pGemmStore: p("gemm_w4f16_store"), pRms: p("rmsnorm_f16"),
		pRes: p("residual_f16"), pSw: p("swiglu_f16"), pRope: p("rope_f16"),
		pKv: p("kv_store_f16"), pAttn: p("attention_prefill"),
	}
}

// PrefillLast ingests M prompt embeddings at positions startPos..startPos+M-1 in ONE command
// buffer via the f16 MMA path (weights read once, amortized across M — unlike the token-by-token
// decode loop), populating the resident KV cache, and returns the LAST token's logits[V] (what a
// generator needs to sample the first output token). Correctness-gated vs the sequential path.
func (r *Resident) PrefillLast(embs [][]float32, startPos int) []float32 {
	r.ensurePrefill()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, pf := r.d, r.pf
	M := len(embs)
	Mpad := (M + 7) / 8 * 8
	H, I, V := r.H, r.I, r.V
	nHhd := r.nH * r.hd
	kvDim := r.kvDim
	qkvDim := nHhd + 2*kvDim
	qDim := nHhd

	// f16 activation scratch (per call, sized to the padded prompt).
	xh := make([]uint16, Mpad*H)
	for m := 0; m < M; m++ {
		for i := 0; i < H; i++ {
			xh[m*H+i] = f32ToF16(embs[m][i])
		}
	}
	xF := d.NewBufferU16s(xh)
	normF := d.NewBufferU16s(make([]uint16, Mpad*H))
	qkvF := d.NewBufferU16s(make([]uint16, Mpad*qkvDim))
	ctxF := d.NewBufferU16s(make([]uint16, Mpad*qDim))
	guF := d.NewBufferU16s(make([]uint16, Mpad*2*I))
	dqF := d.NewBufferU16s(make([]uint16, Mpad*I))
	posv := make([]uint32, Mpad)
	for m := 0; m < M; m++ {
		posv[m] = uint32(startPos + m)
	}
	posB := d.NewBufferUint32s(posv)

	// uniforms
	uM := d.NewBufferU32(uint32(Mpad))
	uH := r.uH
	uI := d.NewBufferU32(uint32(I))
	uV := d.NewBufferU32(uint32(V))
	u2I := d.NewBufferU32(uint32(2 * I))
	uQkv := d.NewBufferU32(uint32(qkvDim))
	uQDim := d.NewBufferU32(uint32(qDim))
	uKvDim := r.uKvDim
	uHd := r.uHd
	uStride := d.NewBufferU32(uint32(qkvDim))
	uKOff := d.NewBufferU32(uint32(nHhd))
	uVOff := d.NewBufferU32(uint32(nHhd + kvDim))
	uStartPos := d.NewBufferU32(uint32(startPos))
	uTotalQ := d.NewBufferU32(uint32(nHhd))
	uTotalK := d.NewBufferU32(uint32(kvDim))
	uBase0 := d.NewBufferU32(0)
	uBaseK := d.NewBufferU32(uint32(nHhd))
	m0, m1, m2 := d.NewBufferU32(0), d.NewBufferU32(1), d.NewBufferU32(2)
	dummyBias := d.NewBufferFloats(make([]float32, 1))

	// gemm grid helper: numRblk×(N/8) simdgroups, rounded up to a full tg (256 threads).
	gg := func(N int) (int, int) {
		numRblk := (Mpad/8 + 3) / 4 // RPS=4
		total := numRblk * (N / 8) * 32
		total = (total + 255) / 256 * 256
		return total, 256
	}

	e := r.q.begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		// pre-attn norm
		e.dispatch(pf.pRms, M*256, 256, xF, L.preNorm, normF, uH, r.uEps)
		// fused QKV (+bias)
		t, tg := gg(qkvDim)
		e.dispatch(pf.pGemmStore, t, tg, normF, L.qkvW, L.qkvS, qkvF, uM, uQkv, uH, L.qkvBias, m1)
		// rope q, k (per-row positions)
		e.dispatch(pf.pRope, M*(nHhd/2), 128, qkvF, r.invf, uHd, posB, uTotalQ, uStride, uBase0)
		e.dispatch(pf.pRope, M*(kvDim/2), 128, qkvF, r.invf, uHd, posB, uTotalK, uStride, uBaseK)
		// scatter K,V to cache
		e.dispatch(pf.pKv, M*kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, uKvDim, uStride, uKOff, uVOff)
		// causal attention → ctx
		e.dispatch(pf.pAttn, M*r.nH*128, 128, qkvF, r.kc[l], r.vc[l], ctxF, r.uNH, r.uNKV, uHd, uStartPos, r.uScale, uStride)
		// o-proj + residual into x
		t, tg = gg(H)
		e.dispatch(pf.pGemmStore, t, tg, ctxF, L.oW, L.oS, xF, uM, uH, uQDim, dummyBias, m2)
		// post-attn norm
		e.dispatch(pf.pRms, M*256, 256, xF, L.postNorm, normF, uH, r.uEps)
		// gate/up
		t, tg = gg(2 * I)
		e.dispatch(pf.pGemmStore, t, tg, normF, L.guW, L.guS, guF, uM, u2I, uH, dummyBias, m0)
		// swiglu
		e.dispatch(pf.pSw, M*I, 256, guF, dqF, uI)
		// down + residual into x
		t, tg = gg(H)
		e.dispatch(pf.pGemmStore, t, tg, dqF, L.dW, L.dS, xF, uM, uH, uI, dummyBias, m2)
	}
	// final norm
	e.dispatch(pf.pRms, M*256, 256, xF, r.finalNorm, normF, uH, r.uEps)
	// lm head on the last-token tile only (8 rows containing M-1) → [8×V]; take the right row.
	tileStart := (M - 1) / 8 * 8
	lmC := d.NewBufferLen(8 * V)
	u8 := d.NewBufferU32(8)
	tLm := (V / 8) * 32
	tLm = (tLm + 255) / 256 * 256
	e.dispatch(pf.pGemm, tLm, 256, normF.At(tileStart*H*2), r.lmW, r.lmS, lmC, u8, uV, uH)
	e.end()

	localRow := (M - 1) - tileStart
	out := make([]float32, V)
	copy(out, lmC.Floats()[localRow*V:localRow*V+V])
	return out
}
