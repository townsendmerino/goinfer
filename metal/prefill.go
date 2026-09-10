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

// rmsnorm_quant_f16: RMSNorm a single f16 row and fused-quantize it to int8 + f32 scale — the
// f16-input twin of the decode path's rmsnorm_quant. Prefill's LM head is the SAME int8-pinned,
// logit-critical head the decode path uses (weights int8, not int4), so the last token's
// final-normed activation must arrive as int8 for gemv_w8a8 — feeding f16/int4 assumptions into
// it is what produced NaN. addOne selects Gemma's (1+w). x already points at the target row.
kernel void rmsnorm_quant_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], constant uint& addOne[[buffer(6)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs){ float v=float(x[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; mx=max(mx,fabs(float(x[i])*rms*g)); }
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; aq[i]=char(clamp(int(round(float(x[i])*rms*g*inv)),-127,127)); }
}

// rmsnorm_f16: one threadgroup per row. out[m] = x[m]*rsqrt(mean(x[m]²)+eps)*w (w f32).
// addOne selects Gemma's (1+w) RMS offset vs plain w — mirrors decoder/rmsnorm.go / the decode
// path's rmsnorm_quant (kernels.go). G8 (docs/task-gpu-paths-2026-09.md): added so this kernel
// can serve BOTH a GEMV-input norm (Llama/Qwen, addOne=0) and Gemma's sandwich norm on a sublayer
// OUTPUT (addOne=1) — same math either way, only the weight convention differs.
kernel void rmsnorm_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    constant uint& addOne[[buffer(5)]],
    uint row[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    device const half* xr = x + row*H; device half* orow = out + row*H;
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float v=float(xr[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; orow[i]=half(float(xr[i])*rms*g); }
}

// residual_f16: x += y (element-wise, grid = M*H).
kernel void residual_f16(device half* x[[buffer(0)]], device const half* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])); }

// residual_f16_from_f32: x += y, x is f16, y is f32 (G8, docs/task-gpu-paths-2026-09.md). The
// batched-prefill MoE row loop's expert-combine dispatches (encodeMoEExperts/
// encodeMoESharedExpert, metal/moe.go) are F32-only — reused UNCHANGED for the per-row loop by
// pointing their accumulate target at an isolated F32 scratch buffer instead of prefill's own F16
// residual, then folding that scratch into the real F16 residual row with this kernel, once.
kernel void residual_f16_from_f32(device half* x[[buffer(0)]], device const float* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+y[i]); }

// zero_f32: x[i] = 0 (grid = N). The MoE row loop's F32 scratch (see residual_f16_from_f32 above)
// must start at zero before encodeMoEExperts' k-expert accumulate loop — unlike decode's own use
// of a scratch buffer (always fully OVERWRITTEN before being read), this one is accumulated into
// by potentially several dispatches (k routed experts plus a shared expert) that all assume
// their target already holds whatever the previous accumulation left, same as r.x does for decode.
kernel void zero_f32(device float* x[[buffer(0)]], uint i[[thread_position_in_grid]]) { x[i] = 0.0; }

// glu_act_f16: SiLU (act=1, decoder.ActKind's ActSiLU) or GELU-tanh (act=0, ActGeluTanh — Gemma).
// The tanh argument is CLAMPED to ±15 — same fix as kernels.go's glu_act (a massive-activation
// gate, e.g. Gemma's <bos>, overflows MSL's tanh to NaN otherwise; duplicated here rather than
// shared since this file compiles as its own separate library (prefillKernels), not allKernels.
inline float glu_act_f16(float x, uint act) {
    if (act == 1u) return x/(1.0f+exp(-x));
    float a = 0.7978845608028654f*(x+0.044715f*x*x*x);
    return 0.5f*x*(1.0f+tanh(clamp(a, -15.0f, 15.0f)));
}
// swiglu_f16: out[m][i] = glu_act_f16(gu[m][i]) * gu[m][I+i], gu is [M×2I]. grid = M*I.
kernel void swiglu_f16(device const half* gu[[buffer(0)]], device half* out[[buffer(1)]],
    constant uint& I[[buffer(2)]], constant uint& act[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint m = gid / I, i = gid % I;
    float g=float(gu[m*2u*I + i]), u=float(gu[m*2u*I + I + i]);
    out[gid]=half(glu_act_f16(g, act)*u);
}

// rope_f16: NeoX half-split, per-row position. Rotates a [M × total] region with row stride
// 'stride' starting at byte-free offset 'base0' (element offset into each row). positions[m].
kernel void rope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;   // nHeads*half; half=rotaryDim/2 (<hd/2 = partial rotary)
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th), s=sin(th);
    float x0=float(x[base+dd]), x1=float(x[base+rhalf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+rhalf+dd]=half(x0*s+x1*c);
}

// qk_norm_f16: per-head Q/K RMSNorm (Qwen3) on the f16 fused qkv[M×stride], before RoPE. One
// threadgroup per (row m, head): head<nH is Q (weight qn), else K (kn, head-nH). Norm over hd.
kernel void qk_norm_f16(device half* qkv[[buffer(0)]], device const float* qn[[buffer(1)]],
    device const float* kn[[buffer(2)]], constant uint& nH[[buffer(3)]], constant uint& nKV[[buffer(4)]],
    constant uint& hd[[buffer(5)]], constant uint& nHhd[[buffer(6)]], constant uint& stride[[buffer(7)]],
    constant float& eps[[buffer(8)]], constant uint& addOne[[buffer(9)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint nHeads = nH + nKV;
    uint m = gid / nHeads, head = gid % nHeads;
    threadgroup float red[128];
    bool isQ = head < nH;
    device const float* w = isQ ? qn : kn;
    uint base = m*stride + (isQ ? head*hd : nHhd + (head-nH)*hd);
    device half* x = qkv + base;
    float ss=0; for(uint i=tid;i<hd;i+=tgs){ float v=float(x[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(hd)+eps);
    for(uint i=tid;i<hd;i+=tgs){ float wt = addOne!=0u ? (1.0f+w[i]) : w[i]; x[i]=half(float(x[i])*rms*wt); }
}

// attention_prefill: one threadgroup per (row m, query head qh). Row m attends CAUSALLY to
// keys[0..startPos+m] over the shared f16 KV cache (GQA via kvh). q read from the fused qkv
// buffer (row stride qStride, q section at offset 0). Reuses the decode attention math (dot →
// threadgroup softmax → value accum), per row with per-row nKeys. O(M²) but that's inherent to
// prefill attention; an MMA/flash version is a later throughput lever.
kernel void attention_prefill(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint nKeys = startPos + m + 1u;
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+s*kvDim; uint d=0;
        // half4 vectorized K-read (coalescing fix, bit-identical; guarded on hd%4==0, scalar tail).
        if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=float(qr[d])*float(k4.x); a+=float(qr[d+1u])*float(k4.y); a+=float(qr[d+2u])*float(k4.z); a+=float(qr[d+3u])*float(k4.w); }
        for (; d<hd; d++) a += float(qr[d])*float(k[d]);
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mmax=-INFINITY; for (uint s=winStart+tid;s<nKeys;s+=tgs) mmax=max(mmax,sc[s]);
    mmax = simd_max(mmax);
    uint sgidA = tid >> 5u, laneA = tid & 31u;
    if (laneA == 0) red[sgidA] = mmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgA = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsgA; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=winStart+tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[s*kvDim+d]); out[m*qDim + qh*hd + d]=half(a/sum); }
}

// attention_prefill_fused: the simdgroup_matrix (MMA) flash-attention twin of attention_prefill
// (L2-Metal, docs/task-prefill-gap.md §4). One simdgroup per (query head, 8-row query tile);
// QKᵀ and PV are both 8×8-tiled MMA matmuls (hd/8 tiles each), replacing attention_prefill's
// per-key scalar dot products and its fully-serial PV scan. K is read K^T-transposed straight
// from the row-major cache via simdgroup_load's transpose flag (confirmed against
// aikit/gpu/metal_vit.go:539's gemm_f32_sg_big, the only prior art for that flag in this
// codebase) — no explicit transpose step. Online (flash) softmax: running max/sum live in
// per-row threadgroup scratch (stat[sgid]) so no O(nKeys) score buffer is needed (the exact
// kernel above allocates sc[4096] and would silently overrun a longer context); the O
// accumulator is rescaled by alpha=exp(mOld-mNew) each key-tile in scalar code, because a
// simdgroup_matrix has no addressable per-row view — every rescale/mask/reduce step goes
// through the float scratch (sScr) or half scratch (pScr) via simdgroup_store/_load, exactly
// as gemm_w4f16_store's epilogue already does for its own per-element bias/residual step.
// NOT bit-identical (the online-softmax rescale reorders the sum vs. the exact kernel's single
// final normalize) — same P19 category as the CUDA L2 twin. Requires hd%8==0 && hd<=128
// (ATTN_MAXHD); the Go dispatch falls back to attention_prefill outside that range.
// ATTN_SGPT simdgroups/threadgroup must match attnFusedSGPT (backend.go's Go-side grid helper).
#define ATTN_SGPT 4
#define ATTN_MAXHD 128
kernel void attention_prefill_fused(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint numRowTiles = (M + 7u) / 8u;
    uint sg = tgid*sgpt + sgid;
    uint totalTiles = nH * numRowTiles;
    if (sg >= totalTiles) return;
    uint qh = sg / numRowTiles, rt = sg % numRowTiles;
    uint r0 = rt * 8u;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint qDim = nH*hd;
    uint hdTiles = hd/8u;
    device const half* qBase = qkv + r0*qStride + qh*hd;
    device const half* kBase = kc + kvh*hd;
    device const half* vBase = vc + kvh*hd;

    threadgroup float sScr[ATTN_SGPT][64];
    threadgroup half  pScr[ATTN_SGPT][64];
    threadgroup float oScr[ATTN_SGPT][8*ATTN_MAXHD];
    threadgroup float stat[ATTN_SGPT][24]; // [0:8)=m  [8:16)=l  [16:24)=alpha (this block)

    for (uint i=lane; i<8u*hd; i+=32u) oScr[sgid][i] = 0.0f;
    if (lane < 8u) { stat[sgid][lane] = -INFINITY; stat[sgid][8u+lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 qTile[ATTN_MAXHD/8];
    for (uint kk=0; kk<hdTiles; kk++) simdgroup_load(qTile[kk], qBase + kk*8u, qStride);

    uint lastRow = min(r0+7u, M-1u);
    uint nKeysMax = startPos + lastRow + 1u;
    uint firstRowKeys = startPos + r0 + 1u;
    uint winStart0 = (window>0u && firstRowKeys>window) ? firstRowKeys-window : 0u;
    uint j0start = (winStart0/8u)*8u;

    for (uint j0=j0start; j0<nKeysMax; j0+=8u) {
        simdgroup_float8x8 Sacc = make_filled_simdgroup_matrix<float,8,8>(0.0);
        for (uint kk=0; kk<hdTiles; kk++) {
            simdgroup_half8x8 kT;
            simdgroup_load(kT, kBase + j0*kvDim + kk*8u, kvDim, ulong2(0,0), true);
            simdgroup_multiply_accumulate(Sacc, qTile[kk], kT, Sacc);
        }
        simdgroup_store(Sacc, sScr[sgid], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8u) {
            uint row = lane, qi = r0+row;
            if (qi < M) {
                uint qpos = startPos+qi, rowKeys = qpos+1u;
                uint rowWin = (window>0u && rowKeys>window) ? rowKeys-window : 0u;
                float sraw[8]; float rowmax = -INFINITY;
                for (uint c=0;c<8u;c++) {
                    uint j = j0+c;
                    float v = sScr[sgid][row*8u+c]*scale;
                    if (j>=rowKeys || j<rowWin) v = -INFINITY;
                    sraw[c]=v; rowmax = max(rowmax, v);
                }
                float mOld = stat[sgid][row];
                float mNew = max(mOld, rowmax);
                float alpha = (mOld <= -INFINITY) ? 0.0f : exp(mOld-mNew);
                float blockSum=0.0f;
                for (uint c=0;c<8u;c++) {
                    float p = (sraw[c] <= -INFINITY) ? 0.0f : exp(sraw[c]-mNew);
                    pScr[sgid][row*8u+c] = half(p);
                    blockSum += p;
                }
                stat[sgid][row] = mNew;
                stat[sgid][8u+row] = stat[sgid][8u+row]*alpha + blockSum;
                stat[sgid][16u+row] = alpha;
            } else {
                for (uint c=0;c<8u;c++) pScr[sgid][row*8u+c] = half(0.0);
                stat[sgid][16u+row] = 1.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint cc=0; cc<hdTiles; cc++) {
            simdgroup_half8x8 pTile, vTile;
            simdgroup_load(pTile, pScr[sgid], 8);
            simdgroup_load(vTile, vBase + j0*kvDim + cc*8u, kvDim);
            simdgroup_float8x8 pvAcc = make_filled_simdgroup_matrix<float,8,8>(0.0);
            simdgroup_multiply_accumulate(pvAcc, pTile, vTile, pvAcc);
            simdgroup_store(pvAcc, sScr[sgid], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 8u) {
                uint row = lane; float a = stat[sgid][16u+row];
                for (uint c=0;c<8u;c++) {
                    uint idx = row*hd + cc*8u+c;
                    oScr[sgid][idx] = oScr[sgid][idx]*a + sScr[sgid][row*8u+c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint e=lane; e<8u*hd; e+=32u) {
        uint row = e/hd, col = e%hd, qi = r0+row;
        if (qi >= M) continue;
        float l = stat[sgid][8u+row];
        out[qi*qDim + qh*hd + col] = half(l > 0.0f ? oScr[sgid][row*hd+col]/l : 0.0f);
    }
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
	// pGemm (gemm_w4f16, no store epilogue) was created but never dispatched — the prefill LM head
	// moved to pRmsQ + pGemvW8, and every GEMM here uses pGemmStore. Removed (audit R-22 / N-09 class).
	pGemmStore, pRms, pRes, pSw, pRope, pKv, pAttn, pQK, pRmsQ Pipeline
	// G8 (docs/task-gpu-paths-2026-09.md): the MoE row loop's F32-scratch bridge (see
	// residual_f16_from_f32/zero_f32's own comments).
	pResF32, pZeroF32 Pipeline
	// L2-Metal (docs/task-prefill-gap.md §4): the simdgroup_matrix flash-attention twin of
	// pAttn. Default ON since §3 gate passed 2026-09-10 (metalFusedAttentionEnabled, backend.go);
	// GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill falls back to pAttn.
	pAttnFused Pipeline
}

func (r *resident) ensurePrefill() {
	if r.pf != nil {
		return
	}
	// M24(c): compile + pipeline creation here runs pool-less on an unpinned thread (PrefillLast
	// calls this BEFORE its own LockOSThread). Pin + hold a pool so the autoreleased temporaries
	// drain; the +1-owned library/pipelines are tracked on the Device and freed at Close.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
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
		pGemmStore: p("gemm_w4f16_store"), pRms: p("rmsnorm_f16"),
		pRes: p("residual_f16"), pSw: p("swiglu_f16"), pRope: p("rope_f16"),
		pKv: p("kv_store_f16"), pAttn: p("attention_prefill"), pQK: p("qk_norm_f16"),
		pRmsQ:   p("rmsnorm_quant_f16"),
		pResF32: p("residual_f16_from_f32"), pZeroF32: p("zero_f32"),
		pAttnFused: p("attention_prefill_fused"),
	}
}

// PrefillLast ingests M prompt embeddings at positions startPos..startPos+M-1 in ONE command
// buffer via the f16 MMA path (weights read once, amortized across M — unlike the token-by-token
// decode loop), populating the resident KV cache, and returns the LAST token's logits[V] (what a
// generator needs to sample the first output token). Correctness-gated vs the sequential path.
func (r *resident) PrefillLast(embs [][]float32, startPos int) []float32 {
	r.ensurePrefill()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, pf := r.d, r.pf
	M := len(embs)
	Mpad := (M + 7) / 8 * 8
	H, I, V := r.H, r.I, r.V
	// Prefill runs only for uniform families (prefillOK declines Gemma's per-layer geometry), so
	// every layer shares one geometry — read it from layer 0.
	g0 := r.layers[0].geom
	nHhd := r.nH * g0.hd
	kvDim := g0.kvDim
	qkvDim := nHhd + 2*kvDim
	qDim := nHhd

	// f16 activation scratch (per call, sized to the padded prompt).
	xh := make([]uint16, Mpad*H)
	for m := range M {
		for i := range H {
			xh[m*H+i] = f32ToF16(embs[m][i])
		}
	}
	xF := NewBufferU16s(d, xh)
	normF := NewBufferU16s(d, make([]uint16, Mpad*H))
	qkvF := NewBufferU16s(d, make([]uint16, Mpad*qkvDim))
	ctxF := NewBufferU16s(d, make([]uint16, Mpad*qDim))
	guF := NewBufferU16s(d, make([]uint16, Mpad*2*I))
	dqF := NewBufferU16s(d, make([]uint16, Mpad*I))
	posv := make([]uint32, Mpad)
	for m := range M {
		posv[m] = uint32(startPos + m)
	}
	posB := NewBufferUint32s(d, posv)

	// uniforms
	uM := NewBufferU32(d, uint32(Mpad))
	uH := r.uH
	uI := NewBufferU32(d, uint32(I))
	u2I := NewBufferU32(d, uint32(2*I))
	uQkv := NewBufferU32(d, uint32(qkvDim))
	uQDim := NewBufferU32(d, uint32(qDim))
	uKvDim := g0.uKvDim
	uHd := g0.uHd
	uStride := NewBufferU32(d, uint32(qkvDim))
	uKOff := NewBufferU32(d, uint32(nHhd))
	uVOff := NewBufferU32(d, uint32(nHhd+kvDim))
	uStartPos := NewBufferU32(d, uint32(startPos))
	uTotalQ := NewBufferU32(d, uint32(nHhd))
	uTotalK := NewBufferU32(d, uint32(kvDim))
	uBase0 := NewBufferU32(d, 0)
	uBaseK := NewBufferU32(d, uint32(nHhd))
	m0, m1, m2 := NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 2)
	dummyBias := NewBufferFloats(d, make([]float32, 1))
	// L2-Metal: attention_prefill_fused's own row-count uniform — REAL M (unpadded), unlike uM
	// above which holds Mpad for the GEMM grid. attention_prefill (the exact kernel) needs no
	// such uniform because its Go-side dispatch grid is already sized off the real M.
	uMReal := NewBufferU32(d, uint32(M))
	// G8: the MoE row loop's F32 accumulate scratch (see the per-layer loop's own comment).
	// Allocated unconditionally (cheap — H floats) rather than gated on whether any layer is
	// MoE, avoiding a nil-buffer special case in the loop below.
	moeDst := NewBufferFloats(d, make([]float32, H))

	// C5: every buffer above is per-call scratch/uniform allocated onto the device ledger, which
	// ReleaseAll frees only at Close — so before this fix each PrefillLast leaked ~24 buffers
	// (~100–150 MB for a 7B; guF alone is Mpad*2I*2), ratcheting until the mustBuf OOM panic killed
	// serve (that panic is recovered only on the BuildResident path, not here). e.End() below
	// commits AND waits, so the GPU is finished with them by the time this returns — release each
	// at end of call. (r.uH / r.uKvDim / r.uHd are resident-owned and reused — deliberately NOT in
	// this list; releasing them would corrupt the decode path.)
	scratch := []Buffer{
		xF, normF, qkvF, ctxF, guF, dqF, posB, moeDst,
		uM, uI, u2I, uQkv, uQDim, uStride, uKOff, uVOff, uStartPos,
		uTotalQ, uTotalK, uBase0, uBaseK, m0, m1, m2, dummyBias, uMReal,
	}
	defer func() {
		for _, b := range scratch {
			d.ReleaseBuf(b)
		}
	}()

	// gemm grid helper: numRblk×(N/8) simdgroups, rounded up to a full tg (256 threads).
	gg := func(N int) (int, int) {
		numRblk := (Mpad/8 + 3) / 4 // RPS=4
		total := numRblk * (N / 8) * 32
		total = (total + 255) / 256 * 256
		return total, 256
	}
	// L2-Metal: attention_prefill_fused's grid — nH×ceil(M/8) simdgroups (ATTN_SGPT=4/threadgroup,
	// prefill.go's own #define, matched here). Real M (unpadded): the tail row-tile's out-of-range
	// rows are masked in-kernel via buffer(11), not dropped from the grid.
	const attnFusedSGPT = 4
	numRowTiles := (M + 7) / 8
	attnFusedTotal := r.nH * numRowTiles
	attnFusedTotal = (attnFusedTotal + attnFusedSGPT - 1) / attnFusedSGPT * attnFusedSGPT * 32
	attnFusedTg := attnFusedSGPT * 32
	// hd%8==0 && hd<=128 (ATTN_MAXHD) — attention_prefill_fused's compile-time cap.
	useFusedAttn := metalFusedAttentionEnabled() && g0.hd%8 == 0 && g0.hd <= 128

	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		// pre-attn norm — addOne (Gemma's 1+w) matters even for a plain non-sandwich family's
		// GEMV-input norm, so it is always passed (0 for every family without RMSAddOne).
		e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, r.uEps, r.uAddOne)
		// fused QKV (+bias)
		t, tg := gg(qkvDim)
		e.Dispatch(pf.pGemmStore, t, tg, normF, L.qkvW, L.qkvS, qkvF, uM, uQkv, uH, L.qkvBias, m1)
		if r.qkNorm { // Qwen3: per-head Q/K RMSNorm before RoPE
			e.Dispatch(pf.pQK, M*(r.nH+g0.nKV)*tgReduceAttn, tgReduceAttn, qkvF, L.qNorm, L.kNorm, r.uNH, g0.uNKV, uHd, g0.uNHhd, uStride, r.uEps, r.uAddOne)
		}
		// rope q, k (per-row positions) — bind the PER-LAYER RoPE table and window, exactly as decode
		// does (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window
		// arch the global layers must see window=0, and each layer its own RoPE base; the model-level
		// bindings applied the local window (and one RoPE table) to every layer (audit M-09). Admitted
		// prefill archs have a uniform RoPE table (FeatPerLayerRoPE is not claimed), so L.invf equals
		// r.invf there — this is behaviour-neutral for them and correct for the mixed-window case.
		e.Dispatch(pf.pRope, M*r.nH*g0.half, 128, qkvF, L.invf, uHd, posB, uTotalQ, uStride, uBase0, g0.uHalf)
		e.Dispatch(pf.pRope, M*g0.nKV*g0.half, 128, qkvF, L.invf, uHd, posB, uTotalK, uStride, uBaseK, g0.uHalf)
		// scatter K,V to cache
		e.Dispatch(pf.pKv, M*kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, uKvDim, uStride, uKOff, uVOff)
		// causal attention → ctx (per-layer window: 0 = full causal on a global layer)
		if useFusedAttn {
			e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg, qkvF, r.kc[l], r.vc[l], ctxF, r.uNH, g0.uNKV, uHd, uStartPos, r.uScale, uStride, L.uWindow, uMReal)
		} else {
			e.Dispatch(pf.pAttn, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, r.kc[l], r.vc[l], ctxF, r.uNH, g0.uNKV, uHd, uStartPos, r.uScale, uStride, L.uWindow)
		}
		// o-proj, then either the plain residual epilogue or Gemma's sandwich norm (G8):
		// mode-0 write into normF (free scratch at this point — its last use, the fused-QKV
		// input, already ran; its next use, the pre-MLP norm's output, is below), norm the
		// sublayer OUTPUT in place (safe — rmsnorm_f16's read pass fully completes before its
		// write pass touches the same buffer), then a separate residual add. Non-sandwich
		// families skip straight to the fused mode-2 residual epilogue, byte-identical to before
		// this row.
		t, tg = gg(H)
		if r.sandwich {
			e.Dispatch(pf.pGemmStore, t, tg, ctxF, L.oW, L.oS, normF, uM, uH, uQDim, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postAttnNorm, normF, uH, r.uEps, r.uAddOne)
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		} else {
			e.Dispatch(pf.pGemmStore, t, tg, ctxF, L.oW, L.oS, xF, uM, uH, uQDim, dummyBias, m2)
		}
		if L.moe != nil {
			// G8 (docs/task-gpu-paths-2026-09.md): MoE FFN — batch the attention half above as
			// usual, but run the FFN ROW BY ROW off the batched residual (xF), reusing the EXACT
			// per-token decode MoE dispatch chain (encodeMoERoute/encodeMoEExperts/
			// encodeMoESharedExpert, metal/moe.go) unchanged — the same approach CUDA's own
			// batched prefill already uses for MoE (cuda/prefill.go: "row by row off the batched
			// residual"). No new routing math, no batched-GEMM-over-experts kernel.
			//
			// Those dispatches are F32-only (decode's own r.x is F32); prefill's residual (xF) is
			// F16. Bridged per row: rmsnorm_quant_f16 norms+quantizes the row DIRECTLY (no F32
			// conversion needed — it already takes an F16 input, unlike r.pRms) into r.mq/r.mSc,
			// the expert loop accumulates into moeDst (an isolated F32 scratch, zeroed first —
			// unlike decode's r.x, which starts each token already holding the residual to
			// accumulate onto), and residual_f16_from_f32 folds that scratch into xF's row once.
			for m := 0; m < M; m++ {
				row := xF.At(m * H * 2)
				e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, row, L.postNorm, r.mq, r.mSc, uH, r.uEps, r.uAddOne)
				r.encodeMoERoute(e, L)
				e.Dispatch(pf.pZeroF32, r.H, 256, moeDst)
				r.encodeMoEExperts(e, L, moeDst)
				e.Dispatch(pf.pResF32, r.H, 256, row, moeDst)
			}
			continue
		}
		// pre-MLP norm
		e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, uH, r.uEps, r.uAddOne)
		// gate/up
		t, tg = gg(2 * I)
		e.Dispatch(pf.pGemmStore, t, tg, normF, L.guW, L.guS, guF, uM, u2I, uH, dummyBias, m0)
		// swiglu/geglu (G8: r.uAct selects — 0 for every family without FeatGatedGELU)
		e.Dispatch(pf.pSw, M*I, 256, guF, dqF, uI, r.uAct)
		// down-proj, same plain-vs-sandwich split as o-proj above.
		t, tg = gg(H)
		if r.sandwich {
			e.Dispatch(pf.pGemmStore, t, tg, dqF, L.dW, L.dS, normF, uM, uH, uI, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postMLPNorm, normF, uH, r.uEps, r.uAddOne)
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		} else {
			e.Dispatch(pf.pGemmStore, t, tg, dqF, L.dW, L.dS, xF, uM, uH, uI, dummyBias, m2)
		}
	}
	// final norm + LM head for the LAST token only, through the SAME int8-pinned head the decode
	// path runs (rmsnorm→int8, then gemv_w8a8). The head weights are int8 (logit-critical); the
	// int4 gemm_w4f16 used here previously misread them as packed nibbles + f16 scales, producing
	// NaN logits. Norm-quant the last token's f16 residual row to int8, then run the decode head.
	e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, xF.At((M-1)*H*2), r.finalNorm, r.aq, r.aSc, uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	r.recordExecErr(e.Err()) // C-09

	out := make([]float32, V)
	copy(out, r.logits.Floats()[:V])
	// G8: Gemma's final-logit softcap — finalizeLogits (the decode path's own entry point) applies
	// this to r.logitsHost, but PrefillLast copies straight out of r.logits into a fresh slice and
	// returns before finalizeLogits ever runs, so it must be applied here too. 0 for every
	// non-softcapped family (softcapParallel no-ops).
	if r.finalSoftcap > 0 {
		softcapParallel(out, r.finalSoftcap)
	}
	return out
}
