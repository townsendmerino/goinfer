//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"
)

// Prefill kernels — the f16 simdgroup_matrix (MMA) path for fast prompt ingestion. Unlike the
// int4/scalar-MAC decode path (which can't amortize batching), an MMA GEMM reuses each weight
// across all M prompt rows → ~2.5× the per-token GEMV, flat with M. Activations flow in f16
// (no int8 quant); weights stay int4 and are dequanted to f16 in-kernel (no extra RAM). Kept in
// a SEPARATE library from allKernels so the decode path is unchanged and prefill is opt-in.
const prefillKernels = `
#include <metal_stdlib>
using namespace metal;

// gemm_w4f16_store: blocked int4→f16 MMA GEMM with a fused epilogue (mode 0 = plain store, 1 =
// +bias for fused QKV, 2 = +residual for o-proj/down) — C[M×N] = A[M×K](f16) · Wᵀ, W = resident
// int4/W4A8 (packed nibbles + f16 group scales), dequanted in-kernel. N-17 (audit-
// metal-2026-09-12.md): this used to be two separate steps, a plain gemm_w4f16 (no epilogue) plus
// hand-written bias/residual variants — gemm_w4f16 itself was fully replaced by this kernel but
// its source stayed in the compiled library with no pipeline ever created from it. Deleted rather
// than left as dead compiled source.
//
// CPS (audit-metal-2026-09-12.md M-03): each simdgroup owns a 32(M, RPS)×32(N, CPS) output block
// instead of 32×8 — all 32 lanes dequant CPS=4 weight tiles per k-step (was 8 of 32 lanes doing
// one tile, 24 idle), and each of the RPS A-row tiles is loaded ONCE from device memory per
// k-step and reused across all CPS column tiles (16 MMAs per barrier pair instead of 4; 4x fewer
// simdgroups stream the same A rows redundantly). N is only guaranteed %8==0 (audit C-10), not
// %32==0, so the last column super-tile is masked per 8-wide sub-tile exactly like the existing
// M-dimension tail (the break idiom below) — never an OOB read of W/WS, never a bogus MMA/store.
#define RPS 4
#define CPS 4
kernel void gemm_w4f16_store(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    threadgroup half wscr[8*(CPS*64)];
    threadgroup float cscr[8*(CPS*64)];
    uint sg = tgid*sgpt + sgid;
    uint tilesN = (N + 31u)/32u; // ceil — the last super-tile may be ragged (N%32 != 0)
    uint rblk = sg / tilesN, tc = sg % tilesN;
    uint r0 = rblk*RPS;
    if (r0*8u >= M) return;
    uint n0 = tc*32u;
    threadgroup half* scr = wscr + sgid*(CPS*64u);
    threadgroup float* cs = cscr + sgid*(CPS*64u);
    uint wpr = K/8u, gpr = K/32u;
    simdgroup_float8x8 acc[RPS*CPS];
    for (uint i=0;i<RPS*CPS;i++) acc[i]=make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        if (lane < 32u) {
            uint c = lane/8u, nl = lane%8u;
            uint ncol = n0 + c*8u + nl;
            if (ncol < N) {
                uint word = W[ncol*wpr + k/8u];
                float sc = float(WS[ncol*gpr + k/32u]);
                for (uint kl=0; kl<8u; kl++)
                    scr[c*64u + kl*8u + nl] = half(float(int((word >> (4u*kl)) & 0xF) - 8) * sc);
            } else {
                for (uint kl=0; kl<8u; kl++)
                    scr[c*64u + kl*8u + nl] = 0.0h; // ragged tail padding lane — never stored
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 bT[CPS];
        for (uint c=0;c<CPS;c++) simdgroup_load(bT[c], scr + c*64u, 8);
        for (uint r=0;r<RPS;r++) {
            if ((r0+r)*8u >= M) break;
            simdgroup_half8x8 a; simdgroup_load(a, A + ((r0+r)*8u)*K + k, K);
            for (uint c=0;c<CPS;c++) {
                if (n0 + c*8u >= N) break;
                simdgroup_multiply_accumulate(acc[r*CPS+c], a, bT[c], acc[r*CPS+c]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: store via threadgroup scratch so lanes can apply bias/residual per element.
    for (uint r=0;r<RPS;r++) {
        uint mrow = r0*8u + r*8u;
        if (mrow >= M) break;
        for (uint c=0;c<CPS;c++) {
            if (n0+c*8u >= N) break;
            simdgroup_store(acc[r*CPS+c], cs + c*64u, 8);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 32u) {
            uint c = lane/8u, nl = lane%8u;
            if (n0 + c*8u < N) {
                for (uint ml=0; ml<8u; ml++) {
                    uint m = mrow + ml, n = n0 + c*8u + nl;
                    float v = cs[c*64u + ml*8u + nl];
                    if (mode == 1u) v += bias[n];                 // fused bias (QKV)
                    if (mode == 2u) v += float(C[m*N + n]);       // residual (o / down)
                    C[m*N + n] = half(v);
                }
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
// path's rmsnorm_quant (kernels.go). G8 (docs/tasks/task-gpu-paths-2026-09.md): added so this kernel
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

// residual_f16_from_f32: x += y, x is f16, y is f32 (G8, docs/tasks/task-gpu-paths-2026-09.md). The
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
// (L2-Metal, docs/completed/task-prefill-gap.md §4). One simdgroup per (query head, 8-row query tile);
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
//
// ATTN_KTILE (audit-metal-2026-09-12.md M-04): the QKᵀ score computation runs on ATTN_KTILE=32
// keys (4 sub-tiles of 8) before the scalar softmax/rescale/PV phase — which is the barrier-heavy
// part (2 barriers per hd-tile, hdTiles up to 16) — instead of on 8 keys as before, amortising
// that phase 4x (the audit's count: ~34 barriers per 8 keys against 32 MMAs of useful work). O
// itself stays in threadgroup scratch with the same scalar per-lane rescale (oScr, unchanged) —
// this does NOT attempt the audit's further register-accumulator + diagonal-α-MMA rewrite, which
// needs its own f32->f16 round-trip to feed the diagonal as an MMA operand and was not obviously
// fewer barriers once that round-trip is counted; scoped out as a separate, riskier follow-up.
#define ATTN_SGPT 4
#define ATTN_MAXHD 128
#define ATTN_KTILE 32
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

    threadgroup float sScr[ATTN_SGPT][8*ATTN_KTILE];
    threadgroup half  pScr[ATTN_SGPT][8*ATTN_KTILE];
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
    uint j0superStart = (winStart0/ATTN_KTILE)*ATTN_KTILE;

    for (uint j0s=j0superStart; j0s<nKeysMax; j0s+=ATTN_KTILE) {
        // nsub: valid 8-key sub-tiles in this group (the last group may be ragged — nKeysMax
        // need not be a multiple of ATTN_KTILE, or even of 8).
        uint nsub = min((nKeysMax - j0s + 7u)/8u, uint(ATTN_KTILE/8));
        for (uint sub=0; sub<nsub; sub++) {
            uint j0 = j0s + sub*8u;
            simdgroup_float8x8 Sacc = make_filled_simdgroup_matrix<float,8,8>(0.0);
            for (uint kk=0; kk<hdTiles; kk++) {
                simdgroup_half8x8 kT;
                simdgroup_load(kT, kBase + j0*kvDim + kk*8u, kvDim, ulong2(0,0), true);
                simdgroup_multiply_accumulate(Sacc, qTile[kk], kT, Sacc);
            }
            simdgroup_store(Sacc, sScr[sgid] + sub*8u, ATTN_KTILE);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint ncols = nsub*8u;
        if (lane < 8u) {
            uint row = lane, qi = r0+row;
            if (qi < M) {
                uint qpos = startPos+qi, rowKeys = qpos+1u;
                uint rowWin = (window>0u && rowKeys>window) ? rowKeys-window : 0u;
                float sraw[ATTN_KTILE]; float rowmax = -INFINITY;
                for (uint c=0;c<ncols;c++) {
                    uint j = j0s+c;
                    float v = sScr[sgid][row*ATTN_KTILE+c]*scale;
                    if (j>=rowKeys || j<rowWin) v = -INFINITY;
                    sraw[c]=v; rowmax = max(rowmax, v);
                }
                float mOld = stat[sgid][row];
                float mNew = max(mOld, rowmax);
                float alpha = (mOld <= -INFINITY) ? 0.0f : exp(mOld-mNew);
                float blockSum=0.0f;
                for (uint c=0;c<ncols;c++) {
                    float p = (sraw[c] <= -INFINITY) ? 0.0f : exp(sraw[c]-mNew);
                    pScr[sgid][row*ATTN_KTILE+c] = half(p);
                    blockSum += p;
                }
                stat[sgid][row] = mNew;
                stat[sgid][8u+row] = stat[sgid][8u+row]*alpha + blockSum;
                stat[sgid][16u+row] = alpha;
            } else {
                for (uint c=0;c<ncols;c++) pScr[sgid][row*ATTN_KTILE+c] = half(0.0);
                stat[sgid][16u+row] = 1.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint cc=0; cc<hdTiles; cc++) {
            simdgroup_float8x8 pvAcc = make_filled_simdgroup_matrix<float,8,8>(0.0);
            for (uint sub=0; sub<nsub; sub++) {
                simdgroup_half8x8 pTile, vTile;
                simdgroup_load(pTile, pScr[sgid] + sub*8u, ATTN_KTILE);
                simdgroup_load(vTile, vBase + (j0s+sub*8u)*kvDim + cc*8u, kvDim);
                simdgroup_multiply_accumulate(pvAcc, pTile, vTile, pvAcc);
            }
            simdgroup_store(pvAcc, sScr[sgid], 8); // scratch reuse: this group's scores are consumed
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
	// G8 (docs/tasks/task-gpu-paths-2026-09.md): the MoE row loop's F32-scratch bridge (see
	// residual_f16_from_f32/zero_f32's own comments).
	pResF32, pZeroF32 Pipeline
	// L2-Metal (docs/completed/task-prefill-gap.md §4): the simdgroup_matrix flash-attention twin of
	// pAttn. Default ON since §3 gate passed 2026-09-10 (metalFusedAttentionEnabled, backend.go);
	// GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill falls back to pAttn.
	pAttnFused Pipeline
}

func (r *resident) ensurePrefill() {
	if r.pf != nil {
		return
	}
	if r.pfErr != nil {
		// N-47 (audit-2026-09-10.md): a prior attempt already failed — re-panic the SAME cached
		// error instead of re-running the full MSL compile just to fail identically again.
		panic(r.pfErr)
	}
	// M24(c): compile + pipeline creation here runs pool-less on an unpinned thread (PrefillLast
	// calls this BEFORE its own LockOSThread). Pin + hold a pool so the autoreleased temporaries
	// drain; the +1-owned library/pipelines are tracked on the Device and freed at Close.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	defer func() {
		if p := recover(); p != nil {
			err := fmt.Errorf("%v", p)
			r.pfErr = err
			panic(err)
		}
	}()
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

// parallelEmbedsF32ToF16 converts M rows of embs (each H wide) into dst[m*H:(m+1)*H] as f16 bits,
// splitting across up to 8 workers by ROW — P-14 (audit-2026-09-10): the serial scalar loop this
// replaces was 7.3M f32ToF16 calls at M=2048, H=3584 on the TTFT path, and model.go's own
// parallelF32ToF16 (built for exactly this conversion in the gemma4-26b expert-paging path)
// already proved the parallel split is byte-identical to serial — every element is independent
// and f32ToF16 is a pure function of its one input. Not reused directly: embs is [][]float32 (one
// slice per row, not necessarily contiguous), where parallelF32ToF16 wants one flat []float32; a
// flatten-then-call would pay its own copy, so this splits by row directly instead, over M×H
// rather than a flat index range, but is otherwise the same threshold/worker shape.
func parallelEmbedsF32ToF16(dst []uint16, embs [][]float32, H int) {
	M := len(embs)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if M*H < 8192 || workers <= 1 || M < 2 {
		for m := range M {
			row := embs[m]
			for i := range H {
				dst[m*H+i] = f32ToF16(row[i])
			}
		}
		return
	}
	chunk := (M + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < M; lo += chunk {
		hi := min(lo+chunk, M)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for m := lo; m < hi; m++ {
				row := embs[m]
				for i := range H {
					dst[m*H+i] = f32ToF16(row[i])
				}
			}
		}(lo, hi)
	}
	wg.Wait()
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
	parallelEmbedsF32ToF16(xh, embs, H)
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
		numRblk := (Mpad/8 + 3) / 4 // RPS=4 (32 M-rows/simdgroup)
		tilesN := (N + 31) / 32     // CPS=4 (32 N-cols/simdgroup, M-03) — ceil: N is only %8, not %32
		total := numRblk * tilesN * 32
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
			// G8 (docs/tasks/task-gpu-paths-2026-09.md): MoE FFN — batch the attention half above as
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
