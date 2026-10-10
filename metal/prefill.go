//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// prefillKernels is the f16 simdgroup_matrix (MMA) prefill library, for fast prompt ingestion: an MMA GEMM reuses each
// weight across all M prompt rows, which the int4 scalar-MAC decode path cannot. Activations flow in f16 (no int8
// quant); weights stay int4 and are dequantized to f16 in-kernel. A separate library from allKernels, so the decode
// path is unchanged; ensurePrefill compiles it lazily.
const prefillKernels = `
#include <metal_stdlib>
using namespace metal;

// gemm_w4f16_store: blocked int4→f16 MMA GEMM with a fused epilogue (mode 0 = plain store, 1 =
// +bias for fused QKV, 2 = +residual for o-proj/down) — C[M×N] = A[M×K](f16) · Wᵀ, W = resident
// int4/W4A8 (packed nibbles + f16 group scales), dequanted in-kernel. Dispatched as a 2-D grid of
// threadgroups, (ceil(N/64), ceil(M/64)) × 128 threads — see gemmGrid in PrefillLast.
//
// R16 (docs/tasks/red-october.md; docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): restructured
// 2026-09-25 in the shape of llama.cpp's classic kernel_mul_mm. The previous kernel (each simdgroup an
// island that fetched its own activations from device memory inside the MMA loop, K stepped by 8) ran
// the MLP projections at ~0.75 TFLOPS under sustained load; this one runs them at ~2.7–2.9, measured
// 3.22× on the 1.5B's GEMM category at K=512 in a pre-registered confirmation run. Structure:
//   - a threadgroup of 4 simdgroups owns a 64-feature × 64-token output tile; simdgroup sg computes
//     features 32*(sg&1).. × tokens 32*(sg>>1).. — 16 accumulators, 8 matrix loads per 16 MMAs;
//   - K advances 32 per slab (exactly one scale group), and both operand tiles are staged into
//     threadgroup memory ONCE per slab, cooperatively, and shared by all four simdgroups — the MMA
//     loop reads threadgroup memory only;
//   - weight blocks are padded to a 72-half stride (spreads one store's lanes across banks).
// Rows past M (the last token tile, and MoE expert GEMMs, which run on few rows) stage zeros and
// are never stored; their MMAs still run — skipping them measured slower (see the MMA loop). What
// that costs a few-row MoE expert GEMM is NOT measured.
// BIT-IDENTICAL to the kernel it replaced, by construction and by measurement: each output still
// accumulates K in ordered 8-wide chunks into an f32 simdgroup_float8x8 via
// simdgroup_multiply_accumulate(acc, a, b, acc), from the same f16 activations and weights
// dequantized by the same expression, with the same epilogue. The retired kernel is kept in
// prefill_gemm_s2_test.go for A/B re-runs.
//
// Alignment it relies on (all held by construction here): K % 32 == 0 (one f16 scale per 32
// weights), so each W row is a multiple of 16 bytes and the two words a thread reads form an
// aligned uint2; activation rows are 64-byte multiples, so the half4 loads are aligned. N % 8 == 0
// (audit C-10); features past N stage zeros and are never stored.
inline void gemm_stage_w(threadgroup half* sa, uint2 wv, float sc, bool wok, ushort fb, ushort kh, ushort nl) {
    for (ushort w = 0; w < 2; w++) {
        uint word = wv[w];
        threadgroup half* p = sa + (fb*4 + kh*2 + w)*72 + nl;
        for (ushort kl = 0; kl < 8; kl++) {
            *p = wok ? half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc) : 0.0h;
            p += 8;
        }
    }
}

inline void gemm_epilogue(threadgroup float* myc, simdgroup_float8x8 acc, uint mb, uint nb, uint M, uint N,
    device half* C, device const float* bias, uint mode, ushort lane) {
    simdgroup_store(acc, myc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort e = lane; e < 64; e += 32) {
        const uint m = mb + e/8, n = nb + e%8;
        if (m < M && n < N) {
            float v = myc[e];
            if (mode == 1u) v += bias[n];                 // fused bias (QKV)
            if (mode == 2u) v += float(C[m*N + n]);       // residual (o / down)
            C[m*N + n] = half(v);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void gemm_w4f16_store(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    threadgroup half sa[32*72];   // weights:     [feature block 0..7][k block 0..3] of 8x8 [k][feature], padded
    threadgroup half sb[64*32];   // activations: [token block 0..7][k block 0..3] of 8x8 [token][k]
    threadgroup float cs[4*64];   // per-simdgroup epilogue scratch

    const uint n0 = tgp.x*64u, m0 = tgp.y*64u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;
    simdgroup_float8x8 acc[16];
    for (ushort i = 0; i < 16; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid >> 1, kh = tid & 1;     // weight row 0..63, k half (two 8-wide k blocks)
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const bool wok = ncol < N;
    const ushort tr = tid >> 1, kh2 = tid & 1;    // token row 0..63, k half
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    const bool aok = mrow < M;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint2 wv = uint2(0u); float sc = 0.0f;
        if (wok) { wv = *(device const uint2*)(W + ncol*wpr + k0/8u + kh*2u); sc = float(WS[ncol*gpr + k0/32u]); }
        gemm_stage_w(sa, wv, sc, wok, fb, kh, nl);
        for (ushort j = 0; j < 2; j++) {
            const ushort kb = kh2*2 + j;
            threadgroup half4* d4 = (threadgroup half4*)(sb + (tb*4 + kb)*64 + rl*8);
            if (aok) {
                device const half4* s4 = (device const half4*)(A + mrow*K + k0 + kb*8u);
                d4[0] = s4[0]; d4[1] = s4[1];
            } else {
                d4[0] = half4(0.0h); d4[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[4], b[4];
            // Constant trip counts and NO per-block predicate — prototype 4's loop exactly. Two attempts
            // to skip the MMAs of token blocks wholly past M (for few-row MoE expert GEMMs) were measured
            // on the 1.5B during wiring (R16, 2026-09-25), both bit-identical and both slower: a runtime
            // loop bound 't < tlive' (no full unroll, acc[] dynamically indexed and spilled) was 5.6x
            // slower than this loop; the same skip as a uniform predicate inside constant loops was
            // still ~37% slower (GEMM category 650.6 vs 476.6 ms).
            for (ushort t = 0; t < 4; t++) simdgroup_load(a[t], sb + ((th*4 + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < 4; f++) simdgroup_load(b[f], sa + ((fh*4 + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < 4; t++)
                for (ushort f = 0; f < 4; f++)
                    simdgroup_multiply_accumulate(acc[t*4 + f], a[t], b[f], acc[t*4 + f]);
        }
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < 4; t++)
        for (ushort f = 0; f < 4; f++)
            gemm_epilogue(myc, acc[t*4 + f], m0 + (th*4 + t)*8u, n0 + (fh*4 + f)*8u, M, N, C, bias, mode, lane);
}

// A-P01 (docs/audit-metal-2026-09-30.md): gemm_w4f16_store's tile as template parameters, TM tokens × TN features per
// threadgroup, for the small-M passes where a 64 × 64 tile leaves cores idle (o, down and qkv run 24-32 threadgroups on
// the 1.5B) and, at M <= 32, runs MMAs for rows that are not there. Same 128 threads, same 2 × 2 simdgroups, each owning
// (TM/2) × (TN/2) outputs; the same 32-k slabs staged once into threadgroup memory; the same dequant, MMA chain and
// epilogue. Each output still accumulates K in ordered 8-wide chunks through simdgroup_multiply_accumulate from the
// same operand values, so only which threadgroup owns it changes: bit-identical to gemm_w4f16_store, which stays as
// the 64 × 64 kernel (gemm_w4f16_m64n64 is this template at that tile, kept for the test that checks the template
// against it). Smaller tiles stage fewer rows per slab: TN = 32 gives each thread one 8-k word of one weight row instead
// of two, TM = 32 one 8-k block of one activation row.
template <ushort TM, ushort TN, bool W8 = false>
kernel void gemm_w4f16_tile(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    constexpr ushort TT = TM/16, TF = TN/16;     // 8-blocks of tokens / features per simdgroup
    constexpr ushort WT = 128/TN, WPT = 4/WT;    // threads per weight row, 8-k words each
    constexpr ushort AT = TM >= 32 ? 128/TM : 4, BPT = 4/AT; // threads per activation row, 8-k blocks each (TM = 16, D-B02: 4 a row, threads 64-127 stage none)
    threadgroup half sa[(TN/8)*4*72];            // weights:     [feature block][k block] of 8x8 [k][feature], padded
    threadgroup half sb[(TM/8)*4*64];            // activations: [token block][k block] of 8x8 [token][k]
    threadgroup float cs[4*64];                  // per-simdgroup epilogue scratch

    const uint n0 = tgp.x*uint(TN), m0 = tgp.y*uint(TM);
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;
    simdgroup_float8x8 acc[TT*TF];
    for (ushort i = 0; i < TT*TF; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid / WT, kq = tid % WT;   // weight row, its k quarter/half
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const bool wok = ncol < N;
    const ushort tr = tid / AT, kq2 = tid % AT;  // token row, its k quarter/half
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    const bool aok = mrow < M;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (W8) {
            // int8 slice 2 (docs/tasks/task-metal-int8-2026-10.md): W is the native int8 path's row-major int8 codes,
            // K bytes a row, and WS its per-row f32 scales; an 8-k block is two words.
            float sc = wok ? reinterpret_cast<device const float*>(WS)[ncol] : 0.0f;
            for (ushort w = 0; w < WPT; w++) {
                const ushort kb = kq*WPT + w;
                const uint2 wd = wok ? *reinterpret_cast<device const uint2*>(W + (ncol*K + k0 + kb*8u)/4u) : uint2(0u);
                const char4 c0 = as_type<char4>(wd.x), c1 = as_type<char4>(wd.y);
                threadgroup half* p = sa + (fb*4 + kb)*72 + nl;
                for (ushort kl = 0; kl < 8; kl++) {
                    *p = wok ? half(float(kl < 4 ? c0[kl] : c1[kl - 4]) * sc) : 0.0h;
                    p += 8;
                }
            }
        } else {
            float sc = wok ? float(WS[ncol*gpr + k0/32u]) : 0.0f;
            for (ushort w = 0; w < WPT; w++) {
                const ushort kb = kq*WPT + w;
                const uint word = wok ? W[ncol*wpr + k0/8u + kb] : 0u;
                threadgroup half* p = sa + (fb*4 + kb)*72 + nl;
                for (ushort kl = 0; kl < 8; kl++) {
                    *p = wok ? half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc) : 0.0h;
                    p += 8;
                }
            }
        }
        for (ushort j = 0; j < BPT && (TM >= 32 || tr < TM); j++) {
            const ushort kb = kq2*BPT + j;
            threadgroup half4* d4 = (threadgroup half4*)(sb + (tb*4 + kb)*64 + rl*8);
            if (aok) {
                device const half4* s4 = (device const half4*)(A + mrow*K + k0 + kb*8u);
                d4[0] = s4[0]; d4[1] = s4[1];
            } else {
                d4[0] = half4(0.0h); d4[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[TT], b[TF];
            for (ushort t = 0; t < TT; t++) simdgroup_load(a[t], sb + ((th*TT + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < TF; f++) simdgroup_load(b[f], sa + ((fh*TF + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < TT; t++)
                for (ushort f = 0; f < TF; f++)
                    simdgroup_multiply_accumulate(acc[t*TF + f], a[t], b[f], acc[t*TF + f]);
        }
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < TT; t++)
        for (ushort f = 0; f < TF; f++)
            gemm_epilogue(myc, acc[t*TF + f], m0 + (th*TT + t)*8u, n0 + (fh*TF + f)*8u, M, N, C, bias, mode, lane);
}
template [[host_name("gemm_w4f16_m64n64")]] kernel decltype(gemm_w4f16_tile<64, 64>) gemm_w4f16_tile<64, 64>;
template [[host_name("gemm_w4f16_m32n64")]] kernel decltype(gemm_w4f16_tile<32, 64>) gemm_w4f16_tile<32, 64>;
template [[host_name("gemm_w4f16_m64n32")]] kernel decltype(gemm_w4f16_tile<64, 32>) gemm_w4f16_tile<64, 32>;
template [[host_name("gemm_w4f16_m32n32")]] kernel decltype(gemm_w4f16_tile<32, 32>) gemm_w4f16_tile<32, 32>;
template [[host_name("gemm_w4f16_m16n64")]] kernel decltype(gemm_w4f16_tile<16, 64>) gemm_w4f16_tile<16, 64>;
template [[host_name("gemm_w4f16_m16n32")]] kernel decltype(gemm_w4f16_tile<16, 32>) gemm_w4f16_tile<16, 32>;
// int8 slice 2: the same tiles over the native int8 path's weights (W8 = true).
template [[host_name("gemm_w8f16_m64n64")]] kernel decltype(gemm_w4f16_tile<64, 64, true>) gemm_w4f16_tile<64, 64, true>;
template [[host_name("gemm_w8f16_m32n64")]] kernel decltype(gemm_w4f16_tile<32, 64, true>) gemm_w4f16_tile<32, 64, true>;
template [[host_name("gemm_w8f16_m64n32")]] kernel decltype(gemm_w4f16_tile<64, 32, true>) gemm_w4f16_tile<64, 32, true>;
template [[host_name("gemm_w8f16_m32n32")]] kernel decltype(gemm_w4f16_tile<32, 32, true>) gemm_w4f16_tile<32, 32, true>;
template [[host_name("gemm_w8f16_m16n32")]] kernel decltype(gemm_w4f16_tile<16, 32, true>) gemm_w4f16_tile<16, 32, true>;

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

// layernorm_f16 / layernorm_quant_f16 (docs/tasks/task-metal-pairwise-followups-2026-10.md, Part B): rmsnorm_f16 and
// rmsnorm_quant_f16 for a bias-free LayerNorm (Cohere / Command-R7B): out = (x - mean) * rsqrt(var + eps) * w, the math of
// the decode path's layernorm_quant with hasBias 0. Same signatures as the RMSNorm twins (addOne is accepted and unused:
// no LayerNorm family carries Gemma's 1+w), so a dispatch site only swaps the pipeline. A biased LayerNorm (GPT-2) does not
// take the pass (batchedPrefill declines it).
kernel void layernorm_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    constant uint& addOne[[buffer(5)]],
    uint row[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    device const half* xr = x + row*H; device half* orow = out + row*H;
    float s=0; for(uint i=tid;i<H;i+=tgs) s+=float(xr[i]);
    red[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean=red[0]/float(H); threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float d=float(xr[i])-mean; ss+=d*d; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs) orow[i]=half((float(xr[i])-mean)*inv*w[i]);
}
kernel void layernorm_quant_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], constant uint& addOne[[buffer(6)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float s=0; for(uint i=tid;i<H;i+=tgs) s+=float(x[i]);
    red[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean=red[0]/float(H); threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float d=float(x[i])-mean; ss+=d*d; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs((float(x[i])-mean)*inv*w[i]));
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float qi=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round((float(x[i])-mean)*inv*w[i]*qi)),-127,127));
}

// residual_f16: x += y (element-wise, grid = M*H).
kernel void residual_f16(device half* x[[buffer(0)]], device const half* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])); }
// residual_f16_scaled: x += y*inv, inv = 1/s a power of two: the add into a residual stored scaled by 1/s (S17,
// resident.prefillResidScale).
kernel void residual_f16_scaled(device half* x[[buffer(0)]], device const half* y[[buffer(1)]], constant float& inv[[buffer(2)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])*inv); }

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

// ple_gelu_mul_f16 (S9 step 2): the batched PLE gate, decode's ple_gelu_mul per row: g[m][j] = gelu_tanh(g[m][j]) times
// row m's own per-layer input ple[m*LP + off + j] (off = l*P). grid = M*P.
kernel void ple_gelu_mul_f16(device half* g[[buffer(0)]], device const float* ple[[buffer(1)]], constant uint& P[[buffer(2)]],
    constant uint& LP[[buffer(3)]], constant uint& off[[buffer(4)]], uint gid[[thread_position_in_grid]]) {
    uint m = gid / P, j = gid % P;
    g[gid] = half(glu_act_f16(float(g[gid]), 0u) * ple[m*LP + off + j]);
}

// layer_scale_f16 (S9 step 2): x *= s[0] over the f16 residual, Gemma 4's dense layer scalar. grid = M*H.
kernel void layer_scale_f16(device half* x[[buffer(0)]], device const float* s[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    x[i] = half(float(x[i]) * s[0]);
}

// rope_f16: NeoX half-split, per-row position. Rotates a [M × total] region with row stride
// 'stride' starting at byte-free offset 'base0' (element offset into each row). positions[m].
kernel void rope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;   // nHeads*half; half=rotaryDim/2 (<hd/2 = partial rotary)
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base+dd]), x1=float(x[base+rhalf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+rhalf+dd]=half(x0*s+x1*c);
}

// rope_mrope_f16 (S16, docs/tasks/task-multimodal-support-2026-10.md): rope_f16 for Qwen's multimodal RoPE. Each row has
// three positions (temporal, height, width) in pos3[m*3..], and rotary pair dd turns by pos3[m*3 + axis[dd]]. axis is the
// host's own table from decoder's mropeComponent / mropeComponentInterleaved (contiguous sections for Qwen2.5-VL, strided
// for Qwen3.5+/Qwen3-VL), built once per model, so the kernel carries no layout formula of its own. A text row has three
// equal positions and reduces to rope_f16 at that position exactly.
kernel void rope_mrope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* pos3[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]], device const uint* axis[[buffer(9)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd;
    float th = float(pos3[m*3u + axis[dd]]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base+dd]), x1=float(x[base+rhalf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+rhalf+dd]=half(x0*s+x1*c);
}

// rope_f16_pw / rope_mrope_f16_pw (docs/tasks/task-metal-pairwise-rope-2026-10.md): rope_f16 / rope_mrope_f16 for GPT-J
// PAIRWISE rotation, pair dd = dims (2dd, 2dd+1), as decoder/rope.go's applyRoPEInterleaved and applyMRoPEPairwise (GLM-OCR's
// contiguous m-RoPE sections come in through the same host axis table). Same arguments and grid; bound for a model whose
// decoder.Model.PairwiseRoPEResident() is true, and nowhere else.
kernel void rope_f16_pw(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd + 2u*dd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base]), x1=float(x[base+1u]);
    x[base]=half(x0*c-x1*s); x[base+1u]=half(x0*s+x1*c);
}
kernel void rope_mrope_f16_pw(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* pos3[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]], device const uint* axis[[buffer(9)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd + 2u*dd;
    float th = float(pos3[m*3u + axis[dd]]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base]), x1=float(x[base+1u]);
    x[base]=half(x0*c-x1*s); x[base+1u]=half(x0*s+x1*c);
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

// attention_prefill_img (S17's Metal image prefill, docs/tasks/task-multimodal-support-2026-10.md): attention_prefill with
// image blocks [b.x, b.y) in absolute positions that attend BIDIRECTIONALLY, as the CPU's KVCache.attendHi and CUDA's
// PrefillImageLast: a query inside a block sees keys up to that block's b.y - 1 (its own block only; several blocks for
// several images, S11), every other query stays causal, and the sliding window's lower bound stays at the causal
// position's (pos + 1), as the CPU's WindowStart. nBlocks == 0 is no block (then this is attention_prefill exactly). A
// separate kernel so every existing dispatch of attention_prefill stays as it was.
kernel void attention_prefill_img(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    device const uint2* blocks[[buffer(11)]], constant uint& nBlocks[[buffer(12)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint pos = startPos + m;
    uint causal = pos + 1u;
    uint winStart = (window>0u && causal>window) ? causal-window : 0u;
    uint nKeys = causal;
    for (uint b=0u; b<nBlocks; b++) { if (pos >= blocks[b].x && pos < blocks[b].y) { nKeys = blocks[b].y; break; } }
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+s*kvDim; uint d=0;
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

// attention_prefill_steel (R19, docs/tasks/red-october.md): prefill attention in the shape of MLX 0.32.0's
// steel_attention (bq32 bk16 bd128), for head dim 128. One threadgroup per (query head, 32 query rows); its 4 simdgroups
// own 8 rows each, and each 16-key K/V block is staged ONCE in padded threadgroup memory and read by all 32 rows. O stays
// in registers (16 8x8 fragments per simdgroup, rescaled in place); the online softmax runs on every lane through the
// fragment lane map (mc3_frag, batch.go: this lane holds row fm, columns fn and fn+1; the 4 lanes of a row differ in
// lane bits 0 and 3), in exp2 with log2e folded into the score scale; P is written straight into PV's left-operand
// fragments; blocks past a row's causal or window limit are skipped. Measured 7.20x attention_prefill_fused in sequence
// at K=3900 on the 1.5B (docs/measurements/metal-prefill-attn-2026-09-27.md). NOT bit-identical to the fused kernel
// (accumulation order, exp2): graded by the §3.2 pooled gate on set A. The fused kernel keeps every other head dim.
#define R19_BQ 32
#define R19_BK 16
#define R19_HD 128
#define R19_LD (R19_HD + 8)

kernel void attention_prefill_steel(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]], ushort tid[[thread_index_in_threadgroup]]) {
    threadgroup half Ks[R19_BK * R19_LD];
    threadgroup half Vs[R19_BK * R19_LD];

    const uint nRB = (M + R19_BQ - 1u) / R19_BQ;
    const uint qh = tgid / nRB, rb = tgid % nRB;
    if (qh >= nH) return;
    const uint kvDim = nKV * R19_HD, kvh = qh / (nH / nKV), qDim = nH * R19_HD;
    const uint tr0 = rb * R19_BQ;                 // the threadgroup's first query row
    const uint r0 = tr0 + uint(sgid) * 8u;        // this simdgroup's first query row
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + fm;                   // the query row this lane's fragment elements belong to
    const uint myKeys = startPos + min(myRow, M - 1u) + 1u;               // causal: keys [0, myKeys)
    const uint myWin = (window > 0u && myKeys > window) ? myKeys - window : 0u;

    // The threadgroup's key range: its first row's window start to its last row's causal end.
    const uint tLast = min(tr0 + R19_BQ - 1u, M - 1u);
    const uint jEnd = startPos + tLast + 1u;
    const uint firstKeys = startPos + tr0 + 1u;
    const uint jStart = ((window > 0u && firstKeys > window) ? firstKeys - window : 0u) / R19_BK * R19_BK;
    // This simdgroup's own range, to skip blocks it has nothing in.
    const uint sLast = min(r0 + 7u, M - 1u);
    const uint sEnd = startPos + sLast + 1u;
    const uint sFirstKeys = startPos + min(r0, M - 1u) + 1u;
    const uint sStart = (window > 0u && sFirstKeys > window) ? sFirstKeys - window : 0u;

    const float sl2 = scale * 1.4426950408889634f;
    // A simdgroup whose 8 rows are all past M (the last block's tail) loads no Q (the qkv buffer is padded to 8 rows,
    // not 32) and computes nothing; it still helps stage K/V and meets every barrier.
    const bool active = r0 < M;
    simdgroup_half8x8 qT[R19_HD / 8];
    if (active) {
        for (ushort kk = 0; kk < R19_HD / 8; kk++) {
            simdgroup_load(qT[kk], qkv + r0 * qStride + qh * R19_HD + kk * 8u, qStride);
        }
    }
    simdgroup_float8x8 oAcc[R19_HD / 8];
    for (ushort cc = 0; cc < R19_HD / 8; cc++) oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mRow = -INFINITY, lRow = 0.0f;

    for (uint j0 = jStart; j0 < jEnd; j0 += R19_BK) {
        // Stage K and V for keys j0 .. j0+15: 2048 halves each, four half4 per thread; keys past jEnd read as zero.
        for (ushort i = 0; i < 4; i++) {
            uint idx = uint(tid) + uint(i) * 128u;
            uint key = idx >> 5, d4 = (idx & 31u) * 4u;
            uint j = j0 + key;
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (j < jEnd) {
                kv4 = *(device const half4*)(kc + j * kvDim + kvh * R19_HD + d4);
                vv4 = *(device const half4*)(vc + j * kvDim + kvh * R19_HD + d4);
            }
            *(threadgroup half4*)(Ks + key * R19_LD + d4) = kv4;
            *(threadgroup half4*)(Vs + key * R19_LD + d4) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && j0 < sEnd && j0 + R19_BK > sStart) {
            // S = Q K^T for this simdgroup's 8 rows and the block's 16 keys (two 8x8 fragments).
            simdgroup_float8x8 S[2];
            for (ushort sub = 0; sub < 2; sub++) {
                S[sub] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (ushort kk = 0; kk < R19_HD / 8; kk++) {
                    simdgroup_half8x8 kTile;
                    simdgroup_load(kTile, Ks + (sub * 8u) * R19_LD + kk * 8u, R19_LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S[sub], qT[kk], kTile, S[sub]);
                }
            }
            // This lane's four scores: row myRow, keys j0 + {fn, fn+1, 8+fn, 9+fn}.
            float s[4];
            s[0] = S[0].thread_elements()[0] * sl2; s[1] = S[0].thread_elements()[1] * sl2;
            s[2] = S[1].thread_elements()[0] * sl2; s[3] = S[1].thread_elements()[1] * sl2;
            const bool edge = (j0 + R19_BK > startPos + r0 + 1u) || (j0 < myWin + R19_BK) || (myRow >= M);
            if (edge) {
                uint js[4] = {j0 + fn, j0 + fn + 1u, j0 + 8u + fn, j0 + 9u + fn};
                for (ushort e = 0; e < 4; e++) {
                    if (js[e] >= myKeys || js[e] < myWin || myRow >= M) s[e] = -INFINITY;
                }
            }
            float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));
            float mNew = max(mRow, bmax);
            float factor = 1.0f;
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (mNew > -INFINITY) {
                factor = (mRow > -INFINITY) ? exp2(mRow - mNew) : 0.0f;
                for (ushort e = 0; e < 4; e++) p[e] = (s[e] > -INFINITY) ? exp2(s[e] - mNew) : 0.0f;
                mRow = mNew;
            }
            float bsum = (p[0] + p[1]) + (p[2] + p[3]);
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;
            simdgroup_half8x8 P[2];
            P[0].thread_elements()[0] = half(p[0]); P[0].thread_elements()[1] = half(p[1]);
            P[1].thread_elements()[0] = half(p[2]); P[1].thread_elements()[1] = half(p[3]);
            for (ushort cc = 0; cc < R19_HD / 8; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                for (ushort sub = 0; sub < 2; sub++) {
                    simdgroup_half8x8 vTile;
                    simdgroup_load(vTile, Vs + (sub * 8u) * R19_LD + cc * 8u, R19_LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P[sub], vTile, oAcc[cc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (myRow < M) {
        const float inv = lRow > 0.0f ? 1.0f / lRow : 0.0f;
        device half* o = out + myRow * qDim + qh * R19_HD + fn;
        for (ushort cc = 0; cc < R19_HD / 8; cc++) {
            o[cc * 8u] = half(oAcc[cc].thread_elements()[0] * inv);
            o[cc * 8u + 1u] = half(oAcc[cc].thread_elements()[1] * inv);
        }
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

// router_gemm_f16: batched router projection — normF[M×H](f16) · routerWᵀ[nE×H](f32) → logits[M×nE](f32).
// Grid = M*nE*32 threads (one simdgroup of 32 lanes per (token, expert) pair).
kernel void router_gemm_f16(
    device const half* in[[buffer(0)]],
    device const float* wf[[buffer(1)]],
    device float* logits[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& nE[[buffer(4)]],
    constant uint& M[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    uint sg = tid >> 5u;
    uint m = sg / nE;
    uint e = sg % nE;
    if (m >= M || e >= nE) return;
    uint lane = tid & 31u;
    device const float* wr = wf + e * H;
    device const half* ar = in + m * H;
    float acc = 0.0f;
    for (uint h = lane; h < H; h += 32u) {
        acc += wr[h] * float(ar[h]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        logits[m * nE + e] = acc;
    }
}

// moe_route_batch: batched top-k router selection for all M prompt rows.
// Grid = M*32 threads (simdgroup per token; lane 0 executes route logic).
kernel void moe_route_batch(
    device const float* logits[[buffer(0)]],
    device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]],
    device float* outWgt[[buffer(3)]],
    constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]],
    constant uint& sigmoid[[buffer(6)]],
    constant uint& norm[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    constant uint& nGroup[[buffer(9)]],
    constant uint& topkGroup[[buffer(10)]],
    constant uint& M[[buffer(11)]],
    uint tid[[thread_position_in_grid]]) {
    uint m = tid >> 5u;
    uint lane = tid & 31u;
    if (m >= M || lane != 0u) return;
    device const float* rowLogits = logits + m * nE;
    device uint* rowIdx = outIdx + m * k;
    device float* rowWgt = outWgt + m * k;

    float score[256];
    float sel[256];
    if (sigmoid != 0u) {
        for (uint i=0u;i<nE;i++) score[i] = 1.0f/(1.0f+exp(-rowLogits[i]));
    } else {
        float mx = rowLogits[0]; for (uint i=1u;i<nE;i++) mx = max(mx, rowLogits[i]);
        float sum = 0.0f; for (uint i=0u;i<nE;i++){ float e = exp(rowLogits[i]-mx); score[i]=e; sum+=e; }
        float inv = 1.0f/sum; for (uint i=0u;i<nE;i++) score[i] *= inv;
    }
    for (uint i=0u;i<nE;i++) sel[i] = score[i] + bias[i];

    if (nGroup > 1u) {
        uint gsz = nE / nGroup;
        float gscore[64];
        for (uint g=0u; g<nGroup; g++) {
            float t1=-INFINITY, t2=-INFINITY;
            for (uint i=g*gsz; i<(g+1u)*gsz; i++) { float v=sel[i]; if (v>t1){t2=t1;t1=v;} else if (v>t2){t2=v;} }
            gscore[g]=t1+t2;
        }
        bool keep[64];
        for (uint g=0u; g<nGroup; g++) keep[g]=false;
        for (uint j=0u; j<topkGroup; j++) {
            uint bg=0u; float bv=-INFINITY; bool found=false;
            for (uint g=0u; g<nGroup; g++) if (!keep[g] && gscore[g]>bv){ bv=gscore[g]; bg=g; found=true; }
            if (found) keep[bg]=true;
        }
        for (uint g=0u; g<nGroup; g++) if (!keep[g]) for (uint i=g*gsz; i<(g+1u)*gsz; i++) sel[i]=-INFINITY;
    }
    float wsum = 0.0f;
    for (uint j=0u; j<k; j++) {
        uint best=0u; float bv=-INFINITY;
        for (uint i=0u; i<nE; i++) if (sel[i]>bv){ bv=sel[i]; best=i; }
        rowIdx[j]=best; rowWgt[j]=score[best]; wsum+=score[best]; sel[best]=-INFINITY;
    }
    if (norm != 0u && wsum > 0.0f) for (uint j=0u; j<k; j++) rowWgt[j] /= wsum;
    if (scale != 0.0f && scale != 1.0f) for (uint j=0u; j<k; j++) rowWgt[j] *= scale;
}

// route_gptoss_batch: batched router for gpt-oss MoE.
kernel void route_gptoss_batch(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& M[[buffer(6)]],
    uint tid[[thread_position_in_grid]]) {
    uint m = tid >> 5u;
    uint lane = tid & 31u;
    if (m >= M || lane != 0u) return;
    device const float* rowLogits = logits + m * nE;
    device uint* rowIdx = outIdx + m * k;
    device float* rowWgt = outWgt + m * k;
    float sc[256];
    for (uint i=0u;i<nE;i++) sc[i] = rowLogits[i] + bias[i];
    bool taken[256];
    for (uint i=0u;i<nE;i++) taken[i]=false;
    float chosen[256];
    for (uint j=0u;j<k;j++) {
        int best=-1; float bv=-INFINITY;
        for (uint i=0u;i<nE;i++) if (!taken[i] && sc[i]>bv) { bv=sc[i]; best=int(i); }
        taken[uint(best)]=true;
        rowIdx[j]=uint(best);
        chosen[j]=bv;
    }
    float mx=chosen[0];
    for (uint j=1u;j<k;j++) mx=max(mx,chosen[j]);
    float sum=0.0f;
    for (uint j=0u;j<k;j++) { chosen[j]=exp(chosen[j]-mx); sum+=chosen[j]; }
    float inv=1.0f/sum;
    for (uint j=0u;j<k;j++) rowWgt[j]=chosen[j]*inv;
}

// gather_rows_f16: gathers Ce rows from src[M×H] into dst[CePad×H] according to rowIndices[Ce],
// writing 0.0h for any padding rows Ce <= i < CePad.
kernel void gather_rows_f16(
    device const half* src[[buffer(0)]],
    device half* dst[[buffer(1)]],
    device const uint* rowIndices[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& Ce[[buffer(4)]],
    constant uint& CePad[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = CePad * H;
    if (tid >= total) return;
    uint i = tid / H;
    uint h = tid % H;
    if (i < Ce) {
        uint srcRow = rowIndices[i];
        dst[i * H + h] = src[srcRow * H + h];
    } else {
        dst[i * H + h] = 0.0h;
    }
}

// scatter_add_weighted_f16: scatters Ce rows from src[CePad×H] into residual[M×H], multiplying by weights[Ce].
kernel void scatter_add_weighted_f16(
    device const half* src[[buffer(0)]],
    device half* residual[[buffer(1)]],
    device const uint* rowIndices[[buffer(2)]],
    device const float* weights[[buffer(3)]],
    constant uint& H[[buffer(4)]],
    constant uint& Ce[[buffer(5)]],
    constant uint& M[[buffer(6)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = Ce * H;
    if (tid >= total) return;
    uint i = tid / H;
    uint h = tid % H;
    uint dstRow = rowIndices[i];
    if (dstRow >= M) return;
    float w = weights[i];
    float val = float(src[i * H + h]) * w;
    residual[dstRow * H + h] += half(val);
}

// qgate_split_f16 (D-B01): a gated-attention layer's q_proj emits [query ‖ gate] PER HEAD at double width (Qwen3.5;
// decode's delta_qsplit). For M rows of 2*nHhd it writes the query into qkv's Q slot (row stride qStride) and the gate
// into gate rows of nHhd.
kernel void qgate_split_f16(device const half* qg[[buffer(0)]], device half* qkv[[buffer(1)]], device half* gate[[buffer(2)]],
    constant uint& nHhd[[buffer(3)]], constant uint& hd[[buffer(4)]], constant uint& qStride[[buffer(5)]],
    constant uint& M[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    if (i >= M*nHhd) return;
    uint m = i / nHhd, t = i % nHhd;
    uint h = t / hd, d = t % hd;
    uint base = m*2u*nHhd + h*2u*hd + d;
    qkv[m*qStride + t] = qg[base];
    gate[i] = qg[base+hd];
}

// copy_cols_f16 (D-B01): rows of n halves into columns off..off+n of rows dstStride wide (the K‖V projection of a gated
// layer into qkv's K and V slots).
kernel void copy_cols_f16(device const half* src[[buffer(0)]], device half* dst[[buffer(1)]], constant uint& n[[buffer(2)]],
    constant uint& dstStride[[buffer(3)]], constant uint& off[[buffer(4)]], constant uint& M[[buffer(5)]],
    uint i[[thread_position_in_grid]]) {
    if (i >= M*n) return;
    uint m = i / n, j = i % n;
    dst[m*dstStride + off + j] = src[i];
}

// attn_gate_f16 (D-B01): ctx *= sigmoid(gate) in place, M rows of n, before o_proj (decode's delta_attn_gate).
kernel void attn_gate_f16(device half* ctx[[buffer(0)]], device const half* gate[[buffer(1)]], constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]) {
    if (i >= n) return;
    ctx[i] = half(float(ctx[i]) / (1.0f + exp(-float(gate[i]))));
}

// shared_gate_add_f16: qwen2_moe gated shared expert — residual[m*H+h] += sigmoid(gl[m]) * src[m*H+h].
kernel void shared_gate_add_f16(
    device const half* src[[buffer(0)]],
    device half* residual[[buffer(1)]],
    device const float* gl[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& M[[buffer(4)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = M * H;
    if (tid >= total) return;
    uint m = tid / H;
    uint h = tid % H;
    float sig = 1.0f / (1.0f + exp(-gl[m]));
    float val = float(src[m * H + h]) * sig;
    residual[m * H + h] += half(val);
}
`

// prefillSteelAttnOff, set only by tests, runs attention_prefill_fused where attention_prefill_steel would run: the
// baseline arm of R19's §3.2 gate and of the decomposition harness's A/B.
var prefillSteelAttnOff bool

// gemmTilePolicy picks gemmTile's rule: "" the default (D-B02's padding rule below), "a01" A-P01's rule as it shipped,
// "shipped" every GEMM on gemm_w4f16_store's 64 × 64 tile, "bm32" / "bn32" only A-P01's 32-token / 32-feature rule.
// Test-only: the arms of the grades (docs/tasks/task-metal-audit-2026-10.md); not an option or an environment variable.
var gemmTilePolicy = ""

// gemmTile is the tile selector for a rows x N prefill GEMM. Every choice is bit-identical (gemm_w4f16_tile's comment,
// TestGemmTile_bitIdentical).
//
// The default: a threadgroup costs the same whatever share of its rows is real, so a smaller token tile is taken when
// it pads the pass to strictly fewer rows: 32 tokens up to 128 rows, 16 up to 16 rows and, when N <= 8192, up to 48.
// Past those, more row tiles cost more than the padding they save, and a gate|up GEMM as wide as the 1.5B's or 7B's
// already fills the cores at 64 (docs/tasks/task-metal-audit-2026-10.md "D-B02: the 16-row tile";
// TestGemmTile16_probe). A 16-token tile always takes 32 features; otherwise 32 features when N <= 2048 and rows <=
// 64, so the narrow GEMMs fill more cores.
//
// gemmTilePolicy "a01" is the older rule: 32 tokens when the pass has at most 32 rows, 32 features when N <= 2048 and
// rows <= 64.
func (pf *prefillState) gemmTile(rows, N int) (Pipeline, int, int) {
	return pf.gemmTileFor(rows, N, pf.w8)
}

// gemmTileFor is gemmTile for a weight of a given kind: w8 picks the W8 tiles (int8 slice 2), else the int4 ones. An
// int4mix resident (w8Attn) passes true for its attention GEMMs and false for its FFN.
func (pf *prefillState) gemmTileFor(rows, N int, w8 bool) (Pipeline, int, int) {
	tm, tn := 64, 64
	switch gemmTilePolicy {
	case "":
		pad := (rows + 63) / 64 * 64
		if p := (rows + 31) / 32 * 32; p < pad && rows <= 128 {
			tm, pad = 32, p
		}
		if p := (rows + 15) / 16 * 16; p < pad && (rows <= 16 || rows <= 48 && N <= 8192) {
			tm = 16
		}
		if tm == 16 || (N <= 2048 && rows <= 64) {
			tn = 32
		}
	case "shipped":
	default:
		if gemmTilePolicy != "bn32" && rows <= 32 {
			tm = 32
		}
		if gemmTilePolicy != "bm32" && N <= 2048 && rows <= 64 {
			tn = 32
		}
	}
	if w8 { // int8 slice 2: the native int8 path's weights, the same tile
		return pf.pW8[[2]int{tm, tn}], tm, tn
	}
	switch {
	case tm == 16 && tn == 32:
		return pf.pGemmM16N32, 16, 32
	case tm == 32 && tn == 32:
		return pf.pGemmM32N32, 32, 32
	case tm == 32:
		return pf.pGemmM32N64, 32, 64
	case tn == 32:
		return pf.pGemmM64N32, 64, 32
	}
	return pf.pGemmStore, 64, 64
}

// prefillState holds the lazily compiled prefill pipelines (ensurePrefill); a decode-only run never builds it.
type prefillState struct {
	pGemmStore, pRms, pRes, pSw, pRope, pKv, pAttn, pQK, pRmsQ Pipeline
	pLN, pLNQ                                                  Pipeline // Part B: layernorm_f16 / layernorm_quant_f16 (bias-free LayerNorm)
	pAttnImg                                                   Pipeline // S17: attention_prefill with a bidirectional image block
	pRopeM                                                     Pipeline // S16: rope_mrope_f16
	pPLEGeluMulF16, pLayerScaleF16                             Pipeline // S9 step 2: the E-model PLE gate and layer scalar
	// A-P01: gemm_w4f16_tile at the smaller tiles (tokens × features), picked by gemmTile.
	pGemmM32N64, pGemmM64N32, pGemmM32N32 Pipeline
	pGemmM16N32                           Pipeline // D-B02's 16-token tile
	// int8 slice 2 (docs/tasks/task-metal-int8-2026-10.md): a native int8 resident's GEMMs, gemm_w4f16_tile<TM, TN,
	// true> by (TM, TN); w8 selects them in gemmTile.
	w8  bool
	pW8 map[[2]int]Pipeline
	// G8 (docs/tasks/task-gpu-paths-2026-09.md): the MoE row loop's F32-scratch bridge (see
	// residual_f16_from_f32/zero_f32's own comments).
	pResF32, pZeroF32 Pipeline
	pResScaled        Pipeline // S17: residual_f16_scaled, the add into a residual stored scaled by 1/s
	// attention_prefill_fused: the simdgroup_matrix flash-attention twin of pAttn
	// (docs/completed/task-prefill-gap.md §4). On by default (metalFusedAttentionEnabled, backend.go);
	// GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill falls back to pAttn.
	pAttnFused Pipeline
	// R19: attention_prefill_steel, the head-dim-128 prefill attention (it replaces pAttnFused there).
	pAttnSteel Pipeline

	// Expert-major MoE prefill pipelines
	pRouterGemm       Pipeline
	pMoeRouteBatch    Pipeline
	pRouteGptOssBatch Pipeline
	pGatherRows       Pipeline
	pScatterAdd       Pipeline
	pSharedGateAdd    Pipeline

	// D-B01: a gated-attention layer (Qwen3.5) on the batched lane; the DeltaNet mixer's own row kernels live in the
	// resident library (r.pDn*Rows), beside the decode kernels they copy.
	pQGateSplit, pCopyCols, pAttnGate Pipeline
}

// ropeF16Name is the batched prefill's rope kernel: rope_f16, or rope_mrope_f16 for the m-RoPE pass, and their pairwise
// twins for a GPT-J pairwise model (docs/tasks/task-metal-pairwise-rope-2026-10.md).
func ropeF16Name(pairwise, mrope bool) string {
	n := "rope_f16"
	if mrope {
		n = "rope_mrope_f16"
	}
	if pairwise {
		n += "_pw"
	}
	return n
}

func (r *resident) ensurePrefill() {
	if r.pf != nil {
		return
	}
	if r.pfErr != nil {
		// A prior attempt already failed: re-panic the same cached error instead of re-running the full MSL compile to
		// fail identically.
		panic(r.pfErr)
	}
	// Compile and pipeline creation here run pool-less on an unpinned thread (PrefillLast calls this before its
	// own LockOSThread). Pin and hold a pool so the autoreleased temporaries drain; the +1-owned library and
	// pipelines are tracked on the Device and freed at Close.
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
	compile := r.d.CompileLibrary
	if (r.w8 || r.w8Attn) && r.preciseMath { // the owner's precise-math decision for native int8 (w8PreciseMath) covers its prefill too
		compile = r.d.CompileLibraryPrecise
	}
	lib, err := compile(prefillKernels, MSL3_1)
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
		pGemmM32N64: p("gemm_w4f16_m32n64"), pGemmM64N32: p("gemm_w4f16_m64n32"), pGemmM32N32: p("gemm_w4f16_m32n32"), pGemmM16N32: p("gemm_w4f16_m16n32"),
		pLN: p("layernorm_f16"), pLNQ: p("layernorm_quant_f16"),
		pRes: p("residual_f16"), pSw: p("swiglu_f16"), pRope: p(ropeF16Name(r.pairwiseRoPE, false)), pRopeM: p(ropeF16Name(r.pairwiseRoPE, true)),
		pPLEGeluMulF16: p("ple_gelu_mul_f16"), pLayerScaleF16: p("layer_scale_f16"),
		pKv: p("kv_store_f16"), pAttn: p("attention_prefill"), pAttnImg: p("attention_prefill_img"), pQK: p("qk_norm_f16"),
		pRmsQ:   p("rmsnorm_quant_f16"),
		pResF32: p("residual_f16_from_f32"), pZeroF32: p("zero_f32"), pResScaled: p("residual_f16_scaled"),
		pAttnFused: p("attention_prefill_fused"), pAttnSteel: p("attention_prefill_steel"),

		pRouterGemm:       p("router_gemm_f16"),
		pMoeRouteBatch:    p("moe_route_batch"),
		pRouteGptOssBatch: p("route_gptoss_batch"),
		pGatherRows:       p("gather_rows_f16"),
		pScatterAdd:       p("scatter_add_weighted_f16"),
		pSharedGateAdd:    p("shared_gate_add_f16"),

		pQGateSplit: p("qgate_split_f16"), pCopyCols: p("copy_cols_f16"), pAttnGate: p("attn_gate_f16"),
	}
	if r.w8 || r.w8Attn {
		r.pf.w8, r.pf.pW8 = r.w8, map[[2]int]Pipeline{}
		for _, t := range [][2]int{{64, 64}, {32, 64}, {64, 32}, {32, 32}, {16, 32}} {
			r.pf.pW8[t] = p(fmt.Sprintf("gemm_w8f16_m%dn%d", t[0], t[1]))
		}
	}
}

// parallelEmbedsF32ToF16 converts M rows of embs (each H wide) into dst[m*H:(m+1)*H] as f16 bits, splitting across up
// to 8 workers by row. Every element is independent and f32ToF16 is a pure function of its input, so the split is
// byte-identical to the serial loop, as parallelF32ToF16's is. It does not reuse parallelF32ToF16 because embs is
// [][]float32 (one slice per row, not necessarily contiguous) where that wants one flat slice, and flattening first
// would pay its own copy; the threshold and worker shape are the same.
func parallelEmbedsF32ToF16(dst []uint16, embs [][]float32, H int) {
	parallelEmbedsF32ToF16Scaled(dst, embs, H, 1)
}

// parallelEmbedsF32ToF16Scaled is parallelEmbedsF32ToF16 with every value times inv first (S17's scaled residual; inv is
// a power of two, so the product is exact, and inv = 1 is the plain conversion bit for bit).
func parallelEmbedsF32ToF16Scaled(dst []uint16, embs [][]float32, H int, inv float32) {
	M := len(embs)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if M*H < 8192 || workers <= 1 || M < 2 {
		for m := range M {
			row := embs[m]
			for i := range H {
				dst[m*H+i] = f32ToF16(row[i] * inv)
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
					dst[m*H+i] = f32ToF16(row[i] * inv)
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

// prefillExactAttnMaxKeys is the exact attention_prefill kernel's bound: it keeps one score per key in `threadgroup
// float sc[4096]`, indexed by the key's absolute position, with no tiling (the decode kernels tile theirs,
// attnScoreTileBound), so above 4096 keys it writes past threadgroup memory. The exact kernel runs whenever the fused
// one cannot (head dim above 128 or not a multiple of 8, or GOINFER_METAL_FUSED_ATTENTION off), and a resident context
// reaches 32768 for an explicit -ctx or a guard-pinned load, so metalResident.PrefillLast declines such a pass to the
// sequential path. TestPrefillExactAttnBound ties this to the kernel source.
const prefillExactAttnMaxKeys = 4096

// prefillAttnKernels is which attention kernel PrefillLast dispatches for this model: fused for hd%8==0 && hd<=128
// (attention_prefill_fused's compile-time cap, ATTN_MAXHD), steel within that for hd 128 (R19: one threadgroup of 128
// threads per query head and 32 rows), and otherwise the exact attention_prefill, bounded by prefillExactAttnMaxKeys.
func (r *resident) prefillAttnKernels() (fused, steel bool) {
	g0 := r.prefillGeom()
	fused = metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION")) && g0.hd%8 == 0 && g0.hd <= 128
	steel = fused && g0.hd == 128 && !prefillSteelAttnOff
	return fused, steel
}

// prefillExactAttn reports whether any attention layer of the pass runs the exact attention kernel (no fused kernel for
// its head dim, or fused attention off), whose score buffer holds prefillExactAttnMaxKeys keys. A Gemma 4 E-model's
// layers differ (S9 step 2); every other family's are one geometry.
func (r *resident) prefillExactAttn() bool {
	fusedOn := metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION"))
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && !(fusedOn && g.hd%8 == 0 && g.hd <= 128) {
			return true
		}
	}
	return false
}

// prefillGeom is the attention geometry the pass uses for every attention layer: the first attention layer's. A
// Gated-DeltaNet hybrid's layer 0 is a DeltaNet layer, which has none (D-B01); every other admitted family's layer 0
// has it.
func (r *resident) prefillGeom() *attnGeom {
	for i := range r.layers {
		if g := r.layers[i].geom; g != nil {
			return g
		}
	}
	return nil
}

// PrefillLast ingests M prompt embeddings at positions startPos..startPos+M-1 in one command buffer via the f16 MMA path
// (weights read once, amortized across M, unlike the token-by-token decode loop), populating the resident KV cache, and
// returns the last token's logits[V], what a generator needs to sample the first output token. Correctness-gated against
// the sequential path.
func (r *resident) PrefillLast(embs [][]float32, startPos int) []float32 {
	return r.prefillLast(embs, startPos, nil, nil)
}

// prefillDeep is Qwen3-VL's DeepStack injection for the batched pass (S16): set l is added to the image rows [start,
// start+n) of the f16 residual after decoder layer l, as decoder's addDeepstack does on the CPU.
type prefillDeep struct {
	start, n int
	sets     [][]float32
}

// prefillParallelDefectForTest is a planted defect (docs/tasks/task-metal-pairwise-followups-2026-10.md, G-B2): a
// parallel block's MLP fed from the post-attention residual instead of the layer's shared input norm. Tests set it.
var prefillParallelDefectForTest bool

// prefillResidScaleForTest, when > 0, replaces prefillResidScale's s for every family it applies to (1 turns the scale
// off: G-RS2's planted defect; a large s measures the residual's peak).
var prefillResidScaleForTest float32

// gemmaPrefillResidScale is s for the families prefillResidScale applies to: the smallest power of two that leaves 4x
// headroom under f16's 65,504 over Gemma 3 4B's residual peak (its <bos> row after layer 31;
// docs/code-notes/metal.md#gemmaPrefillResidScale).
const gemmaPrefillResidScale = 32

// prefillResidScale is the s the batched pass stores its residual divided by (S17,
// docs/tasks/task-multimodal-support-2026-10.md): Gemma 3 4B's residual outgrows f16 (inf at layer 6). It applies where
// every read of the residual is an RMSNorm, which ignores scale except through eps, and every write is an embedding
// upload or a post-normed branch's residual add: sandwich norms with no post-only norm, PLE, MoE or DeepStack rows. s is
// a power of two, so the scaled pass equals the unscaled one wherever that one does not overflow (eps and subnormals
// aside). 1 everywhere else: those families run the unchanged kernels.
func (r *resident) prefillResidScale(deep *prefillDeep) float32 {
	if !r.sandwich || r.postOnly || r.pleP != 0 || r.moe != nil || deep != nil {
		return 1
	}
	if prefillResidScaleForTest > 0 {
		return prefillResidScaleForTest
	}
	return gemmaPrefillResidScale
}

// prefillLast is PrefillLast with optional m-RoPE positions: mrope, when non-nil, holds every absolute position's
// (temporal, height, width) triple from 0 to startPos+len(embs), and the pass rotates q and k by them through
// rope_mrope_f16 and r.mropeAxis. The K/V cache rows are still placed by sequence position.
func (r *resident) prefillLast(embs [][]float32, startPos int, mrope [][3]int, deep *prefillDeep) []float32 {
	return r.prefillLastImg(embs, startPos, mrope, deep, nil)
}

// prefillLastImg is prefillLast with image blocks, each [start, end) in absolute positions, that attend bidirectionally
// (S17's Metal image prefill; several blocks for several images, S11): every layer's attention then runs
// attention_prefill_img, the exact kernel with the blocks, in place of the fused and steel kernels, which carry no block
// mask. No blocks: prefillLast exactly.
func (r *resident) prefillLastImg(embs [][]float32, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) []float32 {
	r.ensurePrefill()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, pf := r.d, r.pf
	normPipe := pf.pRms // the GEMM-input norms: RMSNorm, or a bias-free LayerNorm for Cohere (Part B)
	if r.layerNorm {
		normPipe = pf.pLN
	}
	M := len(embs)
	Mpad := (M + 7) / 8 * 8
	H, I, V := r.H, r.I, r.V
	// Prefill runs only for uniform families (prefillOK declines Gemma's per-layer geometry), so
	// every attention layer shares one geometry.
	g0 := r.prefillGeom()
	nHhd := r.nH * g0.hd
	kvDim := g0.kvDim
	qkvDim := nHhd + 2*kvDim
	qDim := nHhd

	maxI := I
	if r.moe != nil {
		if r.moe.inter > maxI {
			maxI = r.moe.inter
		}
		if r.moe.sharedInter > maxI {
			maxI = r.moe.sharedInter
		}
	}

	// S9 step 2: a Gemma 4 E-model varies the attention geometry and the FFN width by layer; the scratch fits the largest.
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil {
			qkvDim, qDim = max(qkvDim, r.nH*g.hd+2*g.kvDim), max(qDim, r.nH*g.hd)
		}
		maxI = max(maxI, r.layers[l].ffnI)
	}

	// f16 activation scratch (per call, sized to the padded prompt).
	// xF and the five prefillScratchU16 buffers below rely on Metal's zero-fill of a new buffer for their pad rows
	// (aikit gpu.NewBufferLen* contract: Metal only; CUDA's is uninitialized) — see prefillScratchU16.
	// xF too (R-20): the embeddings are converted straight into the buffer's M rows, its pad rows left as Metal zeroed
	// them, instead of into a zeroed Go slice of Mpad rows that was then copied in whole.
	xF := prefillScratchU16(d, Mpad*H)
	residInv := 1 / r.prefillResidScale(deep)
	parallelEmbedsF32ToF16Scaled(xF.U16s()[:M*H], embs, H, residInv)
	normF := prefillScratchU16(d, Mpad*H)
	qkvF := prefillScratchU16(d, Mpad*qkvDim)
	ctxF := prefillScratchU16(d, Mpad*qDim)
	guF := prefillScratchU16(d, Mpad*2*maxI)
	dqF := prefillScratchU16(d, Mpad*maxI)
	posv := make([]uint32, Mpad)
	for m := range M {
		posv[m] = uint32(startPos + m)
	}
	posB := NewBufferUint32s(d, posv)
	var pos3B Buffer // S16: the rows' m-RoPE triples, row-major [Mpad*3]; pad rows repeat the last real row's
	if mrope != nil {
		p3 := make([]uint32, Mpad*3)
		for m := range Mpad {
			t := mrope[startPos+min(m, M-1)]
			p3[m*3], p3[m*3+1], p3[m*3+2] = uint32(t[0]), uint32(t[1]), uint32(t[2])
		}
		pos3B = NewBufferUint32s(d, p3)
	}

	// uniforms
	uM := NewBufferU32(d, uint32(Mpad))
	uH := r.uH
	uI := NewBufferU32(d, uint32(I))
	u2I := NewBufferU32(d, uint32(2*I))
	uQkv := NewBufferU32(d, uint32(qkvDim))
	uQDim := NewBufferU32(d, uint32(qDim))
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

	// MoE expert-major prefill scratch buffers (allocated only when model has MoE layers).
	var (
		moeLogits  Buffer
		moeIdx     Buffer
		moeWgt     Buffer
		expertIn   Buffer
		expertDown Buffer
		rowIdxBuf  Buffer
		rowWgtBuf  Buffer
	)
	if r.moe != nil {
		// R-20: zero-filled by Metal, not copied from zeroed Go slices; the router GEMM, the route kernels and the host
		// fill them before anything reads them (see prefillScratch).
		moeLogits = prefillScratch[float32](d, M*r.moe.nE)
		moeIdx = prefillScratch[uint32](d, M*r.moe.k)
		moeWgt = prefillScratch[float32](d, M*r.moe.k)
		// expertIn/expertDown rely on the same Metal zero-fill for their pad rows as the scratch above (see prefillScratchU16).
		expertIn = prefillScratchU16(d, Mpad*H)
		expertDown = prefillScratchU16(d, Mpad*H)
		rowIdxBuf = prefillScratch[uint32](d, Mpad*r.moe.k)
		rowWgtBuf = prefillScratch[float32](d, Mpad*r.moe.k)
		if prefillScratchPoison { // a test's proof that nothing reads these before the pass writes them
			for _, b := range []Buffer{moeLogits, moeIdx, moeWgt, rowIdxBuf, rowWgtBuf} {
				for i, u := 0, b.U32s(); i < len(u); i++ {
					u[i] = 0xFFFFFFFF
				}
			}
		}
	}

	// Every buffer above is per-call scratch or uniform allocated onto the device ledger, which ReleaseAll frees
	// only at Close, so each is released at the end of the call (e.End() below commits and waits, so the GPU is
	// finished with them by then). Unreleased they ratchet by a prompt's worth per call (guF alone is Mpad*2I*2)
	// until the mustBuf OOM panic kills serve, which is recovered only on the BuildResident path, not here. r.uH,
	// r.uKvDim and r.uHd are resident-owned and reused, so deliberately not in this list: releasing them would
	// corrupt the decode path.
	scratch := []Buffer{
		xF, normF, qkvF, ctxF, guF, dqF, posB, moeDst,
		uM, uI, u2I, uQkv, uQDim, uStride, uKOff, uVOff, uStartPos,
		uTotalQ, uTotalK, uBase0, uBaseK, m0, m1, m2, dummyBias, uMReal,
	}
	if r.moe != nil {
		scratch = append(scratch, moeLogits, moeIdx, moeWgt, expertIn, expertDown, rowIdxBuf, rowWgtBuf)
	}
	// S17: the scaled residual's 1/s, for residual_f16_scaled, and the eps of the norms that read the residual, eps/s², so
	// that x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly (s is a power of two).
	var uResidInv Buffer
	uEpsX := r.uEps
	if residInv != 1 {
		uResidInv = NewBufferFloats(d, []float32{residInv})
		uEpsX = NewBufferFloats(d, []float32{r.uEps.Floats()[0] * residInv * residInv})
		scratch = append(scratch, uResidInv, uEpsX)
	}
	// S9 step 2: a Gemma 4 E-model's per-layer inputs, each row's [L·P] tail of its [h ‖ L·P] embedding, and the PLE
	// gate's scratch.
	var pleInB, pleGF, uP, uLP Buffer
	if r.pleP > 0 {
		P, LP := r.pleP, r.nL*r.pleP
		pv := make([]float32, M*LP)
		for m := range M {
			src := embs[m]
			if emodelBatchDefect == 3 { // G-S9c (3): each row's PLE inputs taken from the next row
				src = embs[min(m+1, M-1)]
			}
			copy(pv[m*LP:(m+1)*LP], src[H:H+LP])
		}
		pleInB, pleGF, uP, uLP = NewBufferFloats(d, pv), prefillScratchU16(d, Mpad*P), NewBufferU32(d, uint32(P)), NewBufferU32(d, uint32(LP))
		scratch = append(scratch, pleInB, pleGF, uP, uLP)
	}
	// D-B01: a Gated-DeltaNet hybrid's mixer scratch, and its gated-attention layers' [query ‖ gate] projection, K‖V
	// projection and split gate.
	var dn *prefillDelta
	var qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN Buffer
	if r.dnet != nil {
		dn = r.newPrefillDelta(M, Mpad)
		qgF = prefillScratchU16(d, Mpad*2*nHhd)
		kvF = prefillScratchU16(d, Mpad*2*kvDim)
		gateF = prefillScratchU16(d, M*nHhd)
		u2NHhd, u2KvDim, uGateN = NewBufferU32(d, uint32(2*nHhd)), NewBufferU32(d, uint32(2*kvDim)), NewBufferU32(d, uint32(M*nHhd))
		scratch = append(append(scratch, dn.bufs...), qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN)
	}
	var imgB, imgN Buffer // the bidirectional image blocks and their count, when there are any (attention_prefill_img)
	if len(blocks) > 0 {
		flat := make([]uint32, 0, 2*len(blocks))
		for _, b := range blocks {
			flat = append(flat, uint32(b[0]), uint32(b[1]))
		}
		imgB, imgN = NewBufferUint32s(d, flat), NewBufferU32(d, uint32(len(blocks)))
		scratch = append(scratch, imgB, imgN)
	}
	defer func() {
		for _, b := range scratch {
			d.ReleaseBuf(b)
		}
	}()

	// Uniform buffer cache to avoid per-expert dynamic allocations
	uConsts := make(map[int]Buffer)
	getU32 := func(val int) Buffer {
		if b, ok := uConsts[val]; ok {
			return b
		}
		b := NewBufferU32(d, uint32(val))
		uConsts[val] = b
		scratch = append(scratch, b)
		return b
	}

	// gemm: one GEMM dispatch over rows×N — a 2-D grid of (ceil(N/TN), ceil(rows/TM)) threadgroups of 128 threads, each
	// owning a TN-feature × TM-token tile: gemm_w4f16_store's 64 × 64 (R16), or for a small pass a smaller tile of the
	// same arithmetic (A-P01, gemmTile).
	gemm := func(e *Encoder, rows, N int, bufs ...Buffer) {
		p, tm, tn := pf.gemmTile(rows, N)
		e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
	}
	// gemmAttn is gemm for the attention projections (qkv, o), whose weights are int8 on an int4mix resident too.
	gemmAttn := func(e *Encoder, rows, N int, bufs ...Buffer) {
		p, tm, tn := pf.gemmTileFor(rows, N, r.w8 || r.w8Attn)
		e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
	}
	// L2-Metal: attention_prefill_fused's grid — nH×ceil(M/8) simdgroups (ATTN_SGPT=4/threadgroup,
	// prefill.go's own #define, matched here). Real M (unpadded): the tail row-tile's out-of-range
	// rows are masked in-kernel via buffer(11), not dropped from the grid.
	const attnFusedSGPT = 4
	numRowTiles := (M + 7) / 8
	attnFusedTotal := r.nH * numRowTiles
	attnFusedTotal = (attnFusedTotal + attnFusedSGPT - 1) / attnFusedSGPT * attnFusedSGPT * 32
	attnFusedTg := attnFusedSGPT * 32
	// S9 step 2: one attention geometry's uniforms, per layer. Every family before Gemma 4's E-models has one geometry, so
	// it gets one bundle with the values the single-geometry pass used; an E-model gets one per head dim, and a KV-shared
	// layer's Q-only projection its own stride. The fused/steel attention kernels hold head dims up to 128; past that the
	// exact kernel runs (its key limit is checked before the pass).
	type layerU struct {
		g                                                            *attnGeom
		qkvDim, nHhd, kvDim                                          int
		uQkv, uStride, uKOff, uVOff, uTotalQ, uTotalK, uBaseK, uQDim Buffer
		fused, steel                                                 bool
	}
	type luKey struct {
		g      *attnGeom
		shared bool
	}
	luCache := map[luKey]*layerU{}
	fusedOn := metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION"))
	layerUFor := func(L *residLayer) *layerU {
		k := luKey{L.geom, L.kvShared}
		if u, ok := luCache[k]; ok {
			return u
		}
		g := L.geom
		nh := r.nH * g.hd
		q := nh + 2*g.kvDim
		if L.kvShared {
			q = nh // Q only: a shared layer projects no K or V
		}
		fused := fusedOn && g.hd%8 == 0 && g.hd <= 128
		u := &layerU{g: g, qkvDim: q, nHhd: nh, kvDim: g.kvDim,
			uQkv: getU32(q), uStride: getU32(q), uKOff: getU32(nh), uVOff: getU32(nh + g.kvDim),
			uTotalQ: getU32(nh), uTotalK: getU32(g.kvDim), uBaseK: getU32(nh), uQDim: getU32(nh),
			fused: fused, steel: fused && g.hd == 128 && !prefillSteelAttnOff}
		luCache[k] = u
		return u
	}
	// G-S9c (1): a shared layer attending over an empty cache instead of its source's (zero in production).
	var zeroKV Buffer
	if emodelBatchDefect == 1 {
		zeroKV = prefillScratchU16(d, (startPos+Mpad+8)*qkvDim)
		scratch = append(scratch, zeroKV)
	}

	e := r.q.Begin()
	// S16: the DeepStack sets, uploaded once; addDeep(k) adds set k to the rows of this pass that fall in the image block.
	// Set k is added after layer k, which is done at the top of iteration k+1 and after the loop, so every branch of the
	// loop body (the MoE ones end in continue) reaches it.
	var deepB []Buffer
	if deep != nil {
		for _, set := range deep.sets {
			deepB = append(deepB, NewBufferFloats(d, set))
		}
	}
	addDeep := func(k int) {
		if k >= len(deepB) {
			return
		}
		lo, hi := max(deep.start, startPos), min(deep.start+deep.n, startPos+M)
		if hi <= lo {
			return
		}
		e.Dispatch(pf.pResF32, (hi-lo)*H, 256, xF.At((lo-startPos)*H*2), deepB[k].At((lo-deep.start)*H*4))
	}
	for l := 0; l < r.nL; l++ {
		if l > 0 {
			addDeep(l - 1)
		}
		L := &r.layers[l]
		if L.delta != nil {
			r.encodePrefillDeltaMixer(e, L, xF, normF, dn, M, Mpad, gemm, uM, m0, m2, dummyBias)
		} else {
			lu := layerUFor(L)
			// pre-attn norm — addOne (Gemma's 1+w) matters even for a plain non-sandwich family's
			// GEMV-input norm, so it is always passed (0 for every family without RMSAddOne).
			// Olmo 3 / Olmo Hybrid (postOnly): no pre-norm; GEMV reads raw residual xF directly.
			inAttn := normF
			if r.postOnly {
				inAttn = xF
			} else {
				e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
			}
			// fused QKV (+bias)
			if L.qGate {
				// D-B01, Qwen3.5's gated softmax layer (decode's encodeAttention qGate branch): q_proj is [query ‖ gate]
				// per head at double width, split on the activation into qkv's Q slot and the gate; K‖V is its own
				// bias-free projection, copied into qkv's K and V slots.
				gemm(e, Mpad, 2*nHhd, inAttn, L.dnQw, L.dnQs, qgF, uM, u2NHhd, uH, dummyBias, m0)
				e.Dispatch(pf.pQGateSplit, M*nHhd, 256, qgF, qkvF, gateF, uTotalQ, uHd, uQkv, uMReal)
				gemm(e, Mpad, 2*kvDim, inAttn, L.qkvW, L.qkvS, kvF, uM, u2KvDim, uH, dummyBias, m0)
				e.Dispatch(pf.pCopyCols, M*2*kvDim, 256, kvF, qkvF, u2KvDim, uQkv, uKOff, uMReal)
			} else {
				gemmAttn(e, Mpad, lu.qkvDim, inAttn, L.qkvW, L.qkvS, qkvF, uM, lu.uQkv, uH, L.qkvBias, m1)
			}
			if r.qkNorm { // Qwen3 / Olmo 3: per-head Q/K RMSNorm before RoPE
				qkNH, qkNKV, qkHD, qkNHhd := r.uNH, lu.g.uNKV, lu.g.uHd, lu.g.uNHhd
				tgCount := r.nH + lu.g.nKV
				if L.kvShared {
					// Q heads only: a shared layer projects no K. The kernel maps a threadgroup to (row, head) by nH+nKV, so nKV
					// must be 0 here too, or rows shift and the "K" head normalizes the next row's Q.
					tgCount, qkNKV = r.nH, r.uZero
				}
				if r.qkNormWhole {
					qkNH, qkNKV, qkHD, qkNHhd = r.uQKWholeOne, r.uQKWholeOne, r.uQKWholeHD, r.uQKWholeHD
					tgCount = 2
				}
				e.Dispatch(pf.pQK, M*tgCount*tgReduceAttn, tgReduceAttn, qkvF, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, qkNHhd, lu.uStride, r.uEps, r.uAddOne)
			}
			// S9 step 2: Gemma 4's scale-less v_norm on every K/V-owning layer's V slot, after Q/K norm and before RoPE, as
			// decode does (encodeAttentionResidualWith): the Q/K-norm kernel over the V slot with nH = 0, a unit weight and
			// addOne 0.
			if !L.kvShared && r.g4VNorm && emodelBatchDefect != 5 {
				e.Dispatch(pf.pQK, M*lu.g.nKV*tgReduceAttn, tgReduceAttn, qkvF.At((lu.nHhd+lu.kvDim)*2), r.vNormUnit, r.vNormUnit,
					r.uZero, lu.g.uNKV, lu.g.uHd, r.uZero, lu.uStride, r.uEps, r.uZero)
			}
			// rope q, k (per-row positions): bind the per-layer RoPE table and window, exactly as decode does
			// (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window arch the global
			// layers must see window=0, and each layer its own RoPE base; FeatPerLayerRoPE is in prefillFeatures, so the
			// per-layer binding is not behaviour-neutral.
			if mrope != nil { // S16: each row's own (t, h, w) rotation
				e.Dispatch(pf.pRopeM, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale, r.mropeAxis)
				if !L.kvShared {
					e.Dispatch(pf.pRopeM, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale, r.mropeAxis)
				}
			} else {
				e.Dispatch(pf.pRope, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale)
				if !L.kvShared {
					e.Dispatch(pf.pRope, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale)
				}
			}
			// scatter K,V to cache; a KV-shared layer stores nothing (its cache is its source's)
			if !L.kvShared {
				e.Dispatch(pf.pKv, M*lu.kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, lu.g.uKvDim, lu.uStride, lu.uKOff, lu.uVOff)
			}
			kc, vc := r.kc[l], r.vc[l]
			if L.kvShared && emodelBatchDefect == 1 {
				kc, vc = zeroKV, zeroKV
			}
			// causal attention → ctx (per-layer window: 0 = full causal on a global layer)
			if imgB != (Buffer{}) {
				e.Dispatch(pf.pAttnImg, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, imgB, imgN)
			} else if lu.steel {
				e.Dispatch(pf.pAttnSteel, r.nH*((M+31)/32)*128, 128, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal)
			} else if lu.fused {
				e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal)
			} else {
				e.Dispatch(pf.pAttn, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow)
			}
			if L.qGate { // ctx *= sigmoid(gate), before o-proj, as decode does
				e.Dispatch(pf.pAttnGate, M*nHhd, 256, ctxF, gateF, uGateN)
			}
			// o-proj, then either the plain residual epilogue, Gemma's sandwich norm, or Olmo 3 postOnly: mode-0 write
			// into normF (free scratch here: its last use, the fused-QKV input, already ran, and its next use, the pre-MLP
			// norm's output, is below), norm the sublayer output in place (safe: rmsnorm_f16's read pass completes before
			// its write pass touches the same buffer), then a separate residual add. Other families take the fused mode-2
			// residual epilogue.
			if r.sandwich || r.postOnly {
				gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, normF, uM, uH, lu.uQDim, dummyBias, m0)
				e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postAttnNorm, normF, uH, r.uEps, r.uAddOne)
				if uResidInv != (Buffer{}) {
					e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
				} else {
					e.Dispatch(pf.pRes, M*H, 256, xF, normF)
				}
			} else {
				gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, xF, uM, uH, lu.uQDim, dummyBias, m2)
			}
		}
		if L.moe != nil {
			if r.knobValue("GOINFER_MOE_EXPERT_MAJOR") == "0" {
				// Fallback to row-by-row path for A/B testing
				for m := range M {
					row := xF.At(m * H * 2)
					e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, row, L.postNorm, r.mq, r.mSc, uH, r.uEps, r.uAddOne)
					r.encodeMoERoute(e, L)
					if r.moeCap.idx != (Buffer{}) && m < r.moeCap.rows {
						r.encodeMoECapture(e, l*r.moeCap.rows+m)
					}
					e.Dispatch(pf.pZeroF32, r.H, 256, moeDst)
					r.encodeMoEExperts(e, L, moeDst)
					e.Dispatch(pf.pResF32, r.H, 256, row, moeDst)
				}
				continue
			}

			// --- EXPERT-MAJOR PATH ---
			// 1. Pre-norm all M rows into normF
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, uH, uEpsX, r.uAddOne)

			// 2. Batched Router GEMM: normF [M, H] x routerW [nE, H] -> moeLogits [M, nE]
			totalRouterThreads := (M*r.moe.nE*32 + 255) / 256 * 256
			e.Dispatch(pf.pRouterGemm, totalRouterThreads, 256, normF, L.moe.routerW, moeLogits, uH, r.moe.uNE, uMReal)

			// 3. Batched Top-k Route: moeLogits -> moeIdx [M, k], moeWgt [M, k]
			if r.moe.isGptOss {
				e.Dispatch(pf.pRouteGptOssBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, uMReal)
			} else {
				e.Dispatch(pf.pMoeRouteBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, r.moe.uSigmoid, r.moe.uNorm, r.moe.uScale, r.moe.uNGroup, r.moe.uTopkGroup, uMReal)
			}

			// 4. Commit and wait so the host can read the routing decisions
			e.End()
			if db02Trace != nil {
				db02Trace(db02Event{layer: l, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
			}
			r.recordExecErr(e.Err())
			if err := r.takeExecErr(); err != nil {
				return nil
			}

			// 5. Host Grouping: group (token m, rank j) by expert e
			type moeSlot struct {
				row  int
				rank int
				wgt  float32
			}
			byExpert := make([][]moeSlot, r.moe.nE)
			idxSlice := moeIdx.U32s()[:M*r.moe.k]
			wgtSlice := moeWgt.Floats()[:M*r.moe.k]
			if dg01Mutation != "" {
				dg01Mutate(idxSlice, wgtSlice, M, r.moe.k)
			}
			if r.moeCap.major != nil {
				r.moeCap.major(l, idxSlice, wgtSlice)
			}
			for m := range M {
				for j := 0; j < r.moe.k; j++ {
					eIdx := int(idxSlice[m*r.moe.k+j])
					w := wgtSlice[m*r.moe.k+j]
					if eIdx >= 0 && eIdx < r.moe.nE {
						byExpert[eIdx] = append(byExpert[eIdx], moeSlot{row: m, rank: j, wgt: w})
					}
				}
			}

			// 6. Begin new command encoder for expert dispatches
			e = r.q.Begin()

			// 7. Dispatch each active expert
			moeInter := r.moe.inter
			rowsPerExpertGu := 2 * moeInter
			rowsPerExpertD := H
			wprGu := H / 8
			gprGu := H / 32
			wprD := moeInter / 8
			gprD := moeInter / 32

			rowIdxHost := rowIdxBuf.U32s()
			rowWgtHost := rowWgtBuf.Floats()

			slotOffset := 0
			for eIdx, slots := range byExpert {
				Ce := len(slots)
				if Ce == 0 {
					continue
				}

				for i, s := range slots {
					rowIdxHost[slotOffset+i] = uint32(s.row)
					rowWgtHost[slotOffset+i] = s.wgt
				}

				CePad := (Ce + 7) / 8 * 8
				if db02Trace != nil {
					_, tmGu, _ := pf.gemmTile(CePad, rowsPerExpertGu)
					_, tmD, _ := pf.gemmTile(CePad, rowsPerExpertD)
					db02Trace(db02Event{layer: l, expert: eIdx, rows: Ce, padRows: CePad,
						tileRowsGu: (CePad + tmGu - 1) / tmGu * tmGu, tileRowsD: (CePad + tmD - 1) / tmD * tmD})
				}
				uCe := getU32(Ce)
				uCePad := getU32(CePad)
				uMoeI := getU32(moeInter)
				u2MoeI := getU32(2 * moeInter)

				expIdxBuf := rowIdxBuf.At(slotOffset * 4)
				expWgtBuf := rowWgtBuf.At(slotOffset * 4)

				// A. Gather Ce rows from normF into expertIn (with padding zeroed)
				gatherTotal := (CePad*H + 255) / 256 * 256
				e.Dispatch(pf.pGatherRows, gatherTotal, 256, normF, expertIn, expIdxBuf, uH, uCe, uCePad)

				// B. Expert Gate/Up GEMM
				guWOff := eIdx * rowsPerExpertGu * wprGu * 4
				guSOff := eIdx * rowsPerExpertGu * gprGu * 2
				if r.w8 { // int8 slice 4: H bytes of int8 codes and one f32 scale a row
					guWOff, guSOff = eIdx*rowsPerExpertGu*H, eIdx*rowsPerExpertGu*4
				}
				gemm(e, CePad, rowsPerExpertGu, expertIn, L.moe.expGuW.At(guWOff), L.moe.expGuS.At(guSOff), guF, uCePad, u2MoeI, uH, dummyBias, m0)

				// C. SwiGLU
				e.Dispatch(pf.pSw, CePad*moeInter, 256, guF, dqF, uMoeI, r.uAct)

				// D. Expert Down GEMM
				dOff := eIdx * rowsPerExpertD * wprD * 4
				dSOff := eIdx * rowsPerExpertD * gprD * 2
				if r.w8 { // moeInter bytes of int8 codes and one f32 scale a row
					dOff, dSOff = eIdx*rowsPerExpertD*moeInter, eIdx*rowsPerExpertD*4
				}
				gemm(e, CePad, rowsPerExpertD, dqF, L.moe.expDW.At(dOff), L.moe.expDS.At(dSOff), expertDown, uCePad, uH, uMoeI, dummyBias, m0)

				// E. Scatter-add weighted output into residual xF
				scatterTotal := (Ce*H + 255) / 256 * 256
				e.Dispatch(pf.pScatterAdd, scatterTotal, 256, expertDown, xF, expIdxBuf, expWgtBuf, uH, uCe, uMReal)

				slotOffset += Ce
			}

			// 8. Shared expert (if present)
			if r.moe.sharedInter > 0 && dg01Mutation != "shared-off" {
				shI := r.moe.sharedInter
				u2ShI := getU32(2 * shI)
				uShI := getU32(shI)

				gemm(e, Mpad, 2*shI, normF, L.moe.shGuW, L.moe.shGuS, guF, uM, u2ShI, uH, dummyBias, m0)
				e.Dispatch(pf.pSw, Mpad*shI, 256, guF, dqF, uShI, r.uAct)

				if r.moe.sharedUngated {
					// dst += down directly into xF
					gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, xF, uM, uH, uShI, dummyBias, m2)
				} else {
					// gated variant (Qwen2-MoE): compute sigmoid(gate)*down
					gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, expertDown, uM, uH, uShI, dummyBias, m0)
					// Gate logit for all M rows
					gateLogitsTotal := (M*32 + 255) / 256 * 256
					e.Dispatch(pf.pRouterGemm, gateLogitsTotal, 256, normF, L.moe.shGateW, moeLogits, uH, getU32(1), uMReal)
					// Combine into xF via shared_gate_add_f16
					e.Dispatch(pf.pSharedGateAdd, (M*H+255)/256*256, 256, expertDown, xF, moeLogits, uH, uMReal)
				}
			}
			continue
		}
		// pre-MLP norm. A parallel block (Cohere) has none: its MLP reads the same shared norm of the layer's input that the
		// attention read, which normF still holds (the o-proj added into xF directly), so x ends as x + attn + mlp.
		inFFN := normF
		switch {
		case r.parallelBlock && !prefillParallelDefectForTest:
		case r.parallelBlock: // Part B's planted defect: the MLP fed from the post-attention residual
			e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
		case r.postOnly:
			inFFN = xF
		default:
			e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, uH, uEpsX, r.uAddOne)
		}
		// S9 step 2: this layer's FFN width (a Gemma 4 E-model's KV-shared layers are double-wide; every other family's
		// layers are the model's I).
		fI := I
		if L.ffnI != 0 {
			fI = L.ffnI
		}
		if emodelBatchDefect == 6 && r.layers[0].ffnI != 0 { // G-S9c (6): every layer at the first layer's width
			fI = r.layers[0].ffnI
		}
		uFI, u2FI := uI, u2I
		if fI != I {
			uFI, u2FI = getU32(fI), getU32(2*fI)
		}
		// gate/up
		gemm(e, Mpad, 2*fI, inFFN, L.guW, L.guS, guF, uM, u2FI, uH, dummyBias, m0)
		// swiglu/geglu (G8: r.uAct selects — 0 for every family without FeatGatedGELU)
		e.Dispatch(pf.pSw, M*fI, 256, guF, dqF, uFI, r.uAct)
		// down-proj, same plain-vs-sandwich split as o-proj above.
		if r.sandwich || r.postOnly {
			gemm(e, Mpad, H, dqF, L.dW, L.dS, normF, uM, uH, uFI, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postMLPNorm, normF, uH, r.uEps, r.uAddOne)
			if uResidInv != (Buffer{}) {
				e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
			} else {
				e.Dispatch(pf.pRes, M*H, 256, xF, normF)
			}
		} else {
			gemm(e, Mpad, H, dqF, L.dW, L.dS, xF, uM, uH, uFI, dummyBias, m2)
		}
		// S9 step 2: a Gemma 4 E-model's PLE block, then its layer scalar, in decode's order (encodeLayerResidualWith):
		// the gate GEMM (H→P) on the raw residual, gelu(gate) times the row's own per-layer input, the projection GEMM
		// (P→H), the post-PLE norm and the residual add; then h *= layer_scalar.
		if L.pleGW != (Buffer{}) && emodelBatchDefect != 2 {
			P := r.pleP
			gemm(e, Mpad, P, xF, L.pleGW, L.pleGS, pleGF, uM, uP, uH, dummyBias, m0)
			e.Dispatch(pf.pPLEGeluMulF16, M*P, 256, pleGF, pleInB, uP, uLP, getU32(l*P))
			gemm(e, Mpad, H, pleGF, L.plePW, L.plePS, normF, uM, uH, uP, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postPLENorm, normF, uH, r.uEps, r.uAddOne)
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		}
		if L.uLayerScalar != (Buffer{}) && emodelBatchDefect != 4 {
			e.Dispatch(pf.pLayerScaleF16, M*H, 256, xF, L.uLayerScalar)
		}
	}
	addDeep(r.nL - 1)
	// Final norm and LM head for the last token only, through the same int8-pinned head the decode path runs
	// (rmsnorm to int8, then gemv_w8a8): the head weights are int8 (logit-critical), and the int4 gemm_w4f16 would
	// misread them as packed nibbles and f16 scales, producing NaN logits. Norm-quant the last token's f16
	// residual row to int8, then run the decode head.
	finalNormQ := pf.pRmsQ
	if r.layerNorm {
		finalNormQ = pf.pLNQ
	}
	e.Dispatch(finalNormQ, tgReduceNorm, tgReduceNorm, xF.At((M-1)*H*2), r.finalNorm, r.aq, r.aSc, uH, uEpsX, r.uAddOne)
	e.Dispatch(r.pGemvW8, V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	// The int8 head turns a NaN row into finite zeros, which the caller's non-finite-logit check (A-C02) passes, so the
	// residual itself is checked. Its last row is enough: a row that overflows before the final layer writes NaN K/V,
	// which reaches the last row through attention; a row that overflows in the final layer writes no further K/V.
	last := xF.U16s()[(M-1)*H : M*H]
	if r.poisonPrefillResidForTest {
		last[0] = 0x7c00 // +inf
	}
	r.prefillResidNonFinite = false
	for _, h := range last {
		if h&0x7c00 == 0x7c00 {
			r.prefillResidNonFinite = true
			break
		}
	}
	if db02Trace != nil {
		db02Trace(db02Event{layer: r.nL, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
	}
	r.recordExecErr(e.Err()) // C-09

	out := make([]float32, V)
	copy(out, r.logits.Floats()[:V])
	// Gemma's final-logit softcap: finalizeLogits (the decode path's entry point) applies it to r.logitsHost, but
	// this pass copies straight out of r.logits into a fresh slice and returns before finalizeLogits runs, so it
	// is applied here too. softcapParallel is a no-op for a non-softcapped family (0).
	if r.finalSoftcap > 0 {
		softcapParallel(out, r.finalSoftcap)
	}
	// Cohere's logit scale (FeatLogitScale), as finalizeLogits applies it for decode.
	if r.logitScale != 0 && r.logitScale != 1 {
		for i, v := range out {
			out[i] = v * r.logitScale
		}
	}
	return out
}

// prefillScratchU16 allocates one of a pass's f16 scratch buffers, n halves, zero-filled. Metal fills a new buffer
// with zeros itself (newBufferWithLength), so the pass does not build a zeroed Go slice of the same size and copy it
// in (prefillScratchCopy restores the copy, for the test that compares the two).
// THIS RELIES ON THE ZERO-FILL, which is a Metal property only: aikit's gpu.NewBufferLen* contract says Metal returns
// zeroed memory and CUDA uninitialized memory, and code shared across backends must not rely on zeros. These buffers
// are sized to Mpad rows and the pass writes M of them, so the pad rows hold the zeros Metal gave them. A CUDA port of
// this pattern must zero the buffer explicitly (Queue.ZeroAsync). TestPrefillScratch_zeroFilled pins the property on
// this backend.
func prefillScratchU16(d *Device, n int) Buffer {
	if prefillScratchCopy {
		return NewBufferU16s(d, make([]uint16, n))
	}
	return gpu.NewBufferLenOf[uint16](d, n)
}

// prefillScratch is prefillScratchU16 for any element type: the f32 and u32 buffers that the router GEMM, the route
// kernels or the host fill before they are read (the expert-major MoE branch's moeLogits, moeIdx, moeWgt, rowIdxBuf,
// rowWgtBuf), and xF, which the embeddings are converted straight into. Same zero-fill reliance and the same copy arm,
// plus prefillR20Copy, which copies these alone so a test can time them apart from the other scratch.
func prefillScratch[T gpu.Scalar](d *Device, n int) Buffer {
	if prefillScratchCopy || prefillR20Copy {
		return gpu.NewBufferOf(d, make([]T, n))
	}
	return gpu.NewBufferLenOf[T](d, n)
}

var prefillScratchCopy = false

// prefillR20Copy restores the copy for R-20's sites only (xF and the five MoE routing buffers), for the A/B that times
// them; tests only.
var prefillR20Copy = false

// prefillScratchPoison fills the expert-major branch's five routing buffers with 0xFF bytes (NaN as f32, an
// out-of-range expert id as u32) before the pass, set only by TestPrefillScratch_moeRoutingWrittenFirst: equal logits
// then prove the router GEMM, the route kernels and the host write every element before anything reads it, so the
// buffers need no zeros at all (R-20).
var prefillScratchPoison = false

// db02Trace is D-B02's probe (T1.12, metal/audit_db02_test.go): nil in production. The expert-major branch reports
// each command buffer it ends (gpuStart/gpuEnd, seconds; layer is the MoE layer whose routing it waits for, or nL for
// the pass's last buffer) and each active expert's rows (real, padded to 8, and padded to each GEMM's row tile).
var db02Trace func(db02Event)

type db02Event struct {
	layer, expert                        int
	gpuStart, gpuEnd                     float64
	rows, padRows, tileRowsGu, tileRowsD int
}

// dg01Mutation names a deliberate defect in the expert-major MoE prefill, for D-G01's gate to prove it can go red
// (metal/dg01_moe_gate_test.go). "" in production; set only by that test.
//   - "topk-renorm": the routing weights renormalised to sum to 1 per token, as a model with norm_topk_prob would
//     have them (Qwen1.5-MoE does not): the scatter weights wrong.
//   - "rank-swap": each token's first two experts swapped while their weights stay put, so each weight lands on the
//     other expert: the route's order wrong.
//   - "shared-off": the shared expert skipped.
var dg01Mutation string

func dg01Mutate(idx []uint32, wgt []float32, M, k int) {
	for m := range M {
		row, w := idx[m*k:(m+1)*k], wgt[m*k:(m+1)*k]
		switch dg01Mutation {
		case "topk-renorm":
			var sum float32
			for _, x := range w {
				sum += x
			}
			if sum > 0 {
				for j := range w {
					w[j] /= sum
				}
			}
		case "rank-swap":
			if k >= 2 {
				row[0], row[1] = row[1], row[0]
			}
		}
	}
}
