package kernels

// PrefillKernels is the f16 simdgroup_matrix (MMA) prefill library, for fast prompt ingestion: an MMA GEMM reuses each
// weight across all M prompt rows, which the int4 scalar-MAC decode path cannot. Activations flow in f16 (no int8
// quant); weights stay int4 and are dequantized to f16 in-kernel. A separate library from allKernels, so the decode
// path is unchanged; ensurePrefill compiles it lazily.
const PrefillKernels = `
#include <metal_stdlib>
using namespace metal;
` + AttnScoreMSL + `
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

inline float glu_act_f16(float x, uint act); // defined with swiglu_f16, below

// gemm_w4f16_swiglu (KS-D5) is gemm_w4f16_store over the gate|up projection followed by swiglu_f16, in one pass: threadgroup x
// stages the 32 gate rows [32x, 32x+32) as features 0..31 and the matching up rows as 32..63, the same dequant and MMA chain,
// then rounds the tile to half as guF held it and writes glu_act_f16(g) * u for its 32 pairs, C = [M x I]. The same values in the
// same order as the two-step form, so the output is the same bits, and guF is never written. grid = (ceil(I/32), ceil(M/64)).
kernel void gemm_w4f16_swiglu(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& I[[buffer(5)]], constant uint& K[[buffer(6)]],
    constant uint& act[[buffer(7)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    threadgroup half sa[32*72];   // weights:     [feature block 0..7][k block 0..3] of 8x8 [k][feature], padded
    threadgroup half sb[64*32];   // activations: [token block 0..7][k block 0..3] of 8x8 [token][k]
    threadgroup float cs[4*64];   // per-simdgroup epilogue scratch
    threadgroup half gu[64*64];   // the tile rounded to half, as guF held it: [token][gate 0..31 | up 0..31]

    const uint i0 = tgp.x*32u, m0 = tgp.y*64u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;
    simdgroup_float8x8 acc[16];
    for (ushort i = 0; i < 16; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid >> 1, kh = tid & 1;     // weight row 0..63, k half (two 8-wide k blocks)
    const ushort fb = fr >> 3, nl = fr & 7;
    const bool wok = i0 + (fr & 31u) < I;
    const uint ncol = fr < 32 ? i0 + fr : I + i0 + (fr - 32u); // features 0..31 the gate rows, 32..63 the matching up rows
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
        for (ushort f = 0; f < 4; f++) {
            simdgroup_store(acc[t*4 + f], myc, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort e = lane; e < 64; e += 32)
                gu[((th*4 + t)*8 + e/8)*64 + (fh*4 + f)*8 + e%8] = half(myc[e]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort e = tid; e < 64*32; e += 128) {
        const ushort r = e / 32, j = e % 32;
        const uint m = m0 + r, i = i0 + j;
        if (m < M && i < I) {
            float g = float(gu[r*64 + j]), u = float(gu[r*64 + 32 + j]);
            C[m*I + i] = half(glu_act_f16(g, act)*u);
        }
    }
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
// addOne selects Gemma's (1+w) RMS offset vs plain w — mirrors decoder's rmsnorm.go / the decode
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
// PAIRWISE rotation, pair dd = dims (2dd, 2dd+1), as decoder's rope.go's applyRoPEInterleaved and applyMRoPEPairwise (GLM-OCR's
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
        sc[s]=attn_score(a*scale);
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

// attention_prefill_tiled: attention_prefill for a pass past its 4,096-score buffer (P28). The scores go through sc[] one tile of
// keys at a time, in two passes: the row's max over every key, then per tile the probabilities, each thread's share of
// the softmax sum and the P*V accumulation. Each thread owns the same keys in the same order as attention_prefill's, the sum
// is reduced by the same tree, and P*V walks the keys in order, so the output equals attention_prefill's wherever both run;
// a score is computed twice (once for the max, once per tile) by the same expression. A thread starts each tile at the first
// key it owns there, so any tile size keeps that ownership. Needs hd <= 4 * the threadgroup size (out-dim accumulators in
// registers).
inline float prefill_score(device const half* qr, device const half* k, uint hd, float scale) {
    float a=0; uint d=0;
    if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=float(qr[d])*float(k4.x); a+=float(qr[d+1u])*float(k4.y); a+=float(qr[d+2u])*float(k4.z); a+=float(qr[d+3u])*float(k4.w); }
    for (; d<hd; d++) a += float(qr[d])*float(k[d]);
    return attn_score(a*scale);
}
kernel void attention_prefill_tiled(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& tile[[buffer(11)]],
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
    float mmax=-INFINITY;
    for (uint s=winStart+tid; s<nKeys; s+=tgs) mmax=max(mmax, prefill_score(qr, kb+s*kvDim, hd, scale));
    mmax = simd_max(mmax);
    uint sgidA = tid >> 5u, laneA = tid & 31u;
    if (laneA == 0) red[sgidA] = mmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgA = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsgA; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0;
    float acc[4] = {0, 0, 0, 0};
    for (uint t0=winStart; t0<nKeys; t0+=tile) {
        uint t1 = min(t0+tile, nKeys);
        uint own = t0 + (tid + tgs - (t0-winStart)%tgs) % tgs; // this thread's first key in the tile: s - winStart == tid mod tgs
        for (uint s=own; s<t1; s+=tgs){ float p=exp(prefill_score(qr, kb+s*kvDim, hd, scale)-mx); sc[s-t0]=p; ls+=p; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0u, d=tid; d<hd; j++, d+=tgs){ float a=acc[j]; for(uint s=t0;s<t1;s++) a += sc[s-t0]*float(vb[s*kvDim+d]); acc[j]=a; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=0u, d=tid; d<hd; j++, d+=tgs) out[m*qDim + qh*hd + d]=half(acc[j]/sum);
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
        sc[s]=attn_score(a*scale);
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
                    float v = attn_score(sScr[sgid][row*ATTN_KTILE+c]*scale);
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
            s[0] = attn_score_log2(S[0].thread_elements()[0] * sl2); s[1] = attn_score_log2(S[0].thread_elements()[1] * sl2);
            s[2] = attn_score_log2(S[1].thread_elements()[0] * sl2); s[3] = attn_score_log2(S[1].thread_elements()[1] * sl2);
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

// route_gptoss_batch: batched router for gpt-oss MoE. A slot with no score above -INFINITY takes the lowest expert not
// yet chosen in its row, as route_gptoss does (moe.go), so no index outside [0, nE) is written.
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
        if (best < 0) {
            best = 0;
            for (uint i=0u;i<nE;i++) if (!taken[i]) { best=int(i); break; }
            bv = -1e30f;
        }
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
