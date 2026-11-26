// gemm_w4f16_store: blocked int4→f16 MMA GEMM with a fused epilogue (mode 0 = plain store, 1 =
// +bias for fused QKV, 2 = +residual for o-proj/down) — C[M×N] = A[M×K](f16) · Wᵀ, W = resident
// int4/W4A8 (packed nibbles + f16 group scales), dequanted in-kernel. Dispatched as a 2-D grid of
// threadgroups, (ceil(N/64), ceil(M/64)) × 128 threads — see gemmGrid in PrefillLast.
//
// R16 (docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): restructured
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
//
// The nibble dequant takes two weights (nibbles kl and kl+4) per half2 operation: OR-ing a nibble pair into 0x6400's mantissa
// gives 1024+n exactly, minus 1032 gives n-8 exactly, and the half multiply by the (half) scale is the same single rounding as
// half(float(n-8) * sc). Equal in every bit to the per-weight form for all 16 nibbles x 65,536 scales
// (TestKSD7_half2DequantExhaustive); KSD7_OLD_DEQUANT compiles the per-weight form back for the A/B.
inline half2 gemm_dq2(uint word, ushort kl, half2 hs) {
    return (as_type<half2>(((word >> (4u*kl)) & 0x000F000Fu) | 0x64006400u) - half2(1032.0h)) * hs;
}

inline void gemm_stage_w(threadgroup half* sa, uint2 wv, float sc, bool wok, ushort fb, ushort kh, ushort nl) {
    for (ushort w = 0; w < 2; w++) {
        uint word = wv[w];
        threadgroup half* p = sa + (fb*4 + kh*2 + w)*72 + nl;
#ifdef KSD7_OLD_DEQUANT
        for (ushort kl = 0; kl < 8; kl++) {
            *p = wok ? half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc) : 0.0h;
            p += 8;
        }
#else
        const half2 hs = half2(half(sc));
        for (ushort kl = 0; kl < 4; kl++) {
            const half2 v = wok ? gemm_dq2(word, kl, hs) : half2(0.0h);
            p[kl*8] = v.x;
            p[(kl+4)*8] = v.y;
        }
#endif
    }
}

// gemm_epilogue_ld stores one 8x8 accumulator into columns [coff, coff+N) of C, whose rows are ldc halves apart; the bias is
// indexed by the column in C. gemm_epilogue is the whole-width case (ldc = N, coff = 0).
inline void gemm_epilogue_ld(threadgroup float* myc, simdgroup_float8x8 acc, uint mb, uint nb, uint M, uint N, uint ldc,
    uint coff, device half* C, device const float* bias, uint mode, ushort lane) {
    simdgroup_store(acc, myc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort e = lane; e < 64; e += 32) {
        const uint m = mb + e/8, n = nb + e%8;
        if (m < M && n < N) {
            const uint o = m*ldc + coff + n;
            float v = myc[e];
            if (mode == 1u) v += bias[coff + n];          // fused bias (QKV)
            if (mode == 2u) v += float(C[o]);             // residual (o / down)
            C[o] = half(v);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

inline void gemm_epilogue(threadgroup float* myc, simdgroup_float8x8 acc, uint mb, uint nb, uint M, uint N,
    device half* C, device const float* bias, uint mode, ushort lane) {
    gemm_epilogue_ld(myc, acc, mb, nb, M, N, N, 0u, C, bias, mode, lane);
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
// W8 (KS-D5c, gemm_w8f16_swiglu) stages the native int8 path's weights as gemm_w4f16_tile<64, 64, true> does, so it is
// gemm_w8f16_m64n64 + swiglu_f16 bit for bit.
// w8Block is one 8-k block of a native int8 weight row, the codes at byte offset off (a multiple of 4): zero when the row is
// out (ok false) or the block starts at or past K, and its upper word zero when only four of its codes are inside K. A native
// int8 row is any multiple of 4 bytes, so a 32-wide slab can end inside a row's last block or past it; two word loads never
// read past the row.
inline uint2 w8Block(device const uint* W, uint off, uint kk, uint K, bool ok) {
    if (!ok || kk >= K) return uint2(0u);
    return uint2(W[off/4u], kk + 8u <= K ? W[off/4u + 1u] : 0u);
}

template <bool W8>
kernel void gemm_swiglu(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& I[[buffer(5)]], constant uint& K[[buffer(6)]],
    constant uint& act[[buffer(7)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    // The staging arrays and, once the K loop is done with them, the half tile share one array: the tile (64 x 64 halves) fits
    // in sa then sb (32*72 + 64*32 halves), so the kernel needs no more threadgroup memory than gemm_w4f16_store.
    threadgroup half sab[32*72 + 64*32];
    threadgroup half* sa = sab;          // weights:     [feature block 0..7][k block 0..3] of 8x8 [k][feature], padded
    threadgroup half* sb = sab + 32*72;  // activations: [token block 0..7][k block 0..3] of 8x8 [token][k]
    threadgroup float cs[4*64];          // per-simdgroup epilogue scratch
    threadgroup half* gu = sab;          // after the K loop: the tile rounded to half, as guF held it: [token][gate 0..31 | up 0..31]

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
        if constexpr (W8) { // K bytes of int8 codes a row and one f32 scale a row; an 8-k block is two words
            const float sc = wok ? reinterpret_cast<device const float*>(WS)[ncol] : 0.0f;
            for (ushort w = 0; w < 2; w++) {
                const ushort kb = kh*2 + w;
                const uint2 wd = w8Block(W, ncol*K + k0 + kb*8u, k0 + kb*8u, K, wok);
                const char4 c0 = as_type<char4>(wd.x), c1 = as_type<char4>(wd.y);
                threadgroup half* p = sa + (fb*4 + kb)*72 + nl;
                for (ushort kl = 0; kl < 8; kl++) {
                    *p = wok ? half(float(kl < 4 ? c0[kl] : c1[kl - 4]) * sc) : 0.0h;
                    p += 8;
                }
            }
        } else {
            uint2 wv = uint2(0u); float sc = 0.0f;
            if (wok) { wv = *(device const uint2*)(W + ncol*wpr + k0/8u + kh*2u); sc = float(WS[ncol*gpr + k0/32u]); }
            gemm_stage_w(sa, wv, sc, wok, fb, kh, nl);
        }
        for (ushort j = 0; j < 2; j++) {
            const ushort kb = kh2*2 + j;
            threadgroup half4* d4 = (threadgroup half4*)(sb + (tb*4 + kb)*64 + rl*8);
            const uint kk = k0 + kb*8u;
            if (aok && (!W8 || kk < K)) {
                device const half4* s4 = (device const half4*)(A + mrow*K + kk);
                d4[0] = s4[0]; d4[1] = (!W8 || kk + 8u <= K) ? s4[1] : half4(0.0h);
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

    threadgroup_barrier(mem_flags::mem_threadgroup); // every simdgroup's last MMA has read sa and sb before the tile overwrites them
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
template [[host_name("gemm_w4f16_swiglu")]] kernel decltype(gemm_swiglu<false>) gemm_swiglu<false>;
template [[host_name("gemm_w8f16_swiglu")]] kernel decltype(gemm_swiglu<true>) gemm_swiglu<true>;

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
            // int8 slice 2: W is the native int8 path's row-major int8 codes,
            // K bytes a row, and WS its per-row f32 scales; an 8-k block is two words.
            float sc = wok ? reinterpret_cast<device const float*>(WS)[ncol] : 0.0f;
            for (ushort w = 0; w < WPT; w++) {
                const ushort kb = kq*WPT + w;
                const uint2 wd = w8Block(W, ncol*K + k0 + kb*8u, k0 + kb*8u, K, wok);
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
#ifdef KSD7_OLD_DEQUANT
                for (ushort kl = 0; kl < 8; kl++) {
                    *p = wok ? half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc) : 0.0h;
                    p += 8;
                }
#else
                const half2 hs = half2(half(sc));
                for (ushort kl = 0; kl < 4; kl++) {
                    const half2 v = wok ? gemm_dq2(word, kl, hs) : half2(0.0h);
                    p[kl*8] = v.x;
                    p[(kl+4)*8] = v.y;
                }
#endif
            }
        }
        for (ushort j = 0; j < BPT && (TM >= 32 || tr < TM); j++) {
            const ushort kb = kq2*BPT + j;
            threadgroup half4* d4 = (threadgroup half4*)(sb + (tb*4 + kb)*64 + rl*8);
            const uint kk = k0 + kb*8u;
            if (aok && (!W8 || kk < K)) {
                device const half4* s4 = (device const half4*)(A + mrow*K + kk);
                d4[0] = s4[0]; d4[1] = (!W8 || kk + 8u <= K) ? s4[1] : half4(0.0h);
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

