// decode_fa.cu — CUDA flash-decode lane (docs/tasks/red-october.md R6 step 2). NOT bit-identical to the
// shipped decode attention, opt-in (GOINFER_CUDA_FLASH_DECODE=S), and fidelity-gated by
// docs/measurements/attn-decode-fa-fidelity-PREREGISTERED.md.
//
// WHY. The exact split-KV path is three launches (scores, softmax, V-sum) that materialise the score row in
// global memory and keep every reduction whole and in order so the result is byte-identical to attn_batched.
// That costs ~447 us/layer on D7 at 8000 against a ~75 us traffic floor (4 kv heads x 8000 keys x 128 x 4 B x
// K and V = 32.8 MB at 448 GB/s); ncu says the V-sum is latency-bound at 9% occupancy and the scores kernel is
// issue-throttled by redundant q loads (docs/measurements/splitkv-stall-profile-2026-09-13.md). A flash-decode
// kernel drops the bit-identity constraint and with it the score row, the separate softmax, and the
// per-output-dim single-thread fold.
//
// SHAPE. One CTA (8 warps) per (kv head, key split); every query head of the GQA group is handled by the SAME
// CTA so each K/V row is read from DRAM once per group, not once per head. Keys are processed in blocks of 8:
//   scores  : 4-lane groups, one key per group (32 lanes = 8 keys). Lane `sub` reads float4 #(j*4+sub) of the
//             key row for j < HD/16, so the 4 lanes of a group read 64 contiguous bytes per step (full sectors),
//             dots it against the group's staged q, and the 4 lanes finish with 2 shuffles.
//   softmax : ONLINE, per head, warp-wide over the 8 keys of the block (3 shuffles for the max, 3 for the sum);
//             the running (m, l, acc) live in registers and are rescaled once per block, warp-uniformly.
//   P.V     : all 32 lanes on one key at a time, each lane owning HD/32 consecutive-by-float4 output dims, so the
//             V row is one coalesced read and there are no shuffles; the probabilities come from shared memory.
// Each WARP finishes with an unnormalised partial (O[HD], m, l) per head; a second kernel merges the
// nSplit*8 partials of a head in FIXED ASCENDING unit order (never atomics), so the result is deterministic
// run to run — the one property the lane must keep while giving up bit-identity to history.
//
// PARTITION. Split sp owns keys [winStart + sp*per, ...) with per = ceil(span/nSplit), a function of
// (nKeys, winStart, nSplit) only. Inside a split, warp w takes 8-key blocks w, w+8, w+16, ... so the 8 warps of
// a CTA sweep neighbouring memory together. Unit u = sp*8 + w indexes the partial buffer.
//
// SCOPE. hd in {64,128,256}, GQA group nH/nKV <= 8, no attention sink (the caller declines gpt-oss). f32 KV.
// Exponentials are __expf, as in splitkv_softmax. `#pragma` FMA: the dots use explicit __fmaf_rn, and
// kernel_fma_lint_test.go checks no other float multiply-add pair is left to the compiler.
#define FA_WARPS 8
#define FA_NEG (-1e30f)
#define FA_SEEN (-5e29f) // a unit with any key has m > FA_SEEN

template <int HD>
__device__ __forceinline__ void fa_partial_impl(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    constexpr int VEC = (HD >= 128) ? 4 : 2;      // output dims per load per lane
    constexpr int NCH = HD / (32 * VEC);          // loads per lane per V row (1 for 64 and 128, 2 for 256)
    constexpr int NQ = HD / 16;                   // float4 steps of the 4-lane score dot
    const int kvh = blockIdx.x, sp = blockIdx.y;
    if (kvh >= nKV || sp >= nSplit) return;
    const int G = nH / nKV;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = lane >> 2, sub = lane & 3;
    const int kvDim = nKV * HD;

    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                               // [G][HD]
    float* pb = smem + G * HD + warp * 64;          // this warp's [8 keys][8 heads] probabilities
    for (int t = threadIdx.x; t < G * HD; t += blockDim.x) qs[t] = q[(long)(kvh * G) * HD + t];
    __syncthreads();

    const int span = nKeys - winStart;
    const int per = (span + nSplit - 1) / nSplit;
    const int lo = winStart + sp * per;
    int hi = lo + per; if (hi > nKeys) hi = nKeys;

    float m[8], l[8], acc[8][NCH * VEC];
#pragma unroll
    for (int g = 0; g < 8; g++) {
        m[g] = FA_NEG; l[g] = 0.f;
#pragma unroll
        for (int c = 0; c < NCH * VEC; c++) acc[g][c] = 0.f;
    }

    for (int b0 = lo + warp * 8; b0 < hi; b0 += FA_WARPS * 8) {
        // ---- scores: one key per 4-lane group ----
        const int key = b0 + grp;
        const bool valid = key < hi;
        float part[8];
#pragma unroll
        for (int g = 0; g < 8; g++) part[g] = 0.f;
        if (valid) {
            const float4* k4 = (const float4*)(kc + (long)key * kvDim + (long)kvh * HD);
#pragma unroll
            for (int j = 0; j < NQ; j++) {
                const float4 kk = k4[j * 4 + sub];
#pragma unroll
                for (int g = 0; g < 8; g++) {
                    if (g < G) {
                        const float4 qq = ((const float4*)qs)[g * (HD / 4) + j * 4 + sub];
                        part[g] = __fmaf_rn(qq.x, kk.x, part[g]);
                        part[g] = __fmaf_rn(qq.y, kk.y, part[g]);
                        part[g] = __fmaf_rn(qq.z, kk.z, part[g]);
                        part[g] = __fmaf_rn(qq.w, kk.w, part[g]);
                    }
                }
            }
        }
        float s[8], alpha[8], p[8];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            float d = part[g];
            d += __shfl_xor_sync(0xffffffff, d, 1);
            d += __shfl_xor_sync(0xffffffff, d, 2);
            s[g] = valid ? d * scale : FA_NEG;
        }
        // ---- online softmax over the block's 8 keys, per head, warp-uniform ----
#pragma unroll
        for (int g = 0; g < 8; g++) {
            if (g < G) {
                float mx = s[g];
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16));
                const float mn = fmaxf(m[g], mx);
                alpha[g] = __expf(m[g] - mn);
                p[g] = valid ? __expf(s[g] - mn) : 0.f;
                float ps = p[g];
                ps += __shfl_xor_sync(0xffffffff, ps, 4);
                ps += __shfl_xor_sync(0xffffffff, ps, 8);
                ps += __shfl_xor_sync(0xffffffff, ps, 16);
                l[g] = __fmaf_rn(l[g], alpha[g], ps);
                m[g] = mn;
                if (sub == 0) pb[grp * 8 + g] = p[g];
            }
        }
        __syncwarp();
        // ---- P.V: all lanes on one key at a time ----
#pragma unroll
        for (int g = 0; g < 8; g++) {
            if (g < G) {
#pragma unroll
                for (int c = 0; c < NCH * VEC; c++) acc[g][c] *= alpha[g];
            }
        }
        const int nb = min(8, hi - b0);
        for (int kk = 0; kk < nb; kk++) {
            const float* vp = vc + (long)(b0 + kk) * kvDim + (long)kvh * HD;
            float v[NCH * VEC];
#pragma unroll
            for (int c = 0; c < NCH; c++) {
                if (VEC == 4) {
                    const float4 t = ((const float4*)vp)[c * 32 + lane];
                    v[c * 4 + 0] = t.x; v[c * 4 + 1] = t.y; v[c * 4 + 2] = t.z; v[c * 4 + 3] = t.w;
                } else {
                    const float2 t = ((const float2*)vp)[lane];
                    v[0] = t.x; v[1] = t.y;
                }
            }
#pragma unroll
            for (int g = 0; g < 8; g++) {
                if (g < G) {
                    const float pv = pb[kk * 8 + g];
#pragma unroll
                    for (int c = 0; c < NCH * VEC; c++) acc[g][c] = __fmaf_rn(pv, v[c], acc[g][c]);
                }
            }
        }
        __syncwarp();
    }

    // ---- unnormalised per-warp partial: O[HD], m, l ----
    const int U = nSplit * FA_WARPS, u = sp * FA_WARPS + warp;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        if (g < G) {
            float* out = partial + ((long)(kvh * G + g) * U + u) * (HD + 4);
#pragma unroll
            for (int c = 0; c < NCH; c++) {
                if (VEC == 4) {
                    ((float4*)out)[c * 32 + lane] = make_float4(acc[g][c * 4], acc[g][c * 4 + 1], acc[g][c * 4 + 2], acc[g][c * 4 + 3]);
                } else {
                    ((float2*)out)[lane] = make_float2(acc[g][0], acc[g][1]);
                }
            }
            if (lane == 0) { out[HD] = m[g]; out[HD + 1] = l[g]; }
        }
    }
}

extern "C" __global__ void fa_partial_64(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<64>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}
extern "C" __global__ void fa_partial_128(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<128>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}
extern "C" __global__ void fa_partial_256(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<256>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}

// fa_combine: grid nH, block hd, dynamic shared memory 3*U floats. Merges the U = nSplit*8 unit partials of a head
// in FIXED ascending order (no atomics, so the result is deterministic). The per-unit (m, l) are loaded once into
// shared memory, the global max M and the weights w_u = exp(m_u - M) are computed once per head instead of once per
// output dim, and the main loop then carries no load-dependent branch, so its V-partial loads are independent and
// pipeline. A unit with no keys carries m = FA_NEG, l = 0 and contributes nothing (w_u = 0); at least one unit
// always saw a key (the caller guarantees nKeys > winStart).
extern "C" __global__ void fa_combine(const float* __restrict__ partial, int hd, int U, float* __restrict__ ctx) {
    extern __shared__ float cs[];
    float* sm = cs; float* sw = cs + U; float* sl = cs + 2 * U;
    const int h = blockIdx.x, d = threadIdx.x;
    const int stride = hd + 4;
    const float* base = partial + (long)h * U * stride;
    for (int u = threadIdx.x; u < U; u += blockDim.x) { sm[u] = base[(long)u * stride + hd]; sl[u] = base[(long)u * stride + hd + 1]; }
    __syncthreads();
    float M = FA_NEG;
    for (int u = 0; u < U; u++) M = fmaxf(M, sm[u]);
    for (int u = threadIdx.x; u < U; u += blockDim.x) sw[u] = (sm[u] > FA_SEEN) ? __expf(sm[u] - M) : 0.f;
    __syncthreads();
    if (d >= hd) return;
    float num = 0.f, den = 0.f;
#pragma unroll 8
    for (int u = 0; u < U; u++) {
        const float w = sw[u];
        num = __fmaf_rn(w, base[(long)u * stride + d], num);
        den = __fmaf_rn(w, sl[u], den);
    }
    ctx[(long)h * hd + d] = num / den;
}

// ============================================================================================================
// MULTI-ROW variant for speculative verify (docs/measurements/attn-decode-fa-verify-PREREGISTERED.md).
//
// A verify batch is R consecutive query rows at consecutive positions. fa_partial_rows handles R of them per CTA with
// ONE pass over K and V (each K/V row loaded once, used for all R rows), and produces, per row, the SAME unnormalised
// per-warp partials fa_partial produces for that row's position — bit for bit — because per row every operation is the
// M=1 kernel's, in the M=1 kernel's order:
//   * the same blocks: warp w takes the 8-key blocks lo + 8w, lo + 8w + 64, ...; rows of a launch share (lo, per) (the host
//     groups consecutive rows with equal partition), and differ only in how far the last chunk extends;
//   * a key at or past a row's own hi is MASKED for that row (score -1e30, p = 0) and its P.V step is SKIPPED, so a row sees
//     exactly the keys the M=1 kernel would visit; a block wholly past the row's hi leaves (m, l, acc) untouched:
//     alpha = __expf(m - max(m, -1e30)) = __expf(0) = 1 and l = fmaf(l, 1, 0) = l exactly, acc *= 1 exactly;
//   * the dot, the 2+3+3 shuffle ladders, __expf, and the FMAs are the same expressions.
// The M=1 kernels above are NOT edited; fa_combine_rows is fa_combine with a row index.
//
// R (rows per CTA) is chosen so the R*G*(HD/32) accumulators stay near 64 registers; the host mirrors faRowsPerCTA().
#define FA_ACC_BUDGET 64
template <int HD, int G>
struct FaRows {
    static constexpr int VEC = (HD >= 128) ? 4 : 2;
    static constexpr int DIMS = (HD / 32);                       // output dims per lane
    static constexpr int RAW = FA_ACC_BUDGET / (DIMS * G);
    static constexpr int R = RAW < 1 ? 1 : (RAW > 8 ? 8 : RAW);
};

template <int HD, int G>
__device__ __forceinline__ void fa_partial_rows_impl(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys0, int per, float scale, int nSplit, int nRows, int row0,
    float* __restrict__ partial) {
    constexpr int R = FaRows<HD, G>::R;
    constexpr int VEC = FaRows<HD, G>::VEC;
    constexpr int NCH = HD / (32 * VEC);
    constexpr int NQ = HD / 16;
    constexpr int ND = NCH * VEC;
    const int kvh = blockIdx.x, sp = blockIdx.y;
    if (kvh >= nKV || sp >= nSplit) return;
    const int rb = blockIdx.z * R;                 // first run-row of this CTA
    if (rb >= nRows) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = lane >> 2, sub = lane & 3;
    const int kvDim = nKV * HD;

    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                                // [R][G][HD]
    float* pb = smem + R * G * HD + warp * (R * 64); // this warp's [R][8 keys][8 heads]
    for (int t = threadIdx.x; t < R * G * HD; t += blockDim.x) {
        const int rr = t / (G * HD), rem = t % (G * HD);
        const int row = rb + rr;
        qs[t] = (row < nRows) ? q[(long)(row0 + row) * nH * HD + (long)(kvh * G) * HD + rem] : 0.f;
    }
    __syncthreads();

    const int lo = winStart + sp * per;
    int hiRow[R];                                     // each row's own end of chunk, as the M=1 kernel computes it
    int hiMax = lo;
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
        const int row = rb + rr;
        int nk = nKeys0 + row;
        int h = lo + per;
        if (h > nk) h = nk;
        hiRow[rr] = (row < nRows) ? h : lo;           // inactive padding rows see no keys
        if (hiRow[rr] > hiMax) hiMax = hiRow[rr];
    }

    float m[R][G], l[R][G], acc[R][G][ND];
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
#pragma unroll
        for (int g = 0; g < G; g++) {
            m[rr][g] = FA_NEG; l[rr][g] = 0.f;
#pragma unroll
            for (int c = 0; c < ND; c++) acc[rr][g][c] = 0.f;
        }
    }

    for (int b0 = lo + warp * 8; b0 < hiMax; b0 += FA_WARPS * 8) {
        const int key = b0 + grp;
        const bool inMax = key < hiMax;
        float part[R][G];
#pragma unroll
        for (int rr = 0; rr < R; rr++) {
#pragma unroll
            for (int g = 0; g < G; g++) part[rr][g] = 0.f;
        }
        if (inMax) {
            const float4* k4 = (const float4*)(kc + (long)key * kvDim + (long)kvh * HD);
#pragma unroll
            for (int j = 0; j < NQ; j++) {
                const float4 kk = k4[j * 4 + sub];
#pragma unroll
                for (int rr = 0; rr < R; rr++) {
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const float4 qq = ((const float4*)qs)[(rr * G + g) * (HD / 4) + j * 4 + sub];
                        part[rr][g] = __fmaf_rn(qq.x, kk.x, part[rr][g]);
                        part[rr][g] = __fmaf_rn(qq.y, kk.y, part[rr][g]);
                        part[rr][g] = __fmaf_rn(qq.z, kk.z, part[rr][g]);
                        part[rr][g] = __fmaf_rn(qq.w, kk.w, part[rr][g]);
                    }
                }
            }
        }
        float alpha[R][G];
#pragma unroll
        for (int rr = 0; rr < R; rr++) {
            const bool valid = key < hiRow[rr];
#pragma unroll
            for (int g = 0; g < G; g++) {
                float d = part[rr][g];
                d += __shfl_xor_sync(0xffffffff, d, 1);
                d += __shfl_xor_sync(0xffffffff, d, 2);
                const float s = valid ? d * scale : FA_NEG;
                float mx = s;
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16));
                const float mn = fmaxf(m[rr][g], mx);
                alpha[rr][g] = __expf(m[rr][g] - mn);
                const float p = valid ? __expf(s - mn) : 0.f;
                float ps = p;
                ps += __shfl_xor_sync(0xffffffff, ps, 4);
                ps += __shfl_xor_sync(0xffffffff, ps, 8);
                ps += __shfl_xor_sync(0xffffffff, ps, 16);
                l[rr][g] = __fmaf_rn(l[rr][g], alpha[rr][g], ps);
                m[rr][g] = mn;
                if (sub == 0) pb[(rr * 8 + grp) * 8 + g] = p;
            }
        }
        __syncwarp();
#pragma unroll
        for (int rr = 0; rr < R; rr++) {
#pragma unroll
            for (int g = 0; g < G; g++) {
#pragma unroll
                for (int c = 0; c < ND; c++) acc[rr][g][c] *= alpha[rr][g];
            }
        }
        const int nb = min(8, hiMax - b0);
        for (int kk = 0; kk < nb; kk++) {
            const float* vp = vc + (long)(b0 + kk) * kvDim + (long)kvh * HD;
            float v[ND];
#pragma unroll
            for (int c = 0; c < NCH; c++) {
                if (VEC == 4) {
                    const float4 t = ((const float4*)vp)[c * 32 + lane];
                    v[c * 4 + 0] = t.x; v[c * 4 + 1] = t.y; v[c * 4 + 2] = t.z; v[c * 4 + 3] = t.w;
                } else {
                    const float2 t = ((const float2*)vp)[lane];
                    v[0] = t.x; v[1] = t.y;
                }
            }
            const int kabs = b0 + kk;
#pragma unroll
            for (int rr = 0; rr < R; rr++) {
                if (kabs < hiRow[rr]) {              // the row's own keys only: masked keys' P.V is SKIPPED, as in the M=1 kernel
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const float pv = pb[(rr * 8 + kk) * 8 + g];
#pragma unroll
                        for (int c = 0; c < ND; c++) acc[rr][g][c] = __fmaf_rn(pv, v[c], acc[rr][g][c]);
                    }
                }
            }
        }
        __syncwarp();
    }

    // ---- per-row unnormalised per-warp partials, laid out [row][head][unit][HD+4] ----
    const int U = nSplit * FA_WARPS, u = sp * FA_WARPS + warp;
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
        if (rb + rr < nRows) {
#pragma unroll
            for (int g = 0; g < G; g++) {
                float* out = partial + (((long)(rb + rr) * nH + (kvh * G + g)) * U + u) * (HD + 4);   // slot = row within THIS launch
#pragma unroll
                for (int c = 0; c < NCH; c++) {
                    if (VEC == 4) {
                        ((float4*)out)[c * 32 + lane] = make_float4(acc[rr][g][c * 4], acc[rr][g][c * 4 + 1], acc[rr][g][c * 4 + 2], acc[rr][g][c * 4 + 3]);
                    } else {
                        ((float2*)out)[lane] = make_float2(acc[rr][g][0], acc[rr][g][1]);
                    }
                }
                if (lane == 0) { out[HD] = m[rr][g]; out[HD + 1] = l[rr][g]; }
            }
        }
    }
}

#define FA_ROWS_KERNEL(HD, G) \
extern "C" __global__ void fa_partial_rows_##HD##_g##G(const float* __restrict__ q, const float* __restrict__ kc, \
    const float* __restrict__ vc, int nH, int nKV, int winStart, int nKeys0, int per, float scale, int nSplit, \
    int nRows, int row0, float* __restrict__ partial) { \
    fa_partial_rows_impl<HD, G>(q, kc, vc, nH, nKV, winStart, nKeys0, per, scale, nSplit, nRows, row0, partial); \
}
#define FA_ROWS_HD(HD) FA_ROWS_KERNEL(HD, 1) FA_ROWS_KERNEL(HD, 2) FA_ROWS_KERNEL(HD, 3) FA_ROWS_KERNEL(HD, 4) \
    FA_ROWS_KERNEL(HD, 5) FA_ROWS_KERNEL(HD, 6) FA_ROWS_KERNEL(HD, 7) FA_ROWS_KERNEL(HD, 8)
FA_ROWS_HD(64)
FA_ROWS_HD(128)
FA_ROWS_HD(256)

// fa_combine_rows: fa_combine for the rows of a run. grid (nH, nRows), block hd, dynamic shared memory 3*U floats. The
// per-row body is fa_combine's, expression for expression, so a row's context equals the M=1 combine's.
extern "C" __global__ void fa_combine_rows(const float* __restrict__ partial, int nH, int hd, int U, int row0, float* __restrict__ ctx) {
    extern __shared__ float cs[];
    float* sm = cs; float* sw = cs + U; float* sl = cs + 2 * U;
    const int h = blockIdx.x, slot = blockIdx.y, row = row0 + blockIdx.y, d = threadIdx.x;   // slot: partial row in this launch; row: q/ctx row in the batch
    const int stride = hd + 4;
    const float* base = partial + ((long)slot * nH + h) * U * stride;
    for (int u = threadIdx.x; u < U; u += blockDim.x) { sm[u] = base[(long)u * stride + hd]; sl[u] = base[(long)u * stride + hd + 1]; }
    __syncthreads();
    float M = FA_NEG;
    for (int u = 0; u < U; u++) M = fmaxf(M, sm[u]);
    for (int u = threadIdx.x; u < U; u += blockDim.x) sw[u] = (sm[u] > FA_SEEN) ? __expf(sm[u] - M) : 0.f;
    __syncthreads();
    if (d >= hd) return;
    float num = 0.f, den = 0.f;
#pragma unroll 8
    for (int u = 0; u < U; u++) {
        const float w = sw[u];
        num = __fmaf_rn(w, base[(long)u * stride + d], num);
        den = __fmaf_rn(w, sl[u], den);
    }
    ctx[((long)row * nH + h) * hd + d] = num / den;
}
