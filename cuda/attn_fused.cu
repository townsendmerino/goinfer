#include <cuda_fp16.h>

// attn_fused: FlashAttention-style fused prefill attention for sm_75 (docs/task-prefill-gap.md
// §4 L2). One block per (head, 64-query tile); K/V streamed in BN-key tiles through shared memory,
// converted f32->f16 on load; QK^T and PV both on tensor cores via mma.sync.aligned.m16n8k8; the
// softmax runs ONLINE in f32 (running max + denominator, O rescaled by exp(m_old - m_new) as the
// max moves), so no BM x nKeys score row is ever materialised.
//
// WHAT IT REPLACES AND WHY. attn_batched (prefill_batched.cu:163) is gridded (head, query row):
// each block streams the ENTIRE K prefix for ONE query, writes the full score row to shared memory,
// and then runs a serial nKeys-long FMA chain for the AV product. K and V are therefore re-read M
// times per layer and the AV tail grows with K -- the O(M*K) traffic with an O(K) serial tail that
// the 2026-09-01 re-anchor measured as marginal cost RISING 0.377 -> 0.932 ms/token where Ollama's
// is flat. This kernel reads each K/V tile ONCE per 64 queries and keeps the running softmax state
// in registers.
//
// NOT BIT-IDENTICAL, deliberately and by exactly one mechanism: the online rescale re-associates
// the softmax denominator and the AV fold. That is the --cpu-fast-attention category (P19), and it
// is why this kernel ships behind GOINFER_CUDA_FAST_PREFILL and why attn_batched REMAINS the exact
// path -- selectable, used by spec-decode verify and every parity gate, and used unconditionally
// below the M floor. Nothing here changes decode.
//
// WHY RAW mma.sync AND NOT wmma. The online softmax must rescale the O accumulator by
// exp(m_old - m_new) PER QUERY ROW between key tiles. wmma's accumulator fragment does not portably
// expose which row an element belongs to, so a wmma version would have to either round-trip O
// through shared memory every tile -- handing back the traffic this kernel exists to remove -- or
// depend on an undocumented Turing fragment layout. mma.sync's C mapping IS documented, and it pays
// a second dividend: the S accumulator's register layout (row = lane/4 and lane/4+8, col =
// (lane%4)*2 and +1) is IDENTICAL to the A-fragment layout the following PV mma wants, so P feeds
// the second matmul straight out of the softmax registers with no shared round trip.
//
// THE sm_75 SHAPE. m16n8k8 f16xf16->f32 is Turing's; m16n8k16 needs sm_80 (llama.cpp's mma.cuh
// makes the same split). Fragment layouts, from the PTX ISA, used throughout:
//   A 16x8 row-major : a0,a1 = row lane/4,   col (lane%4)*2, +1 ; a2,a3 = row lane/4+8, same cols
//   B  8x8 col-major : b0,b1 = col lane/4,   row (lane%4)*2, +1
//   C 16x8 f32       : c0,c1 = row lane/4,   col (lane%4)*2, +1 ; c2,c3 = row lane/4+8, same cols
//
// SHARED MEMORY, sized against the value MEASURED on this card rather than a datasheet: cuda/
// resident.go:131 records MAX_SHARED_MEMORY_PER_BLOCK = 49152 (48 KB) and the OPTIN limit = 65536
// (64 KB) for the RTX 2070 SUPER. This kernel needs Ksh[BN][hd] + Vtsh[hd][BN] in f16, padded --
// ~35 KB at hd=128, BN=64 -- so it fits the DEFAULT limit and needs no cuFuncSetAttribute. That is
// also a capability win independent of speed: the existing launch sizes shared memory as
// (maxNWin+128)*4 and so declines any layer attending more than 12,160 keys (checkPrefillShmem),
// while this kernel's footprint is CONSTANT in K.
//
// V IS STORED TRANSPOSED (Vtsh[hd][BN]) so the PV B-fragment is a contiguous half-pair per thread
// instead of a stride-hd gather; K is stored [BN][hd] so the QK^T B-fragment is likewise contiguous.
//
// KPAD IS 8 FOR A REASON, AND SMALLER IS NOT BETTER. With kStride = hd+8 = 136 halves = 68 words,
// lane L's QK^T B-fragment lands on bank (L/4)*68 + (L%4) == L (mod 32), so all 32 lanes hit 32
// distinct banks -- conflict-free. The V^T read works out identically (vStride = 72 halves = 36
// words, also == L). Dropping KPAD to 2 would save ~2 KB of shared memory AND remove the staging
// write's conflicts, but it puts kStride at 65 words and gives the far hotter mma-loop READS a
// 2-way conflict. The 4-way conflict that remains is on the one-time V^T staging WRITE, which is
// the right side of that trade: it happens once per tile, the reads happen once per mma.
//
// SEAMS -- every one attn_batched handles, handled here, each with its own test:
//   causal          : query row m attends keys [.., startPos+m], i.e. nKeys_m = startPos+m+1
//   sliding window  : per row, winStart_m = max(0, nKeys_m - window) when window > 0
//   GQA             : kvh = h / (nH/nKV)
//   gpt-oss sinks   : a per-head logit with NO key and NO value. Handled exactly by seeding the
//                     online state with m_run = sinks[h], l_run = 1, O = 0 -- an imaginary key whose
//                     logit is the sink and whose value vector is zero. It then joins the max and
//                     the denominator through the ordinary rescale and contributes nothing to the
//                     numerator, which is the definition attn_batched implements in two steps.
//   attnScale       : applied to the QK^T accumulator before the max, as attn_batched does
//
// hd MUST be 64 or 128. The host declines to attn_batched for anything else rather than
// special-casing it.
//
// AND hd IS A TEMPLATE PARAMETER, NOT AN ARGUMENT, for a reason the tree already has a test for.
// The O accumulator is a per-thread array indexed by an hd-derived loop bound; with hd passed as an
// int that bound is a RUNTIME value, the indexing is dynamic, and ptxas is forced to place the
// whole array in LOCAL memory -- turning the one structure this design keeps in registers into
// backing-store traffic, and adding a per-thread reservation of exactly the kind
// TestKernelLocalMemoryCensus was written to find. Templating makes the bound constexpr and the
// array register-resident; the two instantiations are exposed as separate entry points and the
// host picks one by hd.
//
// Regenerate with:  ./build_ptx.sh attn_fused

#define BM 64           // query rows per block (4 warps x 16)
#define BN 64           // key columns per tile
#define WARPS 4
#define KPAD 8          // half-elements of padding per row, to break shared-memory bank conflicts

__device__ __forceinline__ void mma_m16n8k8_f32(float &d0, float &d1, float &d2, float &d3,
                                                unsigned int a0, unsigned int a1, unsigned int b0,
                                                float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <int HD>
__device__ __forceinline__ void attn_fused_impl(const float* __restrict__ q,
                                                const float* __restrict__ kc,
                                                const float* __restrict__ vc, int nH, int nKV,
                                                int startPos, float scale, int window, int M,
                                                float* __restrict__ ctx,
                                                const float* __restrict__ sinks) {
    constexpr int OT = HD / 8;   // 8-wide n-slices of the output, constexpr so acc[] stays in regs
    constexpr int NG = BN / 8;   // 8-wide key groups per tile

    const int h = blockIdx.x;
    if (h >= nH) return;
    const int qTile = blockIdx.y * BM;
    if (qTile >= M) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qRow0 = lane >> 2;          // this lane's first query row within the warp's 16
    const int qRow1 = qRow0 + 8;          // and its second
    const int colBase = (lane & 3) * 2;   // this lane's column pair within each 8-wide group

    const int kvDim = nKV * HD, group = nH / nKV, kvh = h / group;
    const int qDim = nH * HD;

    extern __shared__ __half sh[];
    __half* Ksh  = sh;                                 // [BN][HD + KPAD]
    __half* Vtsh = sh + (size_t)BN * (HD + KPAD);      // [HD][BN + KPAD]
    constexpr int kStride = HD + KPAD;
    constexpr int vStride = BN + KPAD;

    const int mBase = qTile + warp * 16;
    const int m0abs = mBase + qRow0, m1abs = mBase + qRow1;
    const bool has0 = m0abs < M, has1 = m1abs < M;

    // ---- Q into registers as A fragments: qFrag[c] is the 16x8 chunk at hd = 8c. ----
    unsigned int qFrag[OT][2];
#pragma unroll
    for (int c = 0; c < OT; c++) {
        const int k0 = c * 8;
        float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
        if (has0) {
            const float* qh = q + (long)m0abs * qDim + h * HD;
            v00 = qh[k0 + colBase]; v01 = qh[k0 + colBase + 1];
        }
        if (has1) {
            const float* qh = q + (long)m1abs * qDim + h * HD;
            v10 = qh[k0 + colBase]; v11 = qh[k0 + colBase + 1];
        }
        __half2 p0 = __floats2half2_rn(v00, v01);
        __half2 p1 = __floats2half2_rn(v10, v11);
        qFrag[c][0] = *reinterpret_cast<unsigned int*>(&p0);
        qFrag[c][1] = *reinterpret_cast<unsigned int*>(&p1);
    }

    // ---- online softmax state ----
    // Seeding with the sink is what makes it "max and denominator only": an imaginary key whose
    // logit is sinkV and whose value vector is zero. It then rides the ordinary rescale.
    const bool hasSink = (sinks != nullptr);
    const float sinkV = hasSink ? sinks[h] : 0.f;
    float mRun0 = hasSink ? sinkV : -1e30f, mRun1 = mRun0;
    float lRun0 = hasSink ? 1.f : 0.f,      lRun1 = lRun0;
    float acc[OT][4];
#pragma unroll
    for (int t = 0; t < OT; t++) { acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f; }

    // Per-row causal extent and window start. Rows within a tile differ, so the staging loop uses
    // the block's union and the MASK does the per-row work -- the same seams attn_batched applies
    // per block, applied here per element.
    const int nKeys0 = startPos + (has0 ? m0abs : 0) + 1;
    const int nKeys1 = startPos + (has1 ? m1abs : 0) + 1;
    const int winStart0 = (window > 0 && nKeys0 > window) ? nKeys0 - window : 0;
    const int winStart1 = (window > 0 && nKeys1 > window) ? nKeys1 - window : 0;
    const int tileLastRow = (qTile + BM - 1 < M) ? qTile + BM - 1 : M - 1;
    const int blockMaxKeys = startPos + tileLastRow + 1;
    int blockMinWinStart = 0;
    if (window > 0) {
        const int firstRowKeys = startPos + qTile + 1;
        blockMinWinStart = (firstRowKeys > window) ? firstRowKeys - window : 0;
    }

    for (int s0 = blockMinWinStart; s0 < blockMaxKeys; s0 += BN) {
        const int nk = (blockMaxKeys - s0 < BN) ? blockMaxKeys - s0 : BN;
        __syncthreads();
        // ---- stage K [nk][HD] and V^T [HD][nk], f32 -> f16 ----
        for (int idx = threadIdx.x; idx < nk * HD; idx += blockDim.x) {
            const int kk = idx / HD, dd = idx - kk * HD;
            const float* src = kc + (long)(s0 + kk) * kvDim + kvh * HD;
            Ksh[kk * kStride + dd] = __float2half_rn(src[dd]);
        }
        for (int idx = threadIdx.x; idx < nk * HD; idx += blockDim.x) {
            const int kk = idx / HD, dd = idx - kk * HD;
            const float* src = vc + (long)(s0 + kk) * kvDim + kvh * HD;
            Vtsh[dd * vStride + kk] = __float2half_rn(src[dd]);
        }
        __syncthreads();

        // ---- S = Q . K^T for this tile, in registers ----
        float s[NG][4];
#pragma unroll
        for (int g = 0; g < NG; g++) { s[g][0] = 0.f; s[g][1] = 0.f; s[g][2] = 0.f; s[g][3] = 0.f; }
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int kIdx = g * 8 + (lane >> 2);
#pragma unroll
            for (int c = 0; c < OT; c++) {
                // B (8x8 col-major): b0,b1 = key lane/4 of this group, hd (lane%4)*2 and +1.
                unsigned int b0 = 0;
                if (kIdx < nk) b0 = *reinterpret_cast<const unsigned int*>(&Ksh[kIdx * kStride + c * 8 + colBase]);
                mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3], qFrag[c][0], qFrag[c][1], b0,
                                s[g][0], s[g][1], s[g][2], s[g][3]);
            }
        }

        // ---- mask + online softmax ----
        // C: c0,c1 = row qRow0 at key cols colBase, +1 ; c2,c3 = row qRow1 at the same cols.
        //
        // THE SENTINEL IS SAFE FOR EVERY MASKED ELEMENT EXCEPT ONE CASE, AND THAT CASE IS PER-ROW.
        // Masking to -1e30 and letting exp() flush it works whenever the row's running max is a
        // real score: exp(-1e30 - m) underflows to 0. It breaks only when the row has seen NO valid
        // key at all, because then mNew is ALSO -1e30 and exp(-1e30 + 1e30) = exp(0) = 1, so every
        // masked element would add 1.0 to the denominator. With causal masking an early query row
        // in the tile has no valid key in the LAST key tile, so this is reached on essentially every
        // prompt -- it is not a corner case. One bool per row settles it; a per-element validity
        // array would cost ~32 registers to say the same thing.
        float tileMax0 = -1e30f, tileMax1 = -1e30f;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int kA = s0 + g * 8 + colBase, kB = kA + 1;
            const int lim = s0 + nk;
            const float a0 = (kA >= winStart0 && kA < nKeys0 && kA < lim) ? __fmul_rn(s[g][0], scale) : -1e30f;
            const float a1 = (kB >= winStart0 && kB < nKeys0 && kB < lim) ? __fmul_rn(s[g][1], scale) : -1e30f;
            const float b0 = (kA >= winStart1 && kA < nKeys1 && kA < lim) ? __fmul_rn(s[g][2], scale) : -1e30f;
            const float b1 = (kB >= winStart1 && kB < nKeys1 && kB < lim) ? __fmul_rn(s[g][3], scale) : -1e30f;
            s[g][0] = a0; s[g][1] = a1; s[g][2] = b0; s[g][3] = b1;
            tileMax0 = fmaxf(tileMax0, fmaxf(a0, a1));
            tileMax1 = fmaxf(tileMax1, fmaxf(b0, b1));
        }
        // A row's 8 columns live across the 4 lanes of a quad; reduce over lanes 1 and 2.
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffffu, tileMax0, off));
            tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffffu, tileMax1, off));
        }
        const float mNew0 = fmaxf(mRun0, tileMax0), mNew1 = fmaxf(mRun1, tileMax1);
        const float alpha0 = __expf(mRun0 - mNew0), alpha1 = __expf(mRun1 - mNew1);

        // "live" = this row has a real max, so exp() of a masked -1e30 underflows to 0 as intended.
        const bool live0 = mNew0 > -1e29f, live1 = mNew1 > -1e29f;
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const float p0 = live0 ? __expf(s[g][0] - mNew0) : 0.f;
            const float p1 = live0 ? __expf(s[g][1] - mNew0) : 0.f;
            const float p2 = live1 ? __expf(s[g][2] - mNew1) : 0.f;
            const float p3 = live1 ? __expf(s[g][3] - mNew1) : 0.f;
            s[g][0] = p0; s[g][1] = p1; s[g][2] = p2; s[g][3] = p3;
            sum0 = __fadd_rn(sum0, __fadd_rn(p0, p1));
            sum1 = __fadd_rn(sum1, __fadd_rn(p2, p3));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            sum0 = __fadd_rn(sum0, __shfl_xor_sync(0xffffffffu, sum0, off));
            sum1 = __fadd_rn(sum1, __shfl_xor_sync(0xffffffffu, sum1, off));
        }
        lRun0 = __fmaf_rn(alpha0, lRun0, sum0);
        lRun1 = __fmaf_rn(alpha1, lRun1, sum1);
        mRun0 = mNew0; mRun1 = mNew1;

        // ---- O = alpha * O + P . V ----
        // The S registers already hold P in EXACTLY the A-fragment layout (row lane/4 and +8, col
        // (lane%4)*2 and +1), so P feeds the second matmul with no shared-memory round trip.
#pragma unroll
        for (int t = 0; t < OT; t++) {
            acc[t][0] = __fmul_rn(acc[t][0], alpha0);
            acc[t][1] = __fmul_rn(acc[t][1], alpha0);
            acc[t][2] = __fmul_rn(acc[t][2], alpha1);
            acc[t][3] = __fmul_rn(acc[t][3], alpha1);
        }
#pragma unroll
        for (int g = 0; g < NG; g++) {
            __half2 pa = __floats2half2_rn(s[g][0], s[g][1]);
            __half2 pb = __floats2half2_rn(s[g][2], s[g][3]);
            const unsigned int a0 = *reinterpret_cast<unsigned int*>(&pa);
            const unsigned int a1 = *reinterpret_cast<unsigned int*>(&pb);
            const int kIdx = g * 8 + colBase;
#pragma unroll
            for (int t = 0; t < OT; t++) {
                // B (8x8 col-major): b0,b1 = hd-col lane/4 of this slice, key (lane%4)*2 and +1.
                // V^T makes that pair contiguous.
                const int dIdx = t * 8 + (lane >> 2);
                unsigned int b0 = 0;
                if (kIdx + 1 < nk) {
                    b0 = *reinterpret_cast<const unsigned int*>(&Vtsh[dIdx * vStride + kIdx]);
                } else if (kIdx < nk) {
                    __half2 hv = __halves2half2(Vtsh[dIdx * vStride + kIdx], __float2half_rn(0.f));
                    b0 = *reinterpret_cast<unsigned int*>(&hv);
                }
                mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, b0,
                                acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
            }
        }
    }

    // ---- normalise and store ----
    // D of the PV mma: the pair (c0,c1) is one row at ADJACENT hd columns; the row is lane/4 (+8).
    const float inv0 = (lRun0 > 0.f) ? 1.f / lRun0 : 0.f;
    const float inv1 = (lRun1 > 0.f) ? 1.f / lRun1 : 0.f;
#pragma unroll
    for (int t = 0; t < OT; t++) {
        const int dcol = t * 8 + colBase;
        if (has0) {
            float* o = ctx + (long)m0abs * qDim + h * HD;
            o[dcol]     = __fmul_rn(acc[t][0], inv0);
            o[dcol + 1] = __fmul_rn(acc[t][1], inv0);
        }
        if (has1) {
            float* o = ctx + (long)m1abs * qDim + h * HD;
            o[dcol]     = __fmul_rn(acc[t][2], inv1);
            o[dcol + 1] = __fmul_rn(acc[t][3], inv1);
        }
    }
}

// Two instantiations, two entry points. The host picks by hd and declines anything else to
// attn_batched; there is no runtime-hd variant on purpose (see the header note on local memory).
extern "C" __global__ void attn_fused_hd64(const float* __restrict__ q, const float* __restrict__ kc,
                                           const float* __restrict__ vc, int nH, int nKV, int hd,
                                           int startPos, float scale, int window, int M,
                                           float* __restrict__ ctx, const float* __restrict__ sinks) {
    attn_fused_impl<64>(q, kc, vc, nH, nKV, startPos, scale, window, M, ctx, sinks);
}

extern "C" __global__ void attn_fused_hd128(const float* __restrict__ q, const float* __restrict__ kc,
                                            const float* __restrict__ vc, int nH, int nKV, int hd,
                                            int startPos, float scale, int window, int M,
                                            float* __restrict__ ctx, const float* __restrict__ sinks) {
    attn_fused_impl<128>(q, kc, vc, nH, nKV, startPos, scale, window, M, ctx, sinks);
}
