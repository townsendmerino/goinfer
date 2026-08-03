#include <cuda_fp16.h>

// gemv_w4a8_rn: register-blocked batched GEMV. Same contract and same bit-for-bit result as
// gemv_w4a8_batched / gemv_w4a8_fwd, but each warp computes RN OUTPUT ROWS instead of one, so each
// activation int2 load is reused across RN rows (held in registers) — RN× fewer activation loads.
//
// WHY: ncu found gemv_w4a8_batched L1TEX-latency-bound even after the load was coalesced (17.8
// cyc/instr scoreboard stall, no unit saturated) — each dp4a waits on its activation load and there are
// too few in flight to hide it. Occupancy is already 100%, so more warps can't help; the lever is fewer
// loads per MAC (arithmetic intensity). Blocking RN rows per warp cuts the activation load COUNT by RN
// (the reused operand is the bottleneck) while the weight loads (trivial, 1.4 GB/s) grow RN×.
//
// BIT-IDENTITY: the RN rows are independent GEMV rows; each keeps its own facc[r][*] live across ALL K,
// visits words in the same lane→word order as the M=1 GEMV, and warp-reduces once at the end. Only the
// activation OPERAND is shared (same int32 values), so every output equals gemv_w4a8_fwd to the bit.
// RN*MT accumulators live in registers — keep RN*MT ≈ 32 (the un-blocked MT=32) to stay off the spill.
#ifndef RN
#define RN 2
#endif
#ifndef MT
#define MT 16
#endif

extern "C" __global__ void gemv_w4a8_rn(
    const unsigned int* __restrict__ W, const int* __restrict__ A, const __half* __restrict__ gs,
    const float* __restrict__ aScale, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, int M, float* __restrict__ dst, int accum)
{
    int warpsPerBlock = blockDim.x / 32;
    int warpId = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);
    int n0 = warpId * RN;                    // this warp owns rows [n0, n0+RN)
    int lane = threadIdx.x & 31;
    if (n0 >= N) return;

    const unsigned int* wr[RN];
    const __half* sr[RN];
    #pragma unroll
    for (int r = 0; r < RN; r++) {
        int nr = (n0 + r < N) ? n0 + r : 0;   // clamp OOB pointer (guarded on write)
        wr[r] = W + (long)nr * Kwords;
        sr[r] = gs + (long)nr * Kgroups;
    }

    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = (M - m0 < MT) ? (M - m0) : MT;
        float facc[RN][MT];
        #pragma unroll
        for (int r = 0; r < RN; r++)
            for (int t = 0; t < MT; t++) facc[r][t] = 0.f;

        int base = 0;
        for (; base + 64 <= Kwords; base += 64) {
            int wi0 = base + lane, wi1 = base + 32 + lane;
            int lo0[RN], hi0[RN], lo1[RN], hi1[RN];
            float s0[RN], s1[RN];
            #pragma unroll
            for (int r = 0; r < RN; r++) {
                unsigned int w0 = wr[r][wi0], w1 = wr[r][wi1];
                lo0[r] = (int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u);
                hi0[r] = (int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                lo1[r] = (int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u);
                hi1[r] = (int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                s0[r] = __half2float(sr[r][wi0 >> 2]);
                s1[r] = __half2float(sr[r][wi1 >> 2]);
            }
            for (int t = 0; t < mcnt; t++) {
                const int* a = A + (long)(m0 + t) * (2 * Kwords);
                int2 av0 = *(const int2*)(a + 2 * wi0);   // ONE coalesced load, reused for all RN rows
                int2 av1 = *(const int2*)(a + 2 * wi1);
                #pragma unroll
                for (int r = 0; r < RN; r++) {
                    int p0 = 0, p1 = 0;
                    p0 = __dp4a(lo0[r], av0.x, p0);
                    p0 = __dp4a(hi0[r], av0.y, p0);
                    p1 = __dp4a(lo1[r], av1.x, p1);
                    p1 = __dp4a(hi1[r], av1.y, p1);
                    facc[r][t] += (float)p0 * s0[r];
                    facc[r][t] += (float)p1 * s1[r];
                }
            }
        }
        for (; base < Kwords; base += 32) {
            int wi = base + lane;
            if (wi < Kwords) {
                int lo[RN], hi[RN];
                float s[RN];
                #pragma unroll
                for (int r = 0; r < RN; r++) {
                    unsigned int word = wr[r][wi];
                    lo[r] = (int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
                    hi[r] = (int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                    s[r] = __half2float(sr[r][wi >> 2]);
                }
                for (int t = 0; t < mcnt; t++) {
                    const int* a = A + (long)(m0 + t) * (2 * Kwords);
                    int2 av = *(const int2*)(a + 2 * wi);
                    #pragma unroll
                    for (int r = 0; r < RN; r++) {
                        int p = 0;
                        p = __dp4a(lo[r], av.x, p);
                        p = __dp4a(hi[r], av.y, p);
                        facc[r][t] += (float)p * s[r];
                    }
                }
            }
        }
        // one warp-reduce per (row, column), after all K — facc never partially reduced.
        #pragma unroll
        for (int r = 0; r < RN; r++) {
            for (int t = 0; t < mcnt; t++) {
                float v = facc[r][t];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0 && n0 + r < N) {
                    int m = m0 + t, n = n0 + r;
                    float val = v * aScale[m] + (bias ? bias[n] : 0.f);
                    dst[(long)m * N + n] = accum ? dst[(long)m * N + n] + val : val;
                }
            }
        }
    }
}
