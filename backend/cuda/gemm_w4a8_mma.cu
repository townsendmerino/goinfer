#include <cuda_fp16.h>

// gemm_w4a8_mma: tensor-core int4xint8 GEMM with GROUP SCALES, for M>=16 prefill
// (docs/task-prefill-gap.md §4 L3). Same contract as gemv_w4a8_rn / gemv_w4a8_fwd —
// dst[m,n] = (sum_k dequant(W[n,k]) * A[m,k]) * aScale[m] + bias[n] — computed as a real GEMM on
// mma.sync instead of a warp-per-output-row dp4a GEMV that loops M.
//
// WHY THIS IS NOT THE THING THE SIBLING KERNEL'S HEADER REFUSED. gemv_w4a8_batched.cu records five
// successive attributions, four of them refuted by the next measurement, and its verdict is
// "L1TEX-LATENCY-BOUND ... NOT IMMA (compute ceiling unused)". That verdict is about swapping dp4a
// for IMMA *inside the same memory schedule*, and it stands: this kernel is at ~12% of dp4a peak,
// so a bigger compute peak alone buys nothing. What changes here is the SCHEDULE — a 64x64 block
// tile with the activation panel staged once in shared memory and reused across the block's whole
// N-slice, weights read ONCE per block instead of once per MT=16 columns, and 32 accumulators live
// in registers across the K loop. Arithmetic intensity is the lever the profile actually justifies,
// which is the same conclusion that header reaches in its last paragraph; tensor cores are how the
// arithmetic gets cheap enough for that to matter, not the fix by themselves. If this does not beat
// the band it will be because the schedule did not convert the latency bound, and that is the
// measurement, not an assumption.
//
// NOT BIT-IDENTICAL to gemv_w4a8_rn, and only in one way: the cross-group float sum is
// re-associated. Same int8 products, same per-group f16 scale, same f32 accumulation type. Per
// 32-wide group the two m8n8k16 MMAs accumulate EXACTLY in int32 (no rounding at all), and only the
// per-group fold `facc += float(c) * gs[n][g]` is float — which is what the dp4a kernel does per
// WORD today, in a different order. The expected difference is therefore ~K/32 float adds' worth of
// reassociation, not a precision change.
//
// THE sm_75 SHAPE: mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32. m16n8k16 / m16n8k32 need sm_80;
// llama.cpp's mma.cuh issues two m8n8k16 on Turing for the same reason.
//
// FRAGMENTS, from the PTX ISA, and why no transpose or shared staging is needed for the weights:
//   A 8x16 s8 row-major : a0..a3 = row lane/4, k = (lane%4)*4 .. +3  -> ONE activation int word,
//                         since activations are packed 4 int8 per word: word = k0/4 + (lane%4).
//   B 16x8 s8 col-major : b0..b3 = col lane/4 (a WEIGHT ROW n), k = (lane%4)*4 .. +3.
//                         The pack-time permutation (permuteFast/nibblePosFast, cuda/kernels.go)
//                         puts elements 8j..8j+3 in the LOW nibbles of word j and 8j+4..8j+7 in the
//                         HIGH nibbles, so `w & 0x0F0F0F0F` and `(w>>4) & 0x0F0F0F0F` each yield
//                         FOUR CONSECUTIVE k in order. With q = lane%4: word = k0/8 + (q>>1), and
//                         the half is lo when q is even, hi when odd. Verified against
//                         nibblePosFast before this kernel was written, not assumed.
//   C/D 8x8 s32         : c0 = row lane/4, col (lane%4)*2 ; c1 = same row, col +1. The two elements
//                         are two ADJACENT n, hence two group-scale loads per thread per group.
//
// WORK SPLIT: block tile 64(M) x 64(N), 4 warps, and warp w owns ALL 64 M rows over
// n in [16w, 16w+16). That split is chosen for the WEIGHTS: each warp takes a distinct n-range, so
// the block reads every weight word exactly once, while the activation panel — the far smaller
// operand (64xK int8 against 64xK/2 of weight per tile, and the weight matrix is what prefill is
// bandwidth-bound on) — is the one paid for four times, out of shared memory rather than global.
//
// SHARED MEMORY: the activation tile is [64][KSTEP/4 + APAD] int words. APAD=4 makes the row stride
// 36 words, so lane's bank = (m*36 + kw) % 32 = (m*4 + kw) % 32 which is DISTINCT for all 32 lanes
// (m in 0..7, kw in 0..3) -- conflict-free. A stride of 32 puts all eight m of a k-column in ONE
// bank (8-way conflict); a stride of 33 still collides ~3-way. 9 KB at KSTEP=128.
//
// Regenerate with:  ./build_ptx.sh gemm_w4a8_mma

#define GBM 64      // M rows per block
#define GBN 64      // N columns per block
#define GWARPS 4
#define GKSTEP 128  // K elements staged per shared-memory pass
#define GAPAD 4     // int-word padding on the activation tile's row stride (see above)
#define GMSUB 8     // m-subtiles per warp (64 rows / 8)
#define GNSUB 2     // n-subtiles per warp (16 cols / 8)

__device__ __forceinline__ void mma_m8n8k16_s32(int &d0, int &d1, unsigned int a, unsigned int b,
                                                int c0, int c1) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1));
}

extern "C" __global__ void gemm_w4a8_mma(
    const unsigned int* __restrict__ W, const int* __restrict__ A, const __half* __restrict__ gs,
    const float* __restrict__ aScale, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, int M, float* __restrict__ dst, int accum)
{
    const int K = Kwords * 8;              // elements along the contraction dim
    const int aWords = 2 * Kwords;         // activation int words per row (4 int8 each)
    const int m0 = blockIdx.y * GBM;
    const int n0 = blockIdx.x * GBN;
    if (m0 >= M || n0 >= N) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q = lane & 3;                // 0..3: selects the k-quarter of a 16-wide chunk
    const int lr = lane >> 2;              // 0..7: the fragment's row (A/C) or column (B)
    const int nWarp = n0 + warp * 16;      // this warp's n-range: [nWarp, nWarp+16)

    extern __shared__ int Ash[];           // [GBM][GKSTEP/4 + GAPAD]
    const int aStride = GKSTEP / 4 + GAPAD;

    float facc[GMSUB][GNSUB][2];
#pragma unroll
    for (int i = 0; i < GMSUB; i++)
#pragma unroll
        for (int j = 0; j < GNSUB; j++) { facc[i][j][0] = 0.f; facc[i][j][1] = 0.f; }

    for (int k0 = 0; k0 < K; k0 += GKSTEP) {
        const int kLen = (K - k0 < GKSTEP) ? K - k0 : GKSTEP;  // elements this pass
        const int kw = kLen >> 2;                              // activation words this pass
        __syncthreads();
        // Stage the activation panel: 64 rows x kw words, coalesced (consecutive threads take
        // consecutive words of a row). Rows past M are zeroed so an OOB tail contributes nothing.
        for (int idx = threadIdx.x; idx < GBM * kw; idx += blockDim.x) {
            const int rr = idx / kw, cc = idx - rr * kw;
            int v = 0;
            if (m0 + rr < M) v = A[(long)(m0 + rr) * aWords + (k0 >> 2) + cc];
            Ash[rr * aStride + cc] = v;
        }
        __syncthreads();

        // Each 32-wide group folds through its own int32 accumulator, then into f32 by its scale.
        for (int gk = 0; gk < kLen; gk += 32) {
            const int gIdx = (k0 + gk) >> 5;   // absolute group index along K
            if (gIdx >= Kgroups) break;

            // B fragments: 2 n-subtiles x 2 k-chunks. Straight from the packed weight word.
            unsigned int bf[GNSUB][2];
#pragma unroll
            for (int j = 0; j < GNSUB; j++) {
                const int n = nWarp + j * 8 + lr;
                const unsigned int* wr = W + (long)(n < N ? n : 0) * Kwords;
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    // k-chunk t covers absolute k in [k0+gk+16t, +16); this lane wants 4 of them.
                    const int kAbs = k0 + gk + 16 * t + q * 4;
                    unsigned int w = 0;
                    if (n < N && (kAbs >> 3) < Kwords) w = wr[kAbs >> 3];
                    const unsigned int halfw = (q & 1) ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu);
                    bf[j][t] = (n < N) ? __vsub4(halfw, 0x08080808u) : 0u;
                }
            }
            // A fragments: 8 m-subtiles x 2 k-chunks, from shared.
            unsigned int af[GMSUB][2];
#pragma unroll
            for (int i = 0; i < GMSUB; i++) {
                const int rr = i * 8 + lr;
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const int wcol = ((gk + 16 * t) >> 2) + q;
                    af[i][t] = (unsigned int)Ash[rr * aStride + wcol];
                }
            }
            // Two group scales per thread: the C fragment's two elements are adjacent n.
            float sc[GNSUB][2];
#pragma unroll
            for (int j = 0; j < GNSUB; j++) {
                const int nA = nWarp + j * 8 + q * 2, nB = nA + 1;
                sc[j][0] = (nA < N) ? __half2float(gs[(long)nA * Kgroups + gIdx]) : 0.f;
                sc[j][1] = (nB < N) ? __half2float(gs[(long)nB * Kgroups + gIdx]) : 0.f;
            }
#pragma unroll
            for (int i = 0; i < GMSUB; i++) {
#pragma unroll
                for (int j = 0; j < GNSUB; j++) {
                    int c0 = 0, c1 = 0;
                    mma_m8n8k16_s32(c0, c1, af[i][0], bf[j][0], c0, c1);
                    mma_m8n8k16_s32(c0, c1, af[i][1], bf[j][1], c0, c1);
                    // EXACT int32 through the group; only this fold is float, exactly as the dp4a
                    // kernel folds per word. Explicit intrinsics: no compiler contraction discretion.
                    facc[i][j][0] = __fmaf_rn((float)c0, sc[j][0], facc[i][j][0]);
                    facc[i][j][1] = __fmaf_rn((float)c1, sc[j][1], facc[i][j][1]);
                }
            }
        }
    }

    // Epilogue: per-row activation scale, bias, optional accumulate. Same as gemv_w4a8_rn's.
#pragma unroll
    for (int i = 0; i < GMSUB; i++) {
        const int m = m0 + i * 8 + lr;
        if (m >= M) continue;
        const float as = aScale[m];
#pragma unroll
        for (int j = 0; j < GNSUB; j++) {
            const int nA = nWarp + j * 8 + q * 2;
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int n = nA + e;
                if (n >= N) continue;
                const float val = __fmaf_rn(facc[i][j][e], as, (bias ? bias[n] : 0.f));
                dst[(long)m * N + n] = accum ? dst[(long)m * N + n] + val : val;
            }
        }
    }
}
