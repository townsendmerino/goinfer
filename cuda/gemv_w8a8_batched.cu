#include <cuda_fp16.h>

// gemv_w8a8_batched: the weight-stationary batched W8A8 GEMV for M=len prefill. For M activation
// rows against one int8 weight matrix it computes
//   dst[m,n] = ((float)acc[m,n] * wScale[n]) * aScale[m] + bias[n]
// — the SAME contract as aikit/gpu's gemv_w8a8_fwd (the M=1 int8 GEMV the sequential decode/prefill
// uses), evaluated for MT activation columns per weight-row load instead of one. Each W word is read
// once per MT columns rather than once per column; prefill is weight-bandwidth-bound, so batching M
// tokens per weight read is the lever (the int4 batched kernel's argument, verbatim, one axis simpler).
//
// BIT-IDENTICAL TO gemv_w8a8_fwd BY CONSTRUCTION — and unlike the int4 batched kernel this is FREE, not
// engineered. int8 is per-row symmetric-scaled: one wScale[n] per output row, one aScale[m] per
// activation row, NO group axis. So the whole dot accumulates in EXACT int32 via __dp4a and the scales
// apply once at the end. Integer addition is associative and exact, so the accumulator does not depend
// on reduction order — this kernel may tile K / stage / re-associate freely and still produce the
// identical `acc` as the M=1 GEMV. (Contrast gemv_w4a8_batched, whose int4 group scales force the
// cross-group sum into non-associative float and made bit-identity a hand-built visit-order match.)
// The final f32 expression is the byte-for-byte same EXPLICIT form as gemv_w8a8_fwd — mul the two
// scales, fma the activation scale + bias — so there is no bare `a*b*c+d` and no compiler FMA
// contraction (gemv_w8a8_batched.cu is in TestKernelFMALint's file list, audit C-16).
//
// Three conditions (docs/ollama-chase.md §C6), all met:
//   1. No int32 overflow: worst case K·127² needs K > ~133k; real K <= 8192 (~16x margin).
//   2. Final f32 expression evaluated identically to M=1 (the explicit __fmul_rn/__fmaf_rn below).
//   3. Per-row activation scale aScale[m] (the batched quantizer already emits it, as for int4).
//
// Args mirror gemv_w8a8_fwd, plus M and the per-row aScale array:
//   W       [N, Kdiv4] int  (4 int8/word, row-major; K padded to a multiple of 4)
//   A       [M, Kdiv4] int  (4 int8/word) — M activation rows, each independently int8-quantized
//   wScale  [N] f32         — per output row (symmetric int8 weight scale)
//   aScale  [M] f32         — per activation row
//   bias    [N] f32 or null
//   dst     [M, N] f32      — row m = token m's output; accum selects dst += val
// MT = activation columns handled per weight-row load; 32 matches the int4 batched sweet spot. The
// bit-identity gate passes at every MT — tiling never touches an element's (exact-integer) accumulation.
#define MT 32

extern "C" __global__ void gemv_w8a8_batched(
    const int* __restrict__ W, const int* __restrict__ A, const float* __restrict__ wScale,
    const float* __restrict__ aScale, const float* __restrict__ bias,
    int N, int Kdiv4, int M, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int* wr = W + (long)n * Kdiv4;

    // Loop the M dimension in MT-wide tiles; the last tile clamps to the M remainder. Per tile the
    // word loop reads each W word ONCE and drives MT independent int32 accumulators — the amortization.
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = (M - m0 < MT) ? (M - m0) : MT;
        int acc[MT];
        #pragma unroll
        for (int t = 0; t < MT; t++) acc[t] = 0;

        for (int k = lane; k < Kdiv4; k += 32) {
            int w = wr[k]; // one weight word, shared across the MT activation columns
            for (int t = 0; t < mcnt; t++) {
                acc[t] = __dp4a(w, A[(long)(m0 + t) * Kdiv4 + k], acc[t]);
            }
        }
        // Integer warp-reduce per column — exact, so the order is irrelevant to the result.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            for (int t = 0; t < mcnt; t++) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
        }
        if (lane == 0) {
            for (int t = 0; t < mcnt; t++) {
                int m = m0 + t;
                // Identical to gemv_w8a8_fwd: __fmaf_rn(__fmul_rn((float)acc, wScale[n]), aScale[m], bias).
                float val = __fmaf_rn(__fmul_rn((float)acc[t], wScale[n]), aScale[m], (bias ? bias[n] : 0.f));
                dst[(long)m * N + n] = accum ? dst[(long)m * N + n] + val : val;
            }
        }
    }
}
