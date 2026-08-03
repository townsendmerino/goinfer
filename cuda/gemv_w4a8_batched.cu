#include <cuda_fp16.h>

// gemv_w4a8_batched: the weight-stationary batched GEMV for M=len prefill. It computes, for M
// activation rows against one int4 weight matrix, dst[m,n] = (sum_k dequant(W[n,k]) * A[m,k]) *
// aScale[m] + bias[n] — the SAME contract as gemv_w4a8_fwd, evaluated for M columns per weight-row
// load instead of one. The weight-streaming amortization (each W word read once per MT columns rather
// than once per column) is the prefill win: prefill is weight-bandwidth-bound, so batching M tokens
// per weight read is the lever, not raw compute.
//
// BIT-IDENTICAL TO gemv_w4a8_fwd BY CONSTRUCTION. The int4 weights are GROUP-scaled (one f16 scale
// per 32-element group along K), so the cross-group sum MUST be in float — and float add is not
// associative. Bit-identity therefore is NOT "int32-accumulate then scale once"; it requires the
// per-output float accumulation to visit words in the SAME order and warp-reduce the SAME way as the
// M=1 GEMV. This kernel does exactly that: the word loop (64-stride ILP main + 32-stride guarded tail),
// the per-word __dp4a int32 partial, the per-word `facc += (float)p * groupScale`, and the
// shfl_down(16..1) reduce are copied from gemv_w4a8_fwd verbatim; only an MT-wide inner loop over
// activation columns is added, each column running that identical sequence. So the batched path at M=1
// reproduces the GEMV bit-for-bit (the cheapest regression check), and at any M every output element
// equals its sequential-GEMV value to the bit. This is WHY there is no IMMA/MMA path: tensor-core
// accumulation reorders the cross-group float sum and would break the byte-identical gate.
//
// PERFORMANCE CEILING, CHOSEN NOT MISSED: __dp4a is ~1/3 of Turing IMMA int8 throughput. The op is
// weight-bandwidth-bound up to M~=45 and compute-bound above it, so at M=512 expect ~23x on TTFT
// rather than the ~72x an IMMA path could theoretically reach. IMMA is unavailable here without moving
// to a tolerance gate (see the parked "int32-per-group GEMV" candidate in docs) — bit-identity was the
// explicit constraint, so dp4a is the right primitive.
//
// Args mirror gemv_w4a8_fwd, plus M and the per-row activation arrays:
//   W  [N, Kwords] u32 (8 int4/word, nibble-permuted at pack time, permuteFast)
//   A  [M, 2*Kwords] int (4 int8/word) — M activation rows, each row independently int8-quantized
//   gs [N, Kgroups] f16 group scales
//   aScale [M] f32 — per-row activation scale (A[m] used aScale[m])
//   bias [N] f32 or null
//   dst [M, N] f32 (row m = token m's output vector); accum selects dst += val.
// MT = activation columns handled per weight-row load (register-bound; 8 is a safe default).
#define MT 8

extern "C" __global__ void gemv_w4a8_batched(
    const unsigned int* __restrict__ W, const int* __restrict__ A, const __half* __restrict__ gs,
    const float* __restrict__ aScale, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, int M, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;

    // One weight-row load serves MT activation columns. Loop the M dimension in MT-wide tiles; the
    // last tile clamps to the M remainder. Per tile, the word loop reads each W word ONCE and drives
    // MT independent float accumulators — that is the weight-streaming amortization.
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = (M - m0 < MT) ? (M - m0) : MT;
        float facc[MT];
        #pragma unroll
        for (int t = 0; t < MT; t++) facc[t] = 0.f;

        int base = 0;
        for (; base + 64 <= Kwords; base += 64) {
            int wi0 = base + lane, wi1 = base + 32 + lane;
            unsigned int w0 = wr[wi0], w1 = wr[wi1];
            int lo0 = (int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u);
            int hi0 = (int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
            int lo1 = (int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u);
            int hi1 = (int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
            float s0 = __half2float(sr[wi0 >> 2]);
            float s1 = __half2float(sr[wi1 >> 2]);
            for (int t = 0; t < mcnt; t++) {
                const int* a = A + (long)(m0 + t) * (2 * Kwords);
                int p0 = 0, p1 = 0;
                p0 = __dp4a(lo0, a[2 * wi0], p0);
                p0 = __dp4a(hi0, a[2 * wi0 + 1], p0);
                p1 = __dp4a(lo1, a[2 * wi1], p1);
                p1 = __dp4a(hi1, a[2 * wi1 + 1], p1);
                facc[t] += (float)p0 * s0;
                facc[t] += (float)p1 * s1;
            }
        }
        for (; base < Kwords; base += 32) {
            int wi = base + lane;
            if (wi < Kwords) {
                unsigned int word = wr[wi];
                int lo = (int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
                int hi = (int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                float s = __half2float(sr[wi >> 2]);
                for (int t = 0; t < mcnt; t++) {
                    const int* a = A + (long)(m0 + t) * (2 * Kwords);
                    int p = 0;
                    p = __dp4a(lo, a[2 * wi], p);
                    p = __dp4a(hi, a[2 * wi + 1], p);
                    facc[t] += (float)p * s;
                }
            }
        }
        // Identical warp-reduce per column.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            for (int t = 0; t < mcnt; t++) facc[t] += __shfl_down_sync(0xffffffffu, facc[t], off);
        }
        if (lane == 0) {
            for (int t = 0; t < mcnt; t++) {
                int m = m0 + t;
                float val = facc[t] * aScale[m] + (bias ? bias[n] : 0.f);
                dst[(long)m * N + n] = accum ? dst[(long)m * N + n] + val : val;
            }
        }
    }
}
