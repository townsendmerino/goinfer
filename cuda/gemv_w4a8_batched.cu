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
// PERFORMANCE, PROFILED (ncu on sm_75, the gate/up shape N=8960 K=1536 M=512). This kernel is
// L1TEX-LATENCY-BOUND, not compute / DRAM / issue bound: DRAM 1.7%, L2 6%, Compute 46%, but the
// schedulers sit idle (No Eligible 71%, IPC 1.07/4) with ~7.8 active warps stalling ~11 cycles each on
// an L1TEX scoreboard dependency. It took FIVE attributions to get here — each of the first four was a
// plausible mechanism recorded as a conclusion and refuted by the next measurement: (1) "~23x
// weight-amortization ceiling" (MT sweep: ~6%); (2) "needs IMMA" (7.9% of dp4a peak — compute ceiling
// unused); (3) "activation-L2-bandwidth-bound" (built the shared-staging kernel, cut global reads 8x,
// time moved 1.2x — NOT bandwidth; the 1.41 TB/s was the rate the loop DEMANDED, L2-served, not a
// ceiling); (4) "issue-bound on a fat instruction mix" (ncu: 22.78% issue slots — not issue-bound). The
// hardware stated (5) directly.
//
// THE APPLIED FIX — coalesced activation load (the int2 in the inner loop). The activation pair was two
// separate stride-2 loads → adjacent lanes read words 2 apart → only 16 of 32 bytes per L1TEX sector
// used (ncu: 49.99% bytes/sector, L1TEX 93%). Loading the pair as one int2 → 98% bytes/sector, L1TEX
// 93%->65%, ~4.98->4.4 ms. Bit-identical (same int32 values, same dp4a order). MODEST because the bound
// only MOVED: L1TEX latency is still exposed (17.8 cyc/instr scoreboard stall, no unit saturated) — too
// few eligible warps to hide the now-efficient loads. The NEXT lever, and the first the profile
// actually justifies, is arithmetic intensity: register-block RN output rows per warp so each activation
// load feeds RN* the MACs (a LATENCY fix, still bit-identical — per-row facc, one reduce each). NOT
// activation-staging (built, refuted) and NOT IMMA (compute ceiling unused). ptxas: 61 regs @MT=32,
// 100% theoretical / 83% achieved occupancy — occupancy is not the lever.
//
// Args mirror gemv_w4a8_fwd, plus M and the per-row activation arrays:
//   W  [N, Kwords] u32 (8 int4/word, nibble-permuted at pack time, permuteFast)
//   A  [M, 2*Kwords] int (4 int8/word) — M activation rows, each row independently int8-quantized
//   gs [N, Kgroups] f16 group scales
//   aScale [M] f32 — per-row activation scale (A[m] used aScale[m])
//   bias [N] f32 or null
//   dst [M, N] f32 (row m = token m's output vector); accum selects dst += val.
// MT = activation columns handled per weight-row load. 32 is the measured sweet spot (see above): the
// bit-identity gate passes at every MT (tiling never touches an element's accumulation order), 32 gives
// the best prefill, 64 spills registers and regresses.
#define MT 32

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
                // Load each activation pair (a[2*wi], a[2*wi+1]) as ONE 64-bit int2. The two used to be
                // separate stride-2 loads: adjacent lanes read words 2 apart → only even (or odd) words
                // per 32-byte sector → 50% of every L1TEX sector wasted (ncu: 49.99% bytes/sector, 7.31
                // sectors/req; the kernel was L1TEX-latency-bound at 93% L1TEX with DRAM idle). The int2
                // is 8 contiguous bytes/lane, adjacent lanes 8 apart → the warp reads 256 contiguous
                // bytes, fully coalesced. Bit-identical: same int32 values, same dp4a order, same facc.
                int2 av0 = *(const int2*)(a + 2 * wi0);
                int2 av1 = *(const int2*)(a + 2 * wi1);
                int p0 = 0, p1 = 0;
                p0 = __dp4a(lo0, av0.x, p0);
                p0 = __dp4a(hi0, av0.y, p0);
                p1 = __dp4a(lo1, av1.x, p1);
                p1 = __dp4a(hi1, av1.y, p1);
                facc[t] = __fmaf_rn((float)p0, s0, facc[t]);
                facc[t] = __fmaf_rn((float)p1, s1, facc[t]);
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
                    int2 av = *(const int2*)(a + 2 * wi);   // coalesced pair load (see main loop)
                    int p = 0;
                    p = __dp4a(lo, av.x, p);
                    p = __dp4a(hi, av.y, p);
                    facc[t] = __fmaf_rn((float)p, s, facc[t]);
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
                float val = __fmaf_rn(facc[t], aScale[m], (bias ? bias[n] : 0.f));
                dst[(long)m * N + n] = accum ? dst[(long)m * N + n] + val : val;
            }
        }
    }
}
