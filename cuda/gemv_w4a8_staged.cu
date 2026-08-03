#include <cuda_fp16.h>

// gemv_w4a8_staged: the activation-staged batched W4A8 GEMV. Same contract and same bit-for-bit result
// as gemv_w4a8_batched / the M=1 gemv_w4a8_fwd, but it stages the activation tile in SHARED MEMORY so
// each activation element is read from L2/DRAM once per block instead of once per output row.
//
// RESULT — this kernel is the EXPERIMENT that refuted "activation-bandwidth-bound", not a shipped win.
// The un-staged batched GEMV re-reads the whole [M,K] activation once per output-row warp (N*M*K vs the
// M*K minimum). Staging it here and sharing across BLK/32 warps cut global activation reads 8x (7.05 GB
// -> 0.88 GB at the gate/up shape, TestGemvStagedBandwidth) — exactly as the re-read model predicted —
// yet wall time moved only 4.98 -> 4.2 ms (1.2x), 7.9% -> 9.3% of dp4a peak. An 8x traffic cut buying ~0
// means the kernel was NOT bandwidth-bound: the earlier 1.41 TB/s was the read RATE the compute loop
// demanded (served fine by L2), not a saturated ceiling. The 4.2 ms is spent in the compute loop (the
// 0.88 GB load alone is ~0.6 ms), so the real bound is the dp4a + one-shared-read-per-MAC loop —
// LSU/issue throughput at low arithmetic intensity (~4 MACs per activation read). The lever for THAT is
// register-blocking RN output rows per warp (each activation read feeds RN* the MACs), not staging. This
// kernel is kept, gated bit-identical, as the reproducible refutation; it is NOT wired into PrefillLast.
//
// BIT-IDENTITY — the one thing that must not move. Each lane keeps its facc[MT] accumulators LIVE in
// registers across every K-chunk; the lane->word mapping (wi = base+lane, the 64-stride main + 32-stride
// tail) is copied verbatim; the shfl_down warp-reduce happens EXACTLY ONCE, after all chunks. Only the
// activation OPERAND is read from sA[] instead of A[]. A per-chunk partial reduction would reorder the
// float sum and break identity — so there is none. The chunked order equals the un-chunked order iff a
// chunk boundary never splits a 64-stride: KC is REQUIRED to be a multiple of 64, so every non-final
// chunk is whole 64-strides and only the final chunk carries the global 32-tail (exactly as the
// un-chunked kernel). The exhaustive gate (every N*M output vs gemv_w4a8_fwd) — not a spot check — proves
// it at each (MT, KC).
//
// MT = activation columns per weight-row pass (register-bound, facc[MT] per lane). KC = K-chunk width in
// WEIGHT-WORDS, multiple of 64. BLK = threads/block (BLK/32 warps share each staged tile = the reuse
// factor). Shared bytes = MT * 2 * KC * 4 (the launcher must pass this).
#ifndef MT
#define MT 32
#endif
#ifndef KC
#define KC 64
#endif
#ifndef BLK
#define BLK 256
#endif

extern "C" __global__ void gemv_w4a8_staged(
    const unsigned int* __restrict__ W, const int* __restrict__ A, const __half* __restrict__ gs,
    const float* __restrict__ aScale, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, int M, float* __restrict__ dst, int accum)
{
    const int nWarps = BLK / 32;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * nWarps + warp;          // this warp's output row
    const unsigned int* wr = W + (long)(n < N ? n : 0) * Kwords;   // deref only under (n<N)
    const __half* sr = gs + (long)(n < N ? n : 0) * Kgroups;

    // Staged activation, laid out word-major and PADDED as [2*KC][MT+1]: activation-word aw, column t at
    // sA[aw*(MT+1) + t]. Two access patterns fight over the 32 banks and both must stay unconflicted: the
    // hot inner t-loop (word fixed, t varies) needs stride 1 → contiguous t; the warp-parallel read (t
    // fixed, 32 lanes read words 2 apart) would then stride 2*(MT+1). The +1 pad is what does it: with
    // stride MT it is 2*MT (a multiple of 32 → 32-way lane conflict); MT+1 makes it 66 → banks differ by
    // 2 → only 2-way. (An un-padded [MT][2*KC] instead conflicts the t-loop 32-way — measured 4.2 ms.)
    // Layout only — values and accumulation order are untouched, so bit-identity holds.
    const int STRIDE = MT + 1;
    extern __shared__ int sA[];                  // [2*KC][MT+1]

    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = (M - m0 < MT) ? (M - m0) : MT;
        float facc[MT];
        #pragma unroll
        for (int t = 0; t < MT; t++) facc[t] = 0.f;

        for (int c0 = 0; c0 < Kwords; c0 += KC) {
            int cW = (Kwords - c0 < KC) ? (Kwords - c0) : KC;   // weight-words this chunk
            int nCol = 2 * cW;                                  // activation int32 words this chunk
            // Cooperative load A[m0:m0+mcnt, 2*c0 : 2*c0+nCol] -> sA (all BLK threads, incl. n>=N warps).
            __syncthreads();
            for (int i = threadIdx.x; i < mcnt * nCol; i += BLK) {
                int row = i / nCol, col = i - row * nCol;       // row = column t, col = activation word
                sA[col * STRIDE + row] = A[(long)(m0 + row) * (2 * Kwords) + 2 * c0 + col];
            }
            __syncthreads();
            if (n >= N) continue;                 // stayed for the syncs; nothing to compute

            int chunkEnd = c0 + cW;
            int base = c0;
            // 64-stride main (whole strides only — KC%64==0 guarantees no boundary split).
            for (; base + 64 <= chunkEnd; base += 64) {
                int wi0 = base + lane, wi1 = base + 32 + lane;
                unsigned int w0 = wr[wi0], w1 = wr[wi1];
                int lo0 = (int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u);
                int hi0 = (int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                int lo1 = (int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u);
                int hi1 = (int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                float s0 = __half2float(sr[wi0 >> 2]);
                float s1 = __half2float(sr[wi1 >> 2]);
                // Word rows in the transposed tile; the t-loop then strides by 1 (bank-conflict-free).
                const int* a0 = sA + (2 * (wi0 - c0)) * STRIDE;
                const int* a1 = sA + (2 * (wi1 - c0)) * STRIDE;
                for (int t = 0; t < mcnt; t++) {
                    int p0 = 0, p1 = 0;
                    p0 = __dp4a(lo0, a0[t], p0);
                    p0 = __dp4a(hi0, a0[STRIDE + t], p0);
                    p1 = __dp4a(lo1, a1[t], p1);
                    p1 = __dp4a(hi1, a1[STRIDE + t], p1);
                    facc[t] += (float)p0 * s0;
                    facc[t] += (float)p1 * s1;
                }
            }
            // 32-stride tail (only the final chunk, when Kwords%64!=0 — same as the un-chunked kernel).
            for (; base < chunkEnd; base += 32) {
                int wi = base + lane;
                if (wi < chunkEnd) {
                    unsigned int word = wr[wi];
                    int lo = (int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
                    int hi = (int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                    float s = __half2float(sr[wi >> 2]);
                    const int* a = sA + (2 * (wi - c0)) * STRIDE;
                    for (int t = 0; t < mcnt; t++) {
                        int p = 0;
                        p = __dp4a(lo, a[t], p);
                        p = __dp4a(hi, a[STRIDE + t], p);
                        facc[t] += (float)p * s;
                    }
                }
            }
        }
        if (n < N) {
            // ONE warp-reduce, after all chunks — facc never partially reduced (bit-identity).
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
}
