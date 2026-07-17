#include <cuda_fp16.h>
// W4A8 GEMV, COAL2: coalesced consecutive-word reads, but NO per-word segmented reduction.
// All 4 words of a group share one scale, so scale each word's int partial by its group
// scale and accumulate floats per lane — the group sum falls out of the final warp reduce.
// Drops 2 shfl/word (latency on the critical path) for 1 FMA/word (cheap, off the mem path).
extern "C" __global__ void gemv_w4a8_coal2(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    float aScale, int N, int Kwords, int Kgroups, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    for (int base = 0; base < Kwords; base += 32) {
        int wi = base + lane;
        unsigned int word = wr[wi];
        int es = __vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
        int os = __vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        int p = 0;
        p = __dp4a(es, a[2 * wi], p);
        p = __dp4a(os, a[2 * wi + 1], p);
        facc += (float)p * __half2float(sr[wi >> 2]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) dst[n] = facc * aScale;
}
