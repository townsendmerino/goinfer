#include <cuda_fp16.h>
// W4A8 GEMV, COALESCED: consecutive lanes read consecutive words (wr[base+lane]) so each
// warp load is a single 128B transaction, instead of the stride-4 (16B) pattern that
// capped the naive/fast kernels at 43% peak. A group is 4 consecutive words → reduce each
// 4-lane segment (shfl_down 1,2), the segment leader scales by the group's f16 scale.
// Same nibble-permuted fast layout + even/odd + __vsub4 unpack. Requires Kwords % 32 == 0
// (true for all real projection dims: K%256==0).
extern "C" __global__ void gemv_w4a8_coal(
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
        int wi = base + lane;                       // consecutive → coalesced
        unsigned int word = wr[wi];
        int es = __vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
        int os = __vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        int p = 0;
        p = __dp4a(es, a[2 * wi], p);
        p = __dp4a(os, a[2 * wi + 1], p);
        p += __shfl_down_sync(0xffffffffu, p, 1);   // reduce the group's 4 words
        p += __shfl_down_sync(0xffffffffu, p, 2);
        if ((lane & 3) == 0) facc += (float)p * __half2float(sr[wi >> 2]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) dst[n] = facc * aScale;
}
