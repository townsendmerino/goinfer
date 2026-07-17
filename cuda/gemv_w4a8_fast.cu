#include <cuda_fp16.h>
// W4A8 GEMV, fast unpack: nibble-permuted layout so the even/odd byte-mask + __vsub4 SIMD
// path recovers signed int8 in ~4 ops/word (vs the 8-iter scalar loop). Byte b of `e` (low
// nibbles) = weight b; byte b of `o` (high nibbles) = weight b+4. Pairs with the existing
// consecutive int8-activation packing. group=32 nibbles, f16 scale/group.
extern "C" __global__ void gemv_w4a8_fast(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    float aScale, int N, int Kwords, int Kgroups, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    for (int g = lane; g < Kgroups; g += 32) {
        int iacc = 0;
        const unsigned int* wg = wr + 4 * g;
        const int* ag = a + 8 * g;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            unsigned int word = wg[w];
            int es = __vsub4(word & 0x0F0F0F0Fu, 0x08080808u);
            int os = __vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
            iacc = __dp4a(es, ag[2 * w], iacc);
            iacc = __dp4a(os, ag[2 * w + 1], iacc);
        }
        facc += (float)iacc * __half2float(sr[g]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffff, facc, off);
    if (lane == 0) dst[n] = facc * aScale;
}
