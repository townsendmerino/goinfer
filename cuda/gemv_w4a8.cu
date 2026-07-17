#include <cuda_fp16.h>
// W4A8 GEMV (spec §3): W nibble-packed (8 nibbles/u32, K/8 words/row), nibble = q+8 in [1,15];
// group = 32 nibbles, one f16 scale/group; a int8 (4/int). dst[n] = aScale * Σg s[n,g]*Σ(nib-8)*a.
extern "C" __global__ void gemv_w4a8(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    float aScale, int N, int Kwords, int Kgroups, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    for (int g = lane; g < Kgroups; g += 32) {           // 32 nibbles = 4 words = 8 act-ints
        int iacc = 0;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            unsigned int word = wr[4 * g + w];
            int lo = 0, hi = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) { int nb = ((int)((word >> (4 * j)) & 0xf)) - 8; lo |= (nb & 0xff) << (8 * j); }
            #pragma unroll
            for (int j = 0; j < 4; j++) { int nb = ((int)((word >> (4 * (j + 4))) & 0xf)) - 8; hi |= (nb & 0xff) << (8 * j); }
            iacc = __dp4a(lo, a[8 * g + 2 * w], iacc);
            iacc = __dp4a(hi, a[8 * g + 2 * w + 1], iacc);
        }
        facc += (float)iacc * __half2float(sr[g]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffff, facc, o);
    if (lane == 0) dst[n] = facc * aScale;
}
