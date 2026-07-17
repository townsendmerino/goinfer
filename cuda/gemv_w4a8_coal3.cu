#include <cuda_fp16.h>
// W4A8 GEMV, COAL3: COAL2 + 2x ILP unroll. int4 reads half the bytes of int8, so it needs
// more loads in flight to saturate bandwidth. Each lane issues two independent word loads
// (base+lane and base+32+lane) before consuming them; a 32-stride remainder loop handles
// the tail (dims are only guaranteed Kwords % 32 == 0).
extern "C" __global__ void gemv_w4a8_coal3(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    float aScale, int N, int Kwords, int Kgroups, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi0 = base + lane, wi1 = base + 32 + lane;
        unsigned int w0 = wr[wi0];              // two independent loads in flight
        unsigned int w1 = wr[wi1];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0 + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1 + 1], p1);
        facc += (float)p0 * __half2float(sr[wi0 >> 2]);
        facc += (float)p1 * __half2float(sr[wi1 >> 2]);
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        unsigned int word = wr[wi];
        int p = 0;
        p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
        p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
        facc += (float)p * __half2float(sr[wi >> 2]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) dst[n] = facc * aScale;
}
