#include <cuda_fp16.h>
// W4A8 GEMV, COAL4: 4x ILP unroll (four independent word loads in flight per lane) + a
// 32-stride remainder. Pushes memory-level parallelism further to saturate bandwidth on the
// byte-light int4 stream.
extern "C" __global__ void gemv_w4a8_coal4(
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
    for (; base + 128 <= Kwords; base += 128) {
        int wi0 = base + lane, wi1 = base + 32 + lane, wi2 = base + 64 + lane, wi3 = base + 96 + lane;
        unsigned int w0 = wr[wi0], w1 = wr[wi1], w2 = wr[wi2], w3 = wr[wi3];  // 4 loads in flight
        int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0 + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1 + 1], p1);
        p2 = __dp4a((int)__vsub4(w2 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi2], p2);
        p2 = __dp4a((int)__vsub4((w2 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi2 + 1], p2);
        p3 = __dp4a((int)__vsub4(w3 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi3], p3);
        p3 = __dp4a((int)__vsub4((w3 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi3 + 1], p3);
        facc += (float)p0 * __half2float(sr[wi0 >> 2]);
        facc += (float)p1 * __half2float(sr[wi1 >> 2]);
        facc += (float)p2 * __half2float(sr[wi2 >> 2]);
        facc += (float)p3 * __half2float(sr[wi3 >> 2]);
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
