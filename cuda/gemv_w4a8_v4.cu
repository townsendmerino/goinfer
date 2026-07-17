#include <cuda_fp16.h>
// W4A8 GEMV, vectorized: each lane loads its group as ONE uint4 (16B = 4 words), so the
// warp reads 32 consecutive uint4 = 512 contiguous bytes (fully coalesced) AND each lane
// already owns a whole group — no segmented reduction, no per-word shfl. Unpack via even/odd
// + __vsub4. lane iterates groups lane, lane+32, ... dst[n] = aScale * Σg s[g]*Σ(nib-8)*a.
extern "C" __global__ void gemv_w4a8_v4(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    float aScale, int N, int Kwords, int Kgroups, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const uint4* wr4 = reinterpret_cast<const uint4*>(W + (long)n * Kwords);
    const __half* sr = gs + (long)n * Kgroups;
    float facc = 0.f;
    for (int g = lane; g < Kgroups; g += 32) {
        uint4 wv = wr4[g];                          // 4 words = this lane's whole group
        const int* ag = a + 8 * g;
        int iacc = 0;
        unsigned int ws[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            int es = __vsub4(ws[w] & 0x0F0F0F0Fu, 0x08080808u);
            int os = __vsub4((ws[w] >> 4) & 0x0F0F0F0Fu, 0x08080808u);
            iacc = __dp4a(es, ag[2 * w], iacc);
            iacc = __dp4a(os, ag[2 * w + 1], iacc);
        }
        facc += (float)iacc * __half2float(sr[g]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) dst[n] = facc * aScale;
}
