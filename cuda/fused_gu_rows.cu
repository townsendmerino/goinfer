#include <cuda_fp16.h>
// fused_rms_gu_rows: fused_qkv.cu's fused_rms_gu with ROWS-PER-WARP as a runtime parameter (the original hard-codes 8). Same idea as fused_rms_qkv_rows: a warp walking more rows
// off one shared activation means fewer blocks repeat the rmsnorm + int8-quantisation prologue. BIT-IDENTICAL BY CONSTRUCTION: the prologue is a verbatim copy and each
// output row's arithmetic is the original's per-warp sequence; only which warp computes which row changes. rowsPerWarp = 8 is the original kernel.
extern "C" __global__ void fused_rms_gu_rows(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wg, const __half* __restrict__ gsg,
    const unsigned int* __restrict__ Wu, const __half* __restrict__ gsu,
    int I, int Kwords, int Kgroups, int rowsPerWarp,
    float* __restrict__ gOut, float* __restrict__ uOut)
{
    extern __shared__ float sh[];
    float* normed = sh;
    float* red    = normed + H;
    int*   aq     = (int*)(red + 256);
    int t = threadIdx.x, nt = blockDim.x;

    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss = __fmaf_rn(x[k], x[k], ss);
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = rsqrtf(red[0] / H + eps); __syncthreads();
    float ma = 0.f;
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + nrm[k]) : nrm[k]; float v = x[k] * g * rnorm; normed[k] = v; ma = fmaxf(ma, fabsf(v)); }
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float aScale = red[0] / 127.f;
    float inv = aScale > 0.f ? 1.f / aScale : 0.f;
    for (int j = t; j < H / 4; j += nt) {
        int packed = 0;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int q = __float2int_rn(normed[4 * j + b] * inv);
            q = max(-127, min(127, q));
            packed |= (q & 0xff) << (8 * b);
        }
        aq[j] = packed;
    }
    __syncthreads();

    int warp = t / 32, lane = t & 31, warps = nt / 32;
    int blockBase = blockIdx.x * (warps * rowsPerWarp);
    for (int i = 0; i < rowsPerWarp; i++) {
        int r = blockBase + warp * rowsPerWarp + i;   // this warp's i-th row
        if (r >= 2 * I) return;
        const unsigned int* W; const __half* gs; float* dst; int row;
        if (r < I) { W = Wg; gs = gsg; dst = gOut; row = r; }
        else       { W = Wu; gs = gsu; dst = uOut; row = r - I; }
        const unsigned int* wr = W + (long)row * Kwords;
        const __half* sr = gs + (long)row * Kgroups;
        float facc = 0.f;
        int base = 0;
        for (; base + 64 <= Kwords; base += 64) {
            int wi0 = base + lane, wi1 = base + 32 + lane;
            unsigned int w0 = wr[wi0], w1 = wr[wi1];
            int p0 = 0, p1 = 0;
            p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi0], p0);
            p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi0 + 1], p0);
            p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi1], p1);
            p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi1 + 1], p1);
            facc = __fmaf_rn((float)p0, __half2float(sr[wi0 >> 2]), facc);
            facc = __fmaf_rn((float)p1, __half2float(sr[wi1 >> 2]), facc);
        }
        for (; base < Kwords; base += 32) {
            int wi = base + lane;
            if (wi < Kwords) {
                unsigned int word = wr[wi];
                int p = 0;
                p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi], p);
                p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi + 1], p);
                facc = __fmaf_rn((float)p, __half2float(sr[wi >> 2]), facc);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
        if (lane == 0) dst[row] = __fmul_rn(facc, aScale);
    }
}

