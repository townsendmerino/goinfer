#include <cuda_fp16.h>
// fused_rms_qkv_rows: fused_qkv.cu's fused_rms_qkv with ROWS-PER-WARP as a runtime parameter (docs/measurements/fused-rms-qkv-2026-09-21.md).
//
// fused_rms_qkv gives each warp ONE output row, so a block of 8 warps serves 8 rows and EVERY block first redundantly recomputes the layer's rmsnorm and int8
// quantisation of x[H] (three block-wide reductions, ~16 barriers) before it reads a single weight. On D7 (4608 rows) that is 576 blocks paying the prologue, and the
// kernel ran at ~163 GB/s. Letting each warp walk `rowsPerWarp` rows off ONE shared activation divides the redundant prologues by rowsPerWarp
// (fused_rms_gu already does this, with 8).
//
// BIT-IDENTICAL BY CONSTRUCTION: the prologue is a verbatim copy (same block size 256, same reduction order, same expressions, so every block derives the same scale
// as the original), and each output row's arithmetic is exactly the original's per-warp sequence; only WHICH warp computes which row changes. rowsPerWarp = 1
// is the original kernel. A block covers (nt/32)*rowsPerWarp consecutive rows; it may straddle the Q|K|V boundaries, which the per-row selection below handles.
extern "C" __global__ void fused_rms_qkv_rows(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wq, const __half* __restrict__ gsq, const float* __restrict__ bq,
    const unsigned int* __restrict__ Wk, const __half* __restrict__ gsk, const float* __restrict__ bk,
    const unsigned int* __restrict__ Wv, const __half* __restrict__ gsv, const float* __restrict__ bv,
    int qDim, int kvDim, int Kwords, int Kgroups, int rowsPerWarp,
    float* __restrict__ qOut, float* __restrict__ kOut, float* __restrict__ vOut)
{
    extern __shared__ float sh[];
    float* normed = sh;              // [H]
    float* red    = normed + H;      // [256]
    int*   aq     = (int*)(red + 256); // [H/4] packed int8 activation
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

    int warp = t / 32, lane = t & 31;
    const int total = qDim + 2 * kvDim;
    const int blockBase = blockIdx.x * ((nt / 32) * rowsPerWarp);
    for (int i = 0; i < rowsPerWarp; i++) {
        int r = blockBase + warp * rowsPerWarp + i;
        if (r >= total) return;
        const unsigned int* W; const __half* gs; const float* bias; float* dst; int row;
        if (r < qDim)              { W = Wq; gs = gsq; bias = bq; dst = qOut; row = r; }
        else if (r < qDim + kvDim) { W = Wk; gs = gsk; bias = bk; dst = kOut; row = r - qDim; }
        else                       { W = Wv; gs = gsv; bias = bv; dst = vOut; row = r - qDim - kvDim; }

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
        if (lane == 0) dst[row] = __fmaf_rn(facc, aScale, (bias ? bias[row] : 0.f));
    }
}
