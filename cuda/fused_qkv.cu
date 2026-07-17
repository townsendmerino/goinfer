#include <cuda_fp16.h>
// K1 (spec §5.2, super-kernel 1): rmsnorm+quant(x) fused into the Q/K/V GEMV.
//
// Why this wins: rmsnorm_quant runs at GridX:1 — ONE block, ~1/40th of the GPU — and sits
// serially in the chain because its int8 scale is a reduction over all of x[H]. Rather than
// multi-block it (impossible: the scale is global) we make every QKV block REDUNDANTLY
// recompute it into shared memory. x[H] is a few KB and L2-resident, so the redundant read
// is a broadcast hit and the redundant reduction is trivial next to the block's GEMV rows.
// The activation then lives in SHARED (faster than the L2 round-trip it replaces).
//
// Bit-identical to the separate kernels: same block size (256), same reduction order, same
// expression — so every block derives the same scale, equal to what rmsnorm_quant produced.
// All layer projections are int4 (measured), so there is a single weight path here.
//
// grid: (qDim+2*kvDim + 7)/8 blocks × 256 threads (8 warps, one output row per warp).
// A block never straddles a projection boundary (qDim/kvDim are multiples of 8).
extern "C" __global__ void fused_rms_qkv(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wq, const __half* __restrict__ gsq, const float* __restrict__ bq,
    const unsigned int* __restrict__ Wk, const __half* __restrict__ gsk, const float* __restrict__ bk,
    const unsigned int* __restrict__ Wv, const __half* __restrict__ gsv, const float* __restrict__ bv,
    int qDim, int kvDim, int Kwords, int Kgroups,
    float* __restrict__ qOut, float* __restrict__ kOut, float* __restrict__ vOut)
{
    extern __shared__ float sh[];
    float* normed = sh;              // [H]
    float* red    = normed + H;      // [256]
    int*   aq     = (int*)(red + 256); // [H/4] packed int8 activation
    int t = threadIdx.x, nt = blockDim.x;

    // ---- redundant rmsnorm + int8 quant of x (mirrors rmsnorm_quant exactly) ----
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss += x[k] * x[k];
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

    // ---- this warp's output row of Q | K | V ----
    int warp = t / 32, lane = t & 31;
    int r = blockIdx.x * (nt / 32) + warp;
    if (r >= qDim + 2 * kvDim) return;
    const unsigned int* W; const __half* gs; const float* bias; float* dst; int row;
    if (r < qDim)              { W = Wq; gs = gsq; bias = bq; dst = qOut; row = r; }
    else if (r < qDim + kvDim) { W = Wk; gs = gsk; bias = bk; dst = kOut; row = r - qDim; }
    else                       { W = Wv; gs = gsv; bias = bv; dst = vOut; row = r - qDim - kvDim; }

    const unsigned int* wr = W + (long)row * Kwords;
    const __half* sr = gs + (long)row * Kgroups;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {           // 2x ILP unroll (matches gemv_w4a8_fwd)
        int wi0 = base + lane, wi1 = base + 32 + lane;
        unsigned int w0 = wr[wi0], w1 = wr[wi1];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi0], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi0 + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi1], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi1 + 1], p1);
        facc += (float)p0 * __half2float(sr[wi0 >> 2]);
        facc += (float)p1 * __half2float(sr[wi1 >> 2]);
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi + 1], p);
            facc += (float)p * __half2float(sr[wi >> 2]);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
    if (lane == 0) dst[row] = facc * aScale + (bias ? bias[row] : 0.f);
}

// K3a (spec §5.2, super-kernel 3, first half): the pre-MLP rmsnorm+quant folded into the
// gate/up GEMV. Kills the layer's SECOND GridX:1 glue kernel.
//
// CRITICAL: redundant recompute costs PER BLOCK, so it only pays when the block count is
// modest. At 8 rows/block, gate/up (2*I = 17920 rows on the 1.5B) needs 2240 blocks = 2240
// redundant rmsnorms + ~13 MB of redundant x reads — measured an 18% REGRESSION. QKV only
// needs 256 blocks, which is why the same trick wins there. So each warp here walks
// ROWS_PER_WARP rows off ONE shared activation: 64 rows/block → ~280 blocks (1.5B) / ~152
// (0.5B) — still well above the 40 SMs, but 8× less redundancy.
#define ROWS_PER_WARP 8
extern "C" __global__ void fused_rms_gu(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wg, const __half* __restrict__ gsg,
    const unsigned int* __restrict__ Wu, const __half* __restrict__ gsu,
    int I, int Kwords, int Kgroups,
    float* __restrict__ gOut, float* __restrict__ uOut)
{
    extern __shared__ float sh[];
    float* normed = sh;
    float* red    = normed + H;
    int*   aq     = (int*)(red + 256);
    int t = threadIdx.x, nt = blockDim.x;

    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss += x[k] * x[k];
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
    int blockBase = blockIdx.x * (warps * ROWS_PER_WARP);
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        int r = blockBase + warp * ROWS_PER_WARP + i;   // this warp's i-th row
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
            facc += (float)p0 * __half2float(sr[wi0 >> 2]);
            facc += (float)p1 * __half2float(sr[wi1 >> 2]);
        }
        for (; base < Kwords; base += 32) {
            int wi = base + lane;
            if (wi < Kwords) {
                unsigned int word = wr[wi];
                int p = 0;
                p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi], p);
                p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), aq[2 * wi + 1], p);
                facc += (float)p * __half2float(sr[wi >> 2]);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
        if (lane == 0) dst[row] = facc * aScale;
    }
}

// qk_norm: per-head RMSNorm of Q and K over head_dim, BEFORE RoPE (Qwen3 / GLM / Mellum).
// Mirrors decoder/attention.go:94-96 → rmsNorm(q, QNorm, nH, hd, eps, addOne) and the same for
// k with KNorm, i.e. inv = 1/sqrt(ss/hd + eps); row[i] = (v*inv) * (addOne ? 1+w[i] : w[i]).
//
// One block per head: blocks [0,nH) normalise Q, [nH, nH+nKV) normalise K. hd is 64..256 in
// practice, so a single block reduction covers a head with no cross-block dependency — this is
// why it can sit between the QKV GEMV and rope_kv as its own cheap dispatch (and only for
// models that need it).
//
// NOTE the CPU accumulates the sum-of-squares in float64; this reduces in float32. The result
// is parity-green, not bit-identical — the same tiny divergence class as the rest of the
// resident path, far inside the 3%-near-tie rule.
extern "C" __global__ void qk_norm(
    float* __restrict__ q, float* __restrict__ k,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm,
    int nH, int nKV, int hd, float eps, int addOne)
{
    // Accumulate the sum-of-squares in DOUBLE and derive inv exactly as the CPU does:
    //   inv := float32(1.0 / math.Sqrt(ss/float64(dim) + eps))   (decoder/rmsnorm.go)
    // then apply in float32. QK-norm sits before RoPE and attention, so a float32 reduction
    // here propagates through the whole block — matching the CPU's f64 accumulation is what
    // keeps Qwen3 parity tight. hd is ~128, so the f64 reduction is negligible even at
    // sm_75's 1/32 double rate.
    extern __shared__ double red[];
    int h = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    float* base;
    const float* w;
    if (h < nH) { base = q + (long)h * hd;          w = qNorm; }
    else        { base = k + (long)(h - nH) * hd;   w = kNorm; }

    double ss = 0.0;
    for (int i = t; i < hd; i += nt) ss += (double)base[i] * (double)base[i];
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = (float)(1.0 / sqrt(red[0] / (double)hd + (double)eps));
    for (int i = t; i < hd; i += nt) {
        float g = addOne ? (1.f + w[i]) : w[i];
        base[i] = (base[i] * inv) * g;
    }
}
