// prefill_splitkv.cu — the exact batched-prefill attention past the single block's shared-memory bound.
//
// attn_batched (prefill_batched.cu) keeps one row's scores in shared memory, (nWin+128)*4 bytes, so it cannot serve a row
// that attends more keys than singleBlockAttnShmemLimit allows. These three kernels are decode_splitkv.cu's, with the
// query row on the grid's Z axis and the scores in a device buffer: row z of a launch is prompt row m = row0 + z, which
// attends nKeys = startPos + m + 1 keys from the window start attn_batched derives from that nKeys. Per (row, head) the
// instruction sequence is the decode kernel's, which is attn_batched's: the float4 K read with separate FMAs in d order,
// the 128-wide strided partition and ladder for the softmax denominator, and the whole ascending-key fold per output dim.
// Only the addressing of a row's q, score slice, inverse and ctx row is new, so the output is attn_batched's bit for bit
// (TestPrefillSplitKV_rowsEqualAttnBatched).
//
// Its own module, so no shipped PTX changes. Regenerate with:  ./build_ptx.sh prefill_splitkv
#include "kvtype.cuh"

// The row's key range, as attn_batched computes it, or as attn_img_batched computes it when blocks (nBlocks int pairs [start,end)) are
// given: a row whose own position lies inside a block sees that whole block, every other row stays causal, and the window start comes from the
// plain causal count, never the widened one (attn_img_prefill.cu's header says why). With nBlocks == 0 this is attn_batched's formula exactly.
#define ROW_RANGE(m)                                                                          \
    int nKeysCausal = startPos + (m) + 1;                                                     \
    int nKeys = nKeysCausal;                                                                  \
    for (int b_ = 0; b_ < nBlocks; b_++)                                                      \
        if (nKeysCausal - 1 >= blocks[2 * b_] && nKeysCausal - 1 < blocks[2 * b_ + 1]) {      \
            nKeys = blocks[2 * b_ + 1];                                                       \
            break;                                                                            \
        }                                                                                     \
    int winStart = (window > 0 && nKeysCausal > window) ? nKeysCausal - window : 0;           \
    int nWin = nKeys - winStart;

// splitkv_scores_rows: grid nH x ceil(maxNWin/128) x R, block 128, shared hd floats. One thread per key of one row.
// scStore is [R][nH][stride]; a row uses the first nWin entries of its slices.
extern "C" __global__ void splitkv_scores_rows(
    const float* __restrict__ q, const KV_T* __restrict__ kc,
    int nH, int nKV, int hd, int startPos, int window, int row0, float scale,
    float* __restrict__ scStore, int stride, const int* __restrict__ blocks, int nBlocks) {
    int h = blockIdx.x; if (h >= nH) return;   // block-uniform: safe before the barrier below
    int z = blockIdx.z, m = row0 + z;
    ROW_RANGE(m)
    // q is block-invariant: staged once in shared, as splitkv_scores does. Same values, same d order.
    extern __shared__ __align__(16) float qs[];
    const float* qh = q + (long)m * nH * hd + (long)h * hd;
    for (int t = threadIdx.x; t < hd; t += blockDim.x) qs[t] = qh[t];
    __syncthreads();   // before the divergent return below: every thread of the block reaches it

    int i = blockIdx.y * blockDim.x + threadIdx.x;   // window index
    int s = winStart + i;
    if (i >= nWin) return;
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    const KV_T* ks = kc + (long)s * kvDim + (long)kvh * hd;
    float dot = 0.f;
    int d4 = hd >> 2;
    const float4* q4 = (const float4*)qs;
    for (int j = 0; j < d4; j++) {
        float4 qq = q4[j], kk = KV_LD4V(ks, j);
        dot = __fmaf_rn(qq.x, kk.x, dot);
        dot = __fmaf_rn(qq.y, kk.y, dot);
        dot = __fmaf_rn(qq.z, kk.z, dot);
        dot = __fmaf_rn(qq.w, kk.w, dot);
    }
    for (int d = d4 << 2; d < hd; d++) dot = __fmaf_rn(qs[d], KV_LD(ks + d), dot);
    scStore[((long)z * nH + h) * stride + i] = dot * scale;
}

// splitkv_softmax_rows: grid nH x R, block 128 (it MUST be 128: the strided partition and the ladder below are
// attn_batched's only at that width), shared 128 floats. In place: a row's scores become exp(score - max), and
// invStore[z*nH + h] its 1/denominator. sinks is gpt-oss's per-head sink logit (it joins the max and the denominator), or NULL.
extern "C" __global__ void splitkv_softmax_rows(
    float* __restrict__ scStore, int nH, int startPos, int window, int row0, int stride,
    float* __restrict__ invStore, const float* __restrict__ sinks, const int* __restrict__ blocks, int nBlocks) {
    int h = blockIdx.x; if (h >= nH) return;
    int z = blockIdx.y, m = row0 + z;
    ROW_RANGE(m)
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float* sc = scStore + ((long)z * nH + h) * stride;
    float lm = -1e30f;
    for (int i = t; i < nWin; i += nt) lm = fmaxf(lm, sc[i]);
    // A max is exact in any order, so its tree is free; the identity is -1e30f because scores are freely negative.
    for (int o = 16; o > 0; o >>= 1) lm = fmaxf(lm, __shfl_down_sync(0xffffffff, lm, o));
    {
        int lane = t & 31, warp = t >> 5, nWarps = (nt + 31) >> 5;
        if (lane == 0) red[warp] = lm;
        __syncthreads();
        if (warp == 0) {
            float mv = (lane < nWarps) ? red[lane] : -1e30f;
            for (int o = 16; o > 0; o >>= 1) mv = fmaxf(mv, __shfl_down_sync(0xffffffff, mv, o));
            if (lane == 0) red[0] = mv;
        }
        __syncthreads();
    }
    float mx = red[0]; __syncthreads();
    float sink = 0.f; bool hasSink = (sinks != nullptr);
    if (hasSink) { sink = sinks[h]; mx = fmaxf(mx, sink); }
    float ls = 0.f;
    for (int i = t; i < nWin; i += nt) { float e = __expf(sc[i] - mx); sc[i] = e; ls += e; }
    red[t] = ls; __syncthreads();
    // The denominator is a float sum: this ladder is attn_batched's and must not be restructured.
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    if (t == 0) { float denom = red[0]; if (hasSink) denom += __expf(sink - mx); invStore[(long)z * nH + h] = 1.f / denom; }
}

// splitkv_vsum_rows: grid nH x ceil(hd/32) x R, block 32. Each thread owns one output dim of one row and folds
// sum_s sc[s]*v[s][d] in ascending s, the whole per-dim reduction, then scales by the row's inverse.
extern "C" __global__ void splitkv_vsum_rows(
    const float* __restrict__ scStore, const KV_T* __restrict__ vc,
    const float* __restrict__ invStore, int nH, int nKV, int hd, int startPos, int window, int row0,
    int stride, float* __restrict__ ctx, const int* __restrict__ blocks, int nBlocks) {
    int h = blockIdx.x; if (h >= nH) return;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= hd) return;
    int z = blockIdx.z, m = row0 + z;
    ROW_RANGE(m)
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    const float* sc = scStore + ((long)z * nH + h) * stride;
    float acc = 0.f;
#ifdef KV_F16
    // f16 cache: sixteen loads in flight before the sixteen FMAs, which stay in ascending-s order, as attn_batched does it.
    const KV_T* vp = vc + (long)kvh * hd + d;
    int s = winStart;
    for (; s + 16 <= nKeys; s += 16) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; j++) v[j] = KV_LD(vp + (long)(s + j) * kvDim);
#pragma unroll
        for (int j = 0; j < 16; j++) acc = __fmaf_rn(sc[s + j - winStart], v[j], acc);
    }
    for (; s < nKeys; s++) acc = __fmaf_rn(sc[s - winStart], KV_LD(vp + (long)s * kvDim), acc);
#else
    for (int s = winStart; s < nKeys; s++)
        acc = __fmaf_rn(sc[s - winStart], KV_LD(vc + ((long)s * kvDim + (long)kvh * hd + d)), acc);
#endif
    ctx[(long)m * nH * hd + (long)h * hd + d] = acc * invStore[(long)z * nH + h];
}
