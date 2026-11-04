// gumbel.cu — device-side temperature-only sampling by Gumbel-max (R7b, docs/tasks/red-october.md).
//
// The token drawn is  argmax_i ( l_i * invT + G_i ),  G_i = -ln(-ln1p(-w_i)),  w_i = (h_i + 0.5) / 2^32,
// where h_i is a word of Philox4x32-10 keyed by (seed) with counter (i>>2, draw_lo, draw_hi, 0), lane i&3.
// It is the same function as decoder.gumbelDraw (decoder/sampler_gumbel.go), which is the reference: the
// integer words are IDENTICAL on every backend, and the float transform here is f32 where the host's is f64,
// so the two can pick different tokens only when the two best keys are within a few ulps. That rate is
// measured (cuda TestGumbelDeviceAgreesWithHost), not assumed.
//
// A SEPARATE .cu on purpose (own PTX), like argmax.cu and topk.cu: nothing audited is regenerated.
//
// TWO KERNELS. gumbel_stage1: one thread per group of 4 tokens (one Philox call, four keys), block-level argmax,
// one (key, index) per block. gumbel_stage2: one block reduces those. Ties go to the LOWEST index, as on the host.
//
// THE NOISE, f32-SAFELY. w near 1 would make 1-w cancel in f32, so use the symmetry: let v = min(h, ~h) + 0.5;
// for h < 2^31, E = -log1p(-w) with w = v * 2^-32 (small, exact); otherwise 1-w = v * 2^-32 exactly and
// E = -ln(v * 2^-32) directly. Both branches are accurate to f32 rounding across the whole range.
//
// No FMA in the selection or the key: __fmul_rn / __fadd_rn are explicit, so the bits are a function of the source.

#define GB_THREADS 256
#define GB_INV32 2.3283064365386963e-10f // 2^-32
// NVRTC has no <math.h>: -FLT_MAX stands in for -inf as the "nothing yet" key (a real key of -inf is never chosen
// either way: it is not > best, so the index stays -1 exactly as the host's strict > on -Inf behaves).

extern "C" {

__device__ __forceinline__ unsigned int gb_mulhi(unsigned int a, unsigned int b) { return __umulhi(a, b); }

// Philox4x32-10. Must match decoder.philox4x32 (Random123 known-answer vectors).
__device__ __forceinline__ void gb_philox(unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
                                          unsigned int k0, unsigned int k1, unsigned int* out) {
#pragma unroll
    for (int r = 0; r < 10; r++) {
        if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        const unsigned int hi0 = gb_mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        const unsigned int hi1 = gb_mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        const unsigned int n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

__device__ __forceinline__ float gb_gumbel(unsigned int h) {
    float e;
    if (h < 0x80000000u) {
        const float w = __fmul_rn(__fadd_rn((float)h, 0.5f), GB_INV32);
        e = -log1pf(-w);
    } else {
        const float v = __fmul_rn(__fadd_rn((float)(~h), 0.5f), GB_INV32); // = 1 - w, exactly
        e = -logf(v);
    }
    return -logf(e);
}

// a beats b: larger key first; on an equal key the smaller index (idx < 0 means "none", which never wins).
__device__ __forceinline__ bool gb_better(float ak, int ai, float bk, int bi) {
    if (ai < 0) return false;
    if (bi < 0) return true;
    return ak > bk || (ak == bk && ai < bi);
}

__global__ void __launch_bounds__(GB_THREADS)
gumbel_stage1(const float* __restrict__ logits, int V, float invT, unsigned int k0, unsigned int k1,
              unsigned int d0, unsigned int d1, float* __restrict__ outKey, int* __restrict__ outIdx) {
    __shared__ float sk[GB_THREADS];
    __shared__ int si[GB_THREADS];
    const int t = threadIdx.x;
    const int b = blockIdx.x * GB_THREADS + t; // group of 4 consecutive tokens
    float best = -3.402823466e38f;
    int bi = -1;
    if (b * 4 < V) {
        unsigned int r[4];
        gb_philox((unsigned int)b, d0, d1, 0u, k0, k1, r);
#pragma unroll
        for (int lane = 0; lane < 4; lane++) {
            const int i = b * 4 + lane;
            if (i < V) {
                const float k = __fadd_rn(__fmul_rn(logits[i], invT), gb_gumbel(r[lane]));
                if (k > best) { best = k; bi = i; } // ascending index + strict >: lowest wins a tie
            }
        }
    }
    sk[t] = best; si[t] = bi;
    __syncthreads();
    for (int o = GB_THREADS >> 1; o > 0; o >>= 1) {
        if (t < o && gb_better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
        __syncthreads();
    }
    if (t == 0) { outKey[blockIdx.x] = sk[0]; outIdx[blockIdx.x] = si[0]; }
}

__global__ void __launch_bounds__(GB_THREADS)
gumbel_stage2(const float* __restrict__ inKey, const int* __restrict__ inIdx, int n, int* __restrict__ out) {
    __shared__ float sk[GB_THREADS];
    __shared__ int si[GB_THREADS];
    const int t = threadIdx.x;
    float best = -3.402823466e38f;
    int bi = -1;
    for (int i = t; i < n; i += GB_THREADS) {
        if (gb_better(inKey[i], inIdx[i], best, bi)) { best = inKey[i]; bi = inIdx[i]; }
    }
    sk[t] = best; si[t] = bi;
    __syncthreads();
    for (int o = GB_THREADS >> 1; o > 0; o >>= 1) {
        if (t < o && gb_better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
        __syncthreads();
    }
    if (t == 0) out[0] = si[0]; // -1 when nothing was comparable (an all -inf row): the caller falls back to argmax
}

} // extern "C"
