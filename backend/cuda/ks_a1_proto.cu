// ks_a1_proto.cu — KS-A1's first measurement on CUDA (docs/tasks/task-kernel-survey-2026-10.md, "KS-A1 on CUDA"): prototype
// native Q5_K and Q6_K GEMVs against the shipped per-32 int8 kernel. Test-only: TestKSA1_gemvTiming loads this module and
// nothing shipped does. Activations are the shipped per-32 form (actgroup.cu's quantG32): packed int8 codes a, scales aS[K/32],
// then the groups' dequantized sums aS[K/32 + g].

#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int p_q4kByte(int k, unsigned int s0, unsigned int s1, unsigned int s2) {
    unsigned int w = k < 4 ? s0 : (k < 8 ? s1 : s2);
    return (w >> ((k & 3) << 3)) & 0xffu;
}

// p_scaleMin is ggml's get_scale_min_k4 for sub-block j (0..7), the Q4_K and Q5_K scale layout.
__device__ __forceinline__ void p_scaleMin(int j, unsigned int s0, unsigned int s1, unsigned int s2, int* sc, int* m) {
    if (j < 4) {
        *sc = (int)(p_q4kByte(j, s0, s1, s2) & 63u);
        *m = (int)(p_q4kByte(j + 4, s0, s1, s2) & 63u);
    } else {
        unsigned int hi = p_q4kByte(j + 4, s0, s1, s2);
        *sc = (int)((hi & 0x0Fu) | ((p_q4kByte(j - 4, s0, s1, s2) >> 6) << 4));
        *m = (int)((hi >> 4) | ((p_q4kByte(j, s0, s1, s2) >> 6) << 4));
    }
}

__device__ __forceinline__ float p_warpSum(float f) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) f += __shfl_down_sync(0xffffffffu, f, off);
    return f;
}

__device__ __forceinline__ float p_half(unsigned int bits16) {
    return __half2float(__ushort_as_half((unsigned short)(bits16 & 0xffffu)));
}

// Q5_K value bytes of one code word: four 4-bit codes plus their high bits (bit `shift` of each qh byte).
__device__ __forceinline__ int q5lo(unsigned int code, unsigned int qhw, int shift) {
    return (int)((code & 0x0F0F0F0Fu) | (((qhw >> shift) & 0x01010101u) << 4));
}
__device__ __forceinline__ int q5hi(unsigned int code, unsigned int qhw, int shift) {
    return (int)(((code >> 4) & 0x0F0F0F0Fu) | (((qhw >> shift) & 0x01010101u) << 4));
}

// Q5_K block: 44 words. Word 0 d|dmin, words 1-3 the 12 scale bytes, words 4-11 qh[32], words 12-43 qs[128].
// Lane map A (rowQ4K's): lane l on pair j = l & 3 of block b0 + (l >> 2), 64 weights, all 32 qh bytes.
template <int R>
__device__ __forceinline__ void rowsQ5K_A(const unsigned int* __restrict__ W, int n0, int N, const int* __restrict__ a,
                                          const float* __restrict__ aS, const float* __restrict__ asum, int nSB, int lane,
                                          float* facc) {
    int j = lane & 3, sub = lane >> 2;
    for (int b0 = 0; b0 < nSB; b0 += 8) {
        int b = b0 + sub;
        if (b >= nSB) continue;
        const int4* av = reinterpret_cast<const int4*>(a + (b << 6) + (j << 4));
        int4 l0 = av[0], l1 = av[1], h0 = av[2], h1 = av[3];
        int g = (b << 3) + (j << 1);
        float aLo = aS[g], aHi = aS[g + 1], sLo = asum[g], sHi = asum[g + 1];
        int lo[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        int hi[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int n = n0 + r;
            if (n >= N) break;
            const unsigned int* blk = W + ((long)n * nSB + b) * 44;
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            uint4 qh0 = *reinterpret_cast<const uint4*>(blk + 4);
            uint4 qh1 = *reinterpret_cast<const uint4*>(blk + 8);
            const uint4* qv = reinterpret_cast<const uint4*>(blk + 12 + (j << 3));
            uint4 q0 = qv[0], q1 = qv[1];
            float d = p_half(hdr.x), dmin = p_half(hdr.x >> 16);
            int scLo, mLo, scHi, mHi;
            p_scaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
            p_scaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
            unsigned int cw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            unsigned int hw[8] = {qh0.x, qh0.y, qh0.z, qh0.w, qh1.x, qh1.y, qh1.z, qh1.w};
            int sl = j << 1, sh = sl + 1;
            int pLo = 0, pHi = 0;
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                pLo = __dp4a(q5lo(cw[t], hw[t], sl), lo[t], pLo);
                pHi = __dp4a(q5hi(cw[t], hw[t], sh), hi[t], pHi);
            }
            float f = facc[r];
            f = __fmaf_rn((float)pLo, __fmul_rn(__fmul_rn(d, (float)scLo), aLo), f);
            f = __fmaf_rn((float)pHi, __fmul_rn(__fmul_rn(d, (float)scHi), aHi), f);
            f = __fmaf_rn(-__fmul_rn(dmin, (float)mLo), sLo, f);
            f = __fmaf_rn(-__fmul_rn(dmin, (float)mHi), sHi, f);
            facc[r] = f;
        }
    }
}

// Lane map B: 8 lanes a block, 4 blocks in flight. Lane l: block b0 + (l >> 3), pair j = (l & 7) >> 1, half h = l & 1:
// 16 code bytes (pair j's bytes 16h..16h+15: elements 64j+16h.. low, 64j+32+16h.. high), the matching 16 qh bytes, and the
// min term of the low group (h = 0) or the high group (h = 1), so each group's min is added once.
template <int R>
__device__ __forceinline__ void rowsQ5K_B(const unsigned int* __restrict__ W, int n0, int N, const int* __restrict__ a,
                                          const float* __restrict__ aS, const float* __restrict__ asum, int nSB, int lane,
                                          float* facc) {
    int sub = lane >> 3, j = (lane & 7) >> 1, h = lane & 1;
    for (int b0 = 0; b0 < nSB; b0 += 4) {
        int b = b0 + sub;
        if (b >= nSB) continue;
        int4 lo = *reinterpret_cast<const int4*>(a + (b << 6) + (j << 4) + (h << 2));
        int4 hi = *reinterpret_cast<const int4*>(a + (b << 6) + (j << 4) + 8 + (h << 2));
        int g = (b << 3) + (j << 1);
        float aLo = aS[g], aHi = aS[g + 1], sMin = asum[g + h];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int n = n0 + r;
            if (n >= N) break;
            const unsigned int* blk = W + ((long)n * nSB + b) * 44;
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            uint4 qh = *reinterpret_cast<const uint4*>(blk + 4 + (h << 2));
            uint4 q = *reinterpret_cast<const uint4*>(blk + 12 + (j << 3) + (h << 2));
            float d = p_half(hdr.x), dmin = p_half(hdr.x >> 16);
            int scLo, mLo, scHi, mHi;
            p_scaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
            p_scaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
            int sl = j << 1, sh = sl + 1;
            int pLo = 0, pHi = 0;
            pLo = __dp4a(q5lo(q.x, qh.x, sl), lo.x, pLo);
            pLo = __dp4a(q5lo(q.y, qh.y, sl), lo.y, pLo);
            pLo = __dp4a(q5lo(q.z, qh.z, sl), lo.z, pLo);
            pLo = __dp4a(q5lo(q.w, qh.w, sl), lo.w, pLo);
            pHi = __dp4a(q5hi(q.x, qh.x, sh), hi.x, pHi);
            pHi = __dp4a(q5hi(q.y, qh.y, sh), hi.y, pHi);
            pHi = __dp4a(q5hi(q.z, qh.z, sh), hi.z, pHi);
            pHi = __dp4a(q5hi(q.w, qh.w, sh), hi.w, pHi);
            int m = h ? mHi : mLo;
            float f = facc[r];
            f = __fmaf_rn((float)pLo, __fmul_rn(__fmul_rn(d, (float)scLo), aLo), f);
            f = __fmaf_rn((float)pHi, __fmul_rn(__fmul_rn(d, (float)scHi), aHi), f);
            f = __fmaf_rn(-__fmul_rn(dmin, (float)m), sMin, f);
            facc[r] = f;
        }
    }
}

// Q6_K over segment-wide planes: ql[nb·128], qh[nb·64], sc[nb·16] (int8), d[nb] (f16), nb = n·nSB + b.
// Lane k = l & 7 of block b0 + (l >> 3): half hh = k >> 2, elements l0 = 8·(k & 3)…+7 of each of the half's four
// 32-element groups. Codes are (ql nibble | 2 qh bits << 4) − 32; the group's sub-scale (one per 16 elements) multiplies
// the integer dot, then × d × the group's activation scale.
template <int R>
__device__ __forceinline__ void rowsQ6K(const unsigned char* __restrict__ ql, const unsigned char* __restrict__ qhp,
                                        const signed char* __restrict__ scp, const unsigned short* __restrict__ dp,
                                        int n0, int N, const int* __restrict__ a, const float* __restrict__ aS, int nSB,
                                        int lane, float* facc) {
    int sub = lane >> 3, k = lane & 7, hh = k >> 2, q = k & 3, l0 = q << 3;
    for (int b0 = 0; b0 < nSB; b0 += 4) {
        int b = b0 + sub;
        if (b >= nSB) continue;
        const int* ab = a + (b << 6) + (hh << 5) + (q << 1);
        int2 a0 = *reinterpret_cast<const int2*>(ab);
        int2 a1 = *reinterpret_cast<const int2*>(ab + 8);
        int2 a2 = *reinterpret_cast<const int2*>(ab + 16);
        int2 a3 = *reinterpret_cast<const int2*>(ab + 24);
        int g = (b << 3) + (hh << 2);
        float s0 = aS[g], s1 = aS[g + 1], s2 = aS[g + 2], s3 = aS[g + 3];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int n = n0 + r;
            if (n >= N) break;
            long nb = (long)n * nSB + b;
            uint2 qa = *reinterpret_cast<const uint2*>(ql + nb * 128 + (hh << 6) + l0);
            uint2 qb = *reinterpret_cast<const uint2*>(ql + nb * 128 + (hh << 6) + 32 + l0);
            uint2 qhv = *reinterpret_cast<const uint2*>(qhp + nb * 64 + (hh << 5) + l0);
            uint2 scw = *reinterpret_cast<const uint2*>(scp + nb * 16 + (hh << 3));
            float d = p_half(dp[nb]);
            int is = q >> 1;
            // sub-scales sc[8hh + is + 2·grp], grp = 0..3: bytes is, is+2 of scw.x and is, is+2 of scw.y.
            int c0 = (int)(signed char)((scw.x >> (is << 3)) & 0xffu);
            int c1 = (int)(signed char)((scw.x >> ((is + 2) << 3)) & 0xffu);
            int c2 = (int)(signed char)((scw.y >> (is << 3)) & 0xffu);
            int c3 = (int)(signed char)((scw.y >> ((is + 2) << 3)) & 0xffu);
            int v0x = (int)__vsub4((qa.x & 0x0F0F0F0Fu) | ((qhv.x & 0x03030303u) << 4), 0x20202020u);
            int v0y = (int)__vsub4((qa.y & 0x0F0F0F0Fu) | ((qhv.y & 0x03030303u) << 4), 0x20202020u);
            int v1x = (int)__vsub4((qb.x & 0x0F0F0F0Fu) | (((qhv.x >> 2) & 0x03030303u) << 4), 0x20202020u);
            int v1y = (int)__vsub4((qb.y & 0x0F0F0F0Fu) | (((qhv.y >> 2) & 0x03030303u) << 4), 0x20202020u);
            int v2x = (int)__vsub4(((qa.x >> 4) & 0x0F0F0F0Fu) | (((qhv.x >> 4) & 0x03030303u) << 4), 0x20202020u);
            int v2y = (int)__vsub4(((qa.y >> 4) & 0x0F0F0F0Fu) | (((qhv.y >> 4) & 0x03030303u) << 4), 0x20202020u);
            int v3x = (int)__vsub4(((qb.x >> 4) & 0x0F0F0F0Fu) | (((qhv.x >> 6) & 0x03030303u) << 4), 0x20202020u);
            int v3y = (int)__vsub4(((qb.y >> 4) & 0x0F0F0F0Fu) | (((qhv.y >> 6) & 0x03030303u) << 4), 0x20202020u);
            int p0 = __dp4a(v0y, a0.y, __dp4a(v0x, a0.x, 0));
            int p1 = __dp4a(v1y, a1.y, __dp4a(v1x, a1.x, 0));
            int p2 = __dp4a(v2y, a2.y, __dp4a(v2x, a2.x, 0));
            int p3 = __dp4a(v3y, a3.y, __dp4a(v3x, a3.x, 0));
            float f = facc[r];
            f = __fmaf_rn((float)(c0 * p0), __fmul_rn(d, s0), f);
            f = __fmaf_rn((float)(c1 * p1), __fmul_rn(d, s1), f);
            f = __fmaf_rn((float)(c2 * p2), __fmul_rn(d, s2), f);
            f = __fmaf_rn((float)(c3 * p3), __fmul_rn(d, s3), f);
            facc[r] = f;
        }
    }
}

template <int R>
__device__ __forceinline__ void storeRows(const float* facc, int n0, int N, int lane, const float* __restrict__ bias,
                                          float* __restrict__ dst, int accum) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = p_warpSum(facc[r]);
        int n = n0 + r;
        if (lane == 0 && n < N) {
            float val = __fadd_rn(s, (bias ? bias[n] : 0.f));
            dst[n] = accum ? dst[n] + val : val;
        }
    }
}

#define Q5K_KERNEL(NAME, BODY, R)                                                                                        \
    extern "C" __global__ void NAME(const unsigned int* __restrict__ W, const int* __restrict__ a,                       \
                                    const float* __restrict__ aS, const float* __restrict__ bias, int N, int nSB,        \
                                    float* __restrict__ dst, int accum) {                                                \
        int lane = threadIdx.x & 31;                                                                                     \
        int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;                                              \
        if (n0 >= N) return;                                                                                             \
        float facc[R];                                                                                                   \
        _Pragma("unroll") for (int r = 0; r < R; r++) facc[r] = 0.f;                                                     \
        BODY<R>(W, n0, N, a, aS, aS + nSB * 8, nSB, lane, facc);                                                         \
        storeRows<R>(facc, n0, N, lane, bias, dst, accum);                                                               \
    }

Q5K_KERNEL(gemv_q5k_a_r1, rowsQ5K_A, 1)
Q5K_KERNEL(gemv_q5k_a_r2, rowsQ5K_A, 2)
Q5K_KERNEL(gemv_q5k_a_r4, rowsQ5K_A, 4)
Q5K_KERNEL(gemv_q5k_b_r1, rowsQ5K_B, 1)
Q5K_KERNEL(gemv_q5k_b_r2, rowsQ5K_B, 2)
Q5K_KERNEL(gemv_q5k_b_r4, rowsQ5K_B, 4)

#define Q6K_KERNEL(NAME, R)                                                                                              \
    extern "C" __global__ void NAME(const unsigned char* __restrict__ ql, const unsigned char* __restrict__ qh,           \
                                    const signed char* __restrict__ sc, const unsigned short* __restrict__ d,            \
                                    const int* __restrict__ a, const float* __restrict__ aS,                             \
                                    const float* __restrict__ bias, int N, int nSB, float* __restrict__ dst, int accum) { \
        int lane = threadIdx.x & 31;                                                                                     \
        int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;                                              \
        if (n0 >= N) return;                                                                                             \
        float facc[R];                                                                                                   \
        _Pragma("unroll") for (int r = 0; r < R; r++) facc[r] = 0.f;                                                     \
        rowsQ6K<R>(ql, qh, sc, d, n0, N, a, aS, nSB, lane, facc);                                                        \
        storeRows<R>(facc, n0, N, lane, bias, dst, accum);                                                               \
    }

Q6K_KERNEL(gemv_q6k_r1, 1)
Q6K_KERNEL(gemv_q6k_r2, 2)
Q6K_KERNEL(gemv_q6k_r4, 4)

// stream_read: the streaming ceiling, every 16-byte word of buf read once (grid-stride); a block writes only if its
// xor is a planted sentinel, so the loads cannot be dropped.
extern "C" __global__ void stream_read(const uint4* __restrict__ buf, long nWords, unsigned int* __restrict__ out) {
    unsigned int x = 0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < nWords; i += (long)gridDim.x * blockDim.x) {
        uint4 v = buf[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9E3779B9u) out[blockIdx.x] = x;
}
