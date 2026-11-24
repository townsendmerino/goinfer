// prefill_g32.cu — the batched prefill pass's kernels for per-32 activation quantization (the q4k lane).
//
// Each kernel is the M-row twin of a per-32 decode kernel in actgroup.cu, which this file includes for its device functions
// (quantG32, q4kScaleMin, q5kLo/q5kHi, warpSum) with that module's kernels compiled out. Every (output row, activation row)
// pair runs its decode kernel's arithmetic in the same lane order with the same explicit __fmaf_rn / __fmul_rn, and is
// warp-reduced once after all K, so each output equals the decode kernel's to the bit (docs/kernel-bit-identity.md).
//
// LAYOUT. Activation row m is K/4 packed ints at A + m·K/4 and 2·K/32 floats at aS + m·2·K/32: the per-group scales, then the
// per-group sums (quantG32; the resident runner's actScaleLen). Outputs are [M, N] row-major.
//
// THE ROWS GEMVS hold RN output rows' weights for one K step in registers and apply them to an MT tile of activation rows, so a
// weight is loaded once a tile instead of once a row. The quantizers are one block per row, decode's launch shape.

#define ACTGROUP_DEVICE_ONLY
#include "actgroup.cu"

#ifndef RN
#define RN 2
#endif
#ifndef MT
#define MT 8
#endif

extern "C" {

// rmsnorm_quant_g32_batched: rmsnorm_quant_g32 per row, grid.x = M (blockDim 256, the decode ladder). Shared: N + blockDim floats.
__global__ void rmsnorm_quant_g32_batched(const float* __restrict__ x, const float* __restrict__ w,
                                          int N, float eps, int addOne, int* __restrict__ aq, float* __restrict__ aScale) {
    extern __shared__ float sh[];
    int m = blockIdx.x;
    const float* xr = x + (long)m * N;
    float* normed = sh;
    float* red = sh + N;
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int k = t; k < N; k += nt) ss = __fmaf_rn(xr[k], xr[k], ss);
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = 1.f / sqrtf(red[0] / N + eps); __syncthreads();
    for (int k = t; k < N; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; normed[k] = xr[k] * g * rnorm; }
    __syncthreads();
    quantG32(normed, N / 32, aq + (long)m * (N / 4), aScale + (long)m * 2 * (N / 32));
}

// quant_vec_g32_batched: quant_vec_g32 per row, grid.x = M. No shared memory is read.
__global__ void quant_vec_g32_batched(const float* __restrict__ x, int N, int* __restrict__ q, float* __restrict__ scale, int M) {
    int m = blockIdx.x;
    quantG32(x + (long)m * N, N / 32, q + (long)m * (N / 4), scale + (long)m * 2 * (N / 32));
}

// glu_quant_g32_batched: glu_quant_g32 per row, grid.x = M, with glu_quant_batched's argument list. g is [M, I], and u is [M, I]
// row-major from u + uOff (row stride I, as glu_quant_batched reads it), so a gate‖up buffer of row stride 2I is not a valid u.
// The product act(g)·u is written back over the gate row (glu_quant_batched's convention; g carries no __restrict__) and
// quantized after the barrier, which makes the block's global writes visible to the quantizing threads.
__global__ void glu_quant_g32_batched(float* g, const float* __restrict__ u, int uOff, int I, int act,
                                      int* __restrict__ q, float* __restrict__ scale, int M) {
    int m = blockIdx.x;
    float* gr = g + (long)m * I;
    const float* ur = u + (long)m * I + uOff;
    int t = threadIdx.x, nt = blockDim.x;
    for (int k = t; k < I; k += nt) {
        float x = gr[k], a;
        if (act == ACT_SILU) {
            a = x / (1.f + expf(-x));
        } else {
            a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x))));
        }
        gr[k] = a * ur[k];
    }
    __syncthreads();
    quantG32(gr, I / 32, q + (long)m * (I / 4), scale + (long)m * 2 * (I / 32));
}

// rowsEpilogue reduces facc[r][t] and writes dst[m, n] for this warp's RN rows and the tile's rows: decode's epilogue per element.
// wScale is the int8 row scales (finish = sum·wScale[n] + bias) or null (sum + bias).
__device__ __forceinline__ void rowsEpilogue(float (&facc)[RN][MT], int n0, int N, int m0, int mcnt, int lane,
                                             const float* __restrict__ wScale, const float* __restrict__ bias,
                                             float* __restrict__ dst, int accum) {
    #pragma unroll
    for (int r = 0; r < RN; r++) {
        #pragma unroll
        for (int t = 0; t < MT; t++) {
            if (t < mcnt) {
                float s = warpSum(facc[r][t]);
                int n = n0 + r;
                if (lane == 0 && n < N) {
                    float val = wScale ? __fmaf_rn(s, wScale[n], (bias ? bias[n] : 0.f)) : __fadd_rn(s, (bias ? bias[n] : 0.f));
                    long o = (long)(m0 + t) * N + n;
                    dst[o] = accum ? dst[o] + val : val;
                }
            }
        }
    }
}

// gemv_q4k_g32_rows: gemv_q4k_g32 over M activation rows (rowQ4K's lane map: lane l on sub-block pair j = l & 3 of block
// b0 + (l >> 2), 8 blocks a pass). RN rows a warp, 8 warps a 256-thread block.
__global__ void gemv_q4k_g32_rows(const unsigned int* __restrict__ W, const int* __restrict__ A, const float* __restrict__ aS,
                                  const float* __restrict__ bias, int N, int nSB, int M, float* __restrict__ dst, int accum) {
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RN;
    if (n0 >= N) return;
    int j = lane & 3, sub = lane >> 2;
    int Kw = nSB * 64, nG = nSB * 8;
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = min(MT, M - m0);
        float facc[RN][MT];
        #pragma unroll
        for (int r = 0; r < RN; r++)
            #pragma unroll
            for (int t = 0; t < MT; t++) facc[r][t] = 0.f;
        for (int b0 = 0; b0 < nSB; b0 += 8) {
            int b = b0 + sub;
            if (b >= nSB) continue;
            int lo[RN][8], hi[RN][8];
            float sLo[RN], sHi[RN], mnLo[RN], mnHi[RN];
            #pragma unroll
            for (int r = 0; r < RN; r++) {
                int n = min(n0 + r, N - 1); // an out-of-range row computes row N-1 and is not written
                const unsigned int* blk = W + ((long)n * nSB + b) * 36;
                uint4 hdr = *reinterpret_cast<const uint4*>(blk);
                float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xffffu)));
                float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
                int scLo, mLo, scHi, mHi;
                q4kScaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
                q4kScaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
                sLo[r] = __fmul_rn(d, (float)scLo);
                sHi[r] = __fmul_rn(d, (float)scHi);
                mnLo[r] = -__fmul_rn(dmin, (float)mLo);
                mnHi[r] = -__fmul_rn(dmin, (float)mHi);
                const uint4* qv = reinterpret_cast<const uint4*>(blk + 4 + (j << 3));
                uint4 q0 = qv[0], q1 = qv[1];
                unsigned int c[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    lo[r][i] = (int)(c[i] & 0x0F0F0F0Fu);
                    hi[r][i] = (int)((c[i] >> 4) & 0x0F0F0F0Fu);
                }
            }
            int g = (b << 3) + (j << 1);
            #pragma unroll
            for (int t = 0; t < MT; t++) {
                if (t < mcnt) {
                    int m = m0 + t;
                    const int4* av = reinterpret_cast<const int4*>(A + (long)m * Kw + (b << 6) + (j << 4));
                    int4 l0 = av[0], l1 = av[1], h0 = av[2], h1 = av[3];
                    const float* as = aS + (long)m * 2 * nG;
                    float a0 = as[g], a1 = as[g + 1], u0 = as[nG + g], u1 = as[nG + g + 1];
                    #pragma unroll
                    for (int r = 0; r < RN; r++) {
                        int pLo = 0, pHi = 0;
                        pLo = __dp4a(lo[r][0], l0.x, pLo);
                        pLo = __dp4a(lo[r][1], l0.y, pLo);
                        pLo = __dp4a(lo[r][2], l0.z, pLo);
                        pLo = __dp4a(lo[r][3], l0.w, pLo);
                        pLo = __dp4a(lo[r][4], l1.x, pLo);
                        pLo = __dp4a(lo[r][5], l1.y, pLo);
                        pLo = __dp4a(lo[r][6], l1.z, pLo);
                        pLo = __dp4a(lo[r][7], l1.w, pLo);
                        pHi = __dp4a(hi[r][0], h0.x, pHi);
                        pHi = __dp4a(hi[r][1], h0.y, pHi);
                        pHi = __dp4a(hi[r][2], h0.z, pHi);
                        pHi = __dp4a(hi[r][3], h0.w, pHi);
                        pHi = __dp4a(hi[r][4], h1.x, pHi);
                        pHi = __dp4a(hi[r][5], h1.y, pHi);
                        pHi = __dp4a(hi[r][6], h1.z, pHi);
                        pHi = __dp4a(hi[r][7], h1.w, pHi);
                        float f = facc[r][t];
                        f = __fmaf_rn((float)pLo, __fmul_rn(sLo[r], a0), f);
                        f = __fmaf_rn((float)pHi, __fmul_rn(sHi[r], a1), f);
                        f = __fmaf_rn(mnLo[r], u0, f);
                        f = __fmaf_rn(mnHi[r], u1, f);
                        facc[r][t] = f;
                    }
                }
            }
        }
        rowsEpilogue(facc, n0, N, m0, mcnt, lane, (const float*)0, bias, dst, accum);
    }
}

// gemv_q5k_g32_rows: gemv_q5k_g32 over M activation rows (rowsQ5K's lane map: 8 lanes a block, 4 blocks in flight; lane l on
// pair j = (l & 7) >> 1, half h = l & 1).
__global__ void gemv_q5k_g32_rows(const unsigned int* __restrict__ W, const int* __restrict__ A, const float* __restrict__ aS,
                                  const float* __restrict__ bias, int N, int nSB, int M, float* __restrict__ dst, int accum) {
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RN;
    if (n0 >= N) return;
    int sub = lane >> 3, j = (lane & 7) >> 1, h = lane & 1;
    int Kw = nSB * 64, nG = nSB * 8;
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = min(MT, M - m0);
        float facc[RN][MT];
        #pragma unroll
        for (int r = 0; r < RN; r++)
            #pragma unroll
            for (int t = 0; t < MT; t++) facc[r][t] = 0.f;
        for (int b0 = 0; b0 < nSB; b0 += 4) {
            int b = b0 + sub;
            if (b >= nSB) continue;
            int lo[RN][4], hi[RN][4];
            float sLo[RN], sHi[RN], mn[RN];
            #pragma unroll
            for (int r = 0; r < RN; r++) {
                int n = min(n0 + r, N - 1);
                const unsigned int* blk = W + ((long)n * nSB + b) * 44;
                uint4 hdr = *reinterpret_cast<const uint4*>(blk);
                uint4 qh = *reinterpret_cast<const uint4*>(blk + 4 + (h << 2));
                uint4 q = *reinterpret_cast<const uint4*>(blk + 12 + (j << 3) + (h << 2));
                float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xffffu)));
                float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
                int scLo, mLo, scHi, mHi;
                q4kScaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
                q4kScaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
                int sl = j << 1, shf = sl + 1;
                lo[r][0] = q5kLo(q.x, qh.x, sl); lo[r][1] = q5kLo(q.y, qh.y, sl);
                lo[r][2] = q5kLo(q.z, qh.z, sl); lo[r][3] = q5kLo(q.w, qh.w, sl);
                hi[r][0] = q5kHi(q.x, qh.x, shf); hi[r][1] = q5kHi(q.y, qh.y, shf);
                hi[r][2] = q5kHi(q.z, qh.z, shf); hi[r][3] = q5kHi(q.w, qh.w, shf);
                sLo[r] = __fmul_rn(d, (float)scLo);
                sHi[r] = __fmul_rn(d, (float)scHi);
                mn[r] = -__fmul_rn(dmin, (float)(h ? mHi : mLo));
            }
            int g = (b << 3) + (j << 1);
            #pragma unroll
            for (int t = 0; t < MT; t++) {
                if (t < mcnt) {
                    int m = m0 + t;
                    const int* am = A + (long)m * Kw;
                    int4 alo = *reinterpret_cast<const int4*>(am + (b << 6) + (j << 4) + (h << 2));
                    int4 ahi = *reinterpret_cast<const int4*>(am + (b << 6) + (j << 4) + 8 + (h << 2));
                    const float* as = aS + (long)m * 2 * nG;
                    float aLo = as[g], aHi = as[g + 1], sMin = as[nG + g + h];
                    #pragma unroll
                    for (int r = 0; r < RN; r++) {
                        int pLo = 0, pHi = 0;
                        pLo = __dp4a(lo[r][0], alo.x, pLo);
                        pLo = __dp4a(lo[r][1], alo.y, pLo);
                        pLo = __dp4a(lo[r][2], alo.z, pLo);
                        pLo = __dp4a(lo[r][3], alo.w, pLo);
                        pHi = __dp4a(hi[r][0], ahi.x, pHi);
                        pHi = __dp4a(hi[r][1], ahi.y, pHi);
                        pHi = __dp4a(hi[r][2], ahi.z, pHi);
                        pHi = __dp4a(hi[r][3], ahi.w, pHi);
                        float f = facc[r][t];
                        f = __fmaf_rn((float)pLo, __fmul_rn(sLo[r], aLo), f);
                        f = __fmaf_rn((float)pHi, __fmul_rn(sHi[r], aHi), f);
                        f = __fmaf_rn(mn[r], sMin, f);
                        facc[r][t] = f;
                    }
                }
            }
        }
        rowsEpilogue(facc, n0, N, m0, mcnt, lane, (const float*)0, bias, dst, accum);
    }
}

// gemv_q6k_g32_rows: gemv_q6k_g32 over M activation rows (rowsQ6K's lane map and q6kPlanes' layout: lane l on block
// b0 + (l >> 3), half hh = (l >> 2) & 1, elements 8·(l & 3)..+7 of each of the half's four 32-element groups).
__global__ void gemv_q6k_g32_rows(const unsigned int* __restrict__ Wu, const int* __restrict__ A, const float* __restrict__ aS,
                                  const float* __restrict__ bias, int N, int nSB, int M, float* __restrict__ dst, int accum) {
    const unsigned char* W = reinterpret_cast<const unsigned char*>(Wu);
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RN;
    if (n0 >= N) return;
    long T = (long)N * nSB;
    const unsigned char* ql = W;
    const unsigned char* qhp = W + 128 * T;
    const unsigned char* scp = W + 192 * T;
    const unsigned short* dp = reinterpret_cast<const unsigned short*>(W + 208 * T);
    int sub = lane >> 3, k = lane & 7, hh = k >> 2, q = k & 3, l0 = q << 3, is = q >> 1;
    int Kw = nSB * 64, nG = nSB * 8;
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = min(MT, M - m0);
        float facc[RN][MT];
        #pragma unroll
        for (int r = 0; r < RN; r++)
            #pragma unroll
            for (int t = 0; t < MT; t++) facc[r][t] = 0.f;
        for (int b0 = 0; b0 < nSB; b0 += 4) {
            int b = b0 + sub;
            if (b >= nSB) continue;
            int v[RN][8], c[RN][4];
            float dd[RN];
            #pragma unroll
            for (int r = 0; r < RN; r++) {
                int n = min(n0 + r, N - 1);
                long nb = (long)n * nSB + b;
                const unsigned char* qlb = ql + nb * 128 + (hh << 6) + l0;
                const unsigned char* qhb = qhp + nb * 64 + (hh << 5) + l0;
                const unsigned char* scb = scp + nb * 16 + (hh << 3);
                uint2 qa = *reinterpret_cast<const uint2*>(qlb);
                uint2 qb = *reinterpret_cast<const uint2*>(qlb + 32);
                uint2 qhv = *reinterpret_cast<const uint2*>(qhb);
                uint2 scw = *reinterpret_cast<const uint2*>(scb);
                dd[r] = __half2float(__ushort_as_half(dp[nb]));
                c[r][0] = (int)(signed char)((scw.x >> (is << 3)) & 0xffu);
                c[r][1] = (int)(signed char)((scw.x >> ((is + 2) << 3)) & 0xffu);
                c[r][2] = (int)(signed char)((scw.y >> (is << 3)) & 0xffu);
                c[r][3] = (int)(signed char)((scw.y >> ((is + 2) << 3)) & 0xffu);
                v[r][0] = (int)__vsub4((qa.x & 0x0F0F0F0Fu) | ((qhv.x & 0x03030303u) << 4), 0x20202020u);
                v[r][1] = (int)__vsub4((qa.y & 0x0F0F0F0Fu) | ((qhv.y & 0x03030303u) << 4), 0x20202020u);
                v[r][2] = (int)__vsub4((qb.x & 0x0F0F0F0Fu) | (((qhv.x >> 2) & 0x03030303u) << 4), 0x20202020u);
                v[r][3] = (int)__vsub4((qb.y & 0x0F0F0F0Fu) | (((qhv.y >> 2) & 0x03030303u) << 4), 0x20202020u);
                v[r][4] = (int)__vsub4(((qa.x >> 4) & 0x0F0F0F0Fu) | (((qhv.x >> 4) & 0x03030303u) << 4), 0x20202020u);
                v[r][5] = (int)__vsub4(((qa.y >> 4) & 0x0F0F0F0Fu) | (((qhv.y >> 4) & 0x03030303u) << 4), 0x20202020u);
                v[r][6] = (int)__vsub4(((qb.x >> 4) & 0x0F0F0F0Fu) | (((qhv.x >> 6) & 0x03030303u) << 4), 0x20202020u);
                v[r][7] = (int)__vsub4(((qb.y >> 4) & 0x0F0F0F0Fu) | (((qhv.y >> 6) & 0x03030303u) << 4), 0x20202020u);
            }
            int g = (b << 3) + (hh << 2);
            #pragma unroll
            for (int t = 0; t < MT; t++) {
                if (t < mcnt) {
                    int m = m0 + t;
                    const int* ab = A + (long)m * Kw + (b << 6) + (hh << 5) + (q << 1);
                    int2 a0 = *reinterpret_cast<const int2*>(ab);
                    int2 a1 = *reinterpret_cast<const int2*>(ab + 8);
                    int2 a2 = *reinterpret_cast<const int2*>(ab + 16);
                    int2 a3 = *reinterpret_cast<const int2*>(ab + 24);
                    const float* as = aS + (long)m * 2 * nG;
                    float s0 = as[g], s1 = as[g + 1], s2 = as[g + 2], s3 = as[g + 3];
                    #pragma unroll
                    for (int r = 0; r < RN; r++) {
                        int p0 = __dp4a(v[r][1], a0.y, __dp4a(v[r][0], a0.x, 0));
                        int p1 = __dp4a(v[r][3], a1.y, __dp4a(v[r][2], a1.x, 0));
                        int p2 = __dp4a(v[r][5], a2.y, __dp4a(v[r][4], a2.x, 0));
                        int p3 = __dp4a(v[r][7], a3.y, __dp4a(v[r][6], a3.x, 0));
                        float f = facc[r][t];
                        f = __fmaf_rn((float)(c[r][0] * p0), __fmul_rn(dd[r], s0), f);
                        f = __fmaf_rn((float)(c[r][1] * p1), __fmul_rn(dd[r], s1), f);
                        f = __fmaf_rn((float)(c[r][2] * p2), __fmul_rn(dd[r], s2), f);
                        f = __fmaf_rn((float)(c[r][3] * p3), __fmul_rn(dd[r], s3), f);
                        facc[r][t] = f;
                    }
                }
            }
        }
        rowsEpilogue(facc, n0, N, m0, mcnt, lane, (const float*)0, bias, dst, accum);
    }
}

// gemv_w8a8_g32_rows: gemv_w8a8_g32 over M activation rows (rowW8g32: lane l on words l, l + 32, ...; the per-row weight scale in
// the epilogue).
__global__ void gemv_w8a8_g32_rows(const int* __restrict__ W, const int* __restrict__ A, const float* __restrict__ wScale,
                                   const float* __restrict__ aS, const float* __restrict__ bias,
                                   int N, int Kdiv4, int M, float* __restrict__ dst, int accum) {
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RN;
    if (n0 >= N) return;
    int sStride = 2 * (Kdiv4 >> 3);
    for (int m0 = 0; m0 < M; m0 += MT) {
        int mcnt = min(MT, M - m0);
        float facc[RN][MT];
        #pragma unroll
        for (int r = 0; r < RN; r++)
            #pragma unroll
            for (int t = 0; t < MT; t++) facc[r][t] = 0.f;
        for (int k = lane; k < Kdiv4; k += 32) {
            int w[RN];
            #pragma unroll
            for (int r = 0; r < RN; r++) w[r] = W[(long)min(n0 + r, N - 1) * Kdiv4 + k];
            #pragma unroll
            for (int t = 0; t < MT; t++) {
                if (t < mcnt) {
                    int m = m0 + t;
                    int a = A[(long)m * Kdiv4 + k];
                    float s = aS[(long)m * sStride + (k >> 3)];
                    #pragma unroll
                    for (int r = 0; r < RN; r++) facc[r][t] = __fmaf_rn((float)__dp4a(w[r], a, 0), s, facc[r][t]);
                }
            }
        }
        rowsEpilogue(facc, n0, N, m0, mcnt, lane, wScale, bias, dst, accum);
    }
}

} // extern "C"
