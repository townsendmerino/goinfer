// actgroup.cu — per-32 ACTIVATION quantization for the resident decode path
// (docs/tasks/task-actquant-pergroup-2026-09.md, queue-engineering.md H2).
//
// WHY ITS OWN FILE. The per-row kernels these mirror live in glue.cu (rmsnorm_quant, quant_vec,
// glu_quant) and aikit's gemv_quant.cu (gemv_w4a8_fwd, gemv_w8a8_fwd); glue.ptx is an audited
// artifact. Same precedent as decode_splitkv.cu and gptoss_act.cu: a new module, nothing audited
// modified, and the per-row path byte-identical because it never loads this module's kernels.
//
// THE CHANGE. The per-row kernels scale a whole activation vector by one max/127, and a model with
// massive activation outliers (Phi-3-mini: max/rms ~80-90 at down_proj inputs) loses nearly the
// whole vector to rounding. Here every group of 32 consecutive elements gets its own max/127 scale,
// written to aScale[g]. Each quantizer is its per-row sibling up to the epilogue: same normalize /
// activation passes, same __float2int_rn, same [-127, 127] clamp, same zero-scale convention
// (sc > 0 ? 1/sc : 0), same 4-int8-per-int packing. The GEMVs are their per-row siblings with the
// activation scale moved inside the per-word multiply: word w (8 int4 or 4 int8 elements) belongs to
// group w>>2 (int4) or w>>3 (int8). Every multiply-accumulate is an explicit __fmaf_rn / __fmul_rn
// (docs/kernel-bit-identity.md): no compiler contraction discretion.
//
// PRECONDITION: the vector length is a multiple of 32 (the resident path declines otherwise).

#include <cuda_fp16.h>

#define ACT_GELU_TANH 0
#define ACT_SILU      1

// quantG32 quantizes vals[0 .. 32*nG) group by group: one thread per group of 32, max/127 scale into
// aScale[g], 8 packed ints into q[8g .. 8g+8). It also writes aScale[nG + g] = aScale[g] · Σ codes,
// the group's dequantized activation sum, which gemv_q4k_g32's per-sub-block minimum multiplies
// (docs/tasks/task-int4-weight-quality-2026-09.md Phase 1b). So aScale holds 2·nG floats (the
// resident runner's actScaleLen), and every other kernel reads only the first nG. Called by all
// threads of the block after vals is complete and visible.
__device__ __forceinline__ void quantG32(const float* __restrict__ vals, int nG,
                                         int* __restrict__ q, float* __restrict__ aScale) {
    for (int g = threadIdx.x; g < nG; g += blockDim.x) {
        const float* v = vals + 32 * g;
        float ma = 0.f;
        for (int i = 0; i < 32; i++) ma = fmaxf(ma, fabsf(v[i]));
        float sc = ma / 127.f;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        aScale[g] = sc;
        int sum = 0;
        for (int j = 0; j < 8; j++) {
            int packed = 0;
            for (int b = 0; b < 4; b++) {
                int x = __float2int_rn(v[4 * j + b] * inv);
                x = max(-127, min(127, x));
                sum += x;
                packed |= (x & 0xff) << (8 * b);
            }
            q[8 * g + j] = packed;
        }
        aScale[nG + g] = __fmul_rn(sc, (float)sum);
    }
}

// q4kByte is byte k (0..11) of a Q4_K block's 12 scale bytes, held as three words. Selects and
// shifts only: an indexed byte array here compiled to a per-block local-memory spill (a 48-byte
// __local_depot) that made gemv_q4k_g32 slower than the int8 GEMV it replaces.
__device__ __forceinline__ unsigned int q4kByte(int k, unsigned int s0, unsigned int s1, unsigned int s2) {
    unsigned int w = k < 4 ? s0 : (k < 8 ? s1 : s2);
    return (w >> ((k & 3) << 3)) & 0xffu;
}

// q4kScaleMin is ggml's get_scale_min_k4 for sub-block j (0..7) of one Q4_K block.
__device__ __forceinline__ void q4kScaleMin(int j, unsigned int s0, unsigned int s1, unsigned int s2,
                                            int* sc, int* m) {
    if (j < 4) {
        *sc = (int)(q4kByte(j, s0, s1, s2) & 63u);
        *m = (int)(q4kByte(j + 4, s0, s1, s2) & 63u);
    } else {
        unsigned int hi = q4kByte(j + 4, s0, s1, s2);
        *sc = (int)((hi & 0x0Fu) | ((q4kByte(j - 4, s0, s1, s2) >> 6) << 4));
        *m = (int)((hi >> 4) | ((q4kByte(j, s0, s1, s2) >> 6) << 4));
    }
}

// q5kLo / q5kHi are four Q5_K values of one code word: the low or high nibbles plus their fifth bits, bit `shift` of each qh byte.
__device__ __forceinline__ int q5kLo(unsigned int code, unsigned int qhw, int shift) {
    return (int)((code & 0x0F0F0F0Fu) | (((qhw >> shift) & 0x01010101u) << 4));
}
__device__ __forceinline__ int q5kHi(unsigned int code, unsigned int qhw, int shift) {
    return (int)(((code >> 4) & 0x0F0F0F0Fu) | (((qhw >> shift) & 0x01010101u) << 4));
}

// rowsQ5K: R rows n0.. of GGUF Q5_K weights (the file's 176-byte blocks, 44 words: d|dmin, three words of scales and minimums,
// qh[32], qs[128]) × per-32 int8 activations, one lane partial a row into facc[r]. asum[g] = aS[g]·Σaq (quantG32).
//
// LANE MAPPING: 8 lanes a block, 4 blocks in flight. Lane l on block b0 + (l >> 3), pair j = (l & 7) >> 1 (sub-blocks 2j, 2j+1)
// and half h = l & 1: 16 code bytes (elements 64j+16h.. in the low nibbles, 64j+32+16h.. in the high), the matching 16 qh bytes
// (bits 2j and 2j+1), one 16-byte activation load per sub-block. Each group's minimum is added once, by the even lane for the
// low sub-block and the odd lane for the high one. Per block and lane:
//   facc += aS_lo·d·sc_lo · idot_lo + aS_hi·d·sc_hi · idot_hi − dmin·m_h · asum_h
// Every load is 16-byte aligned. The activation loads are shared by the R rows; each row's arithmetic is the same at every R.
template <int R>
__device__ __forceinline__ void rowsQ5K(const unsigned int* __restrict__ W, int n0, int N, const int* __restrict__ a,
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
            float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xffffu)));
            float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
            int scLo, mLo, scHi, mHi;
            q4kScaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
            q4kScaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
            int sl = j << 1, sh = sl + 1;
            int pLo = 0, pHi = 0;
            pLo = __dp4a(q5kLo(q.x, qh.x, sl), lo.x, pLo);
            pLo = __dp4a(q5kLo(q.y, qh.y, sl), lo.y, pLo);
            pLo = __dp4a(q5kLo(q.z, qh.z, sl), lo.z, pLo);
            pLo = __dp4a(q5kLo(q.w, qh.w, sl), lo.w, pLo);
            pHi = __dp4a(q5kHi(q.x, qh.x, sh), hi.x, pHi);
            pHi = __dp4a(q5kHi(q.y, qh.y, sh), hi.y, pHi);
            pHi = __dp4a(q5kHi(q.z, qh.z, sh), hi.z, pHi);
            pHi = __dp4a(q5kHi(q.w, qh.w, sh), hi.w, pHi);
            int m = h ? mHi : mLo;
            float f = facc[r];
            f = __fmaf_rn((float)pLo, __fmul_rn(__fmul_rn(d, (float)scLo), aLo), f);
            f = __fmaf_rn((float)pHi, __fmul_rn(__fmul_rn(d, (float)scHi), aHi), f);
            f = __fmaf_rn(-__fmul_rn(dmin, (float)m), sMin, f);
            facc[r] = f;
        }
    }
}

// rowsQ6K: R rows n0.. of GGUF Q6_K weights in q6kPlanes' layout (cuda/pack_stream.go) × per-32 int8 activations, one lane
// partial a row into facc[r]. The weight's T = N·nSB blocks sit in four planes of one buffer: ql at 0, qh at 128T, the int8
// scales at 192T, d (f16) at 208T, block g = n·nSB + b at g·128, g·64, g·16 and g·2 within each; every plane is 16-byte aligned.
//
// LANE MAPPING: lane l on block b0 + (l >> 3), half hh = (l >> 2) & 1, elements 8·(l & 3)…+7 of each of the half's four 32-element
// groups (one uint2 of ql low, one of ql high, one of qh, one of scales). A code is (4 ql bits | 2 qh bits << 4) − 32; the group's
// one sub-scale (per 16 elements) multiplies the integer dot, then d and the group's activation scale:
//   facc += (sc·idot) · (d·aS_g)
// The activation loads are shared by the R rows; each row's arithmetic is the same at every R.
template <int R>
__device__ __forceinline__ void rowsQ6K(const unsigned char* __restrict__ W, int n0, int N, int nSB, const int* __restrict__ a,
                                        const float* __restrict__ aS, int lane, float* facc) {
    long T = (long)N * nSB;
    const unsigned char* ql = W;
    const unsigned char* qhp = W + 128 * T;
    const unsigned char* scp = W + 192 * T;
    const unsigned short* dp = reinterpret_cast<const unsigned short*>(W + 208 * T);
    int sub = lane >> 3, k = lane & 7, hh = k >> 2, q = k & 3, l0 = q << 3, is = q >> 1;
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
            float d = __half2float(__ushort_as_half(dp[nb]));
            // The group sub-scales sc[8hh + is + 2·grp], grp 0..3: bytes is and is + 2 of scw.x, then of scw.y.
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

extern "C" {

// rmsnorm_quant_g32: rmsnorm_quant's passes 1-2 unchanged (sum of squares ladder, normed values),
// then per-32 quantization. aScale holds H/32 floats. Shared: H + blockDim floats.
__global__ void rmsnorm_quant_g32(const float* __restrict__ x, const float* __restrict__ w,
                                  int H, float eps, int addOne, int* __restrict__ aq, float* __restrict__ aScale) {
    extern __shared__ float sh[];
    float* normed = sh;
    float* red = sh + H;
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss = __fmaf_rn(x[k], x[k], ss);
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = 1.f / sqrtf(red[0] / H + eps); __syncthreads();
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; normed[k] = x[k] * g * rnorm; }
    __syncthreads();
    quantG32(normed, H / 32, aq, aScale);
}

// quant_vec_g32: quant_vec with per-32 scales (scale holds N/32 floats). No shared memory.
__global__ void quant_vec_g32(const float* __restrict__ x, int N, int* __restrict__ q, float* __restrict__ scale) {
    quantG32(x, N / 32, q, scale);
}

// glu_quant_g32: glu_quant's activation pass unchanged (dscratch[k] = act(g)*u), then per-32
// quantization of dscratch (scale holds I/32 floats).
__global__ void glu_quant_g32(const float* __restrict__ g, const float* __restrict__ u,
                              int gOff, int uOff, int I, int act,
                              int* __restrict__ q, float* __restrict__ scale, float* __restrict__ dscratch) {
    int t = threadIdx.x, nt = blockDim.x;
    for (int k = t; k < I; k += nt) {
        float x = g[gOff + k], a;
        if (act == ACT_SILU) {
            a = x / (1.f + expf(-x));
        } else {
            a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x))));
        }
        dscratch[k] = a * u[uOff + k];
    }
    __syncthreads();
    quantG32(dscratch, I / 32, q, scale);
}

// ---- Per-32 matvec row bodies. Each returns ONE lane's partial sum for output row `n` (the caller
// does the warp reduction and the epilogue). The standalone matvecs below and the fused per-32
// kernels (fused_rms_qkv_g32 / fused_rms_gu_g32) call these same functions, so a fused projection
// is bit-identical to the unfused chain by construction, not by a copied expression.

// rowW4g32: gemv_w4a8_fwd's row with per-32 activation scales aS[Kgroups]. Each word's dp4a partial
// is scaled by weightScale(group) * aS[group] (group = word>>2) instead of the weight scale alone.
__device__ __forceinline__ float rowW4g32(const unsigned int* __restrict__ wr, const int* __restrict__ a,
                                          const __half* __restrict__ sr, const float* __restrict__ aS,
                                          int Kwords, int lane) {
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi0 = base + lane, wi1 = base + 32 + lane;
        unsigned int w0 = wr[wi0], w1 = wr[wi1];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi0 + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi1 + 1], p1);
        facc = __fmaf_rn((float)p0, __fmul_rn(__half2float(sr[wi0 >> 2]), aS[wi0 >> 2]), facc);
        facc = __fmaf_rn((float)p1, __fmul_rn(__half2float(sr[wi1 >> 2]), aS[wi1 >> 2]), facc);
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
            facc = __fmaf_rn((float)p, __fmul_rn(__half2float(sr[wi >> 2]), aS[wi >> 2]), facc);
        }
    }
    return facc;
}

// rowW8g32: gemv_w8a8_fwd's row with per-32 activation scales aS[Kdiv4/8]. The per-row kernel
// accumulates the whole row in one int32; here each 4-element word's dp4a partial is scaled by its
// group's activation scale (group = word>>3) and accumulated in f32. The per-row weight scale is
// the epilogue's (finishRow).
__device__ __forceinline__ float rowW8g32(const int* __restrict__ wr, const int* __restrict__ a,
                                          const float* __restrict__ aS, int Kdiv4, int lane) {
    float facc = 0.f;
    for (int k = lane; k < Kdiv4; k += 32) facc = __fmaf_rn((float)__dp4a(wr[k], a[k], 0), aS[k >> 3], facc);
    return facc;
}

// rowQ4K: one row of GGUF Q4_K weights (the raw super-blocks, 36 words per 256 weights: d|dmin,
// three words of packed 6-bit scales and minimums, 32 words of codes) × per-32 int8 activations.
// asum[g] = aS[g]·Σaq over group g (quantG32's second half).
//
// LANE MAPPING: lane l owns sub-block pair j = l & 3 of block b0 + (l >> 2), so a warp covers 8
// blocks per pass and a lane 64 weights: one 16-byte header load, one scale/min unpack, two 16-byte
// code loads, four 16-byte activation loads, 16 dp4a. (A lane per code word paid the header decode
// per 8 weights and was compute-bound; 8 lanes per row, lever 1, was 7% slower on the 7B and did not
// help the 1.5B — docs/tasks/task-int4-weight-quality-2026-09.md.) Code word t of pair j holds
// elements 64j+4t..+3 (low nibbles, sub-block 2j) and 64j+32+4t..+3 (high, 2j+1); their activation
// words are 64b+16j+t and +8. A block is 144 = 9·16 bytes and its codes start at byte 16, so every
// vector load is 16-byte aligned. Per block and group:
//   facc += aS_g·d·sc_g · idot_g − dmin·m_g · asum_g
__device__ __forceinline__ float rowQ4K(const unsigned int* __restrict__ wr, const int* __restrict__ a,
                                        const float* __restrict__ aS, const float* __restrict__ asum,
                                        int nSB, int lane) {
    int j = lane & 3, sub = lane >> 2;
    float facc = 0.f;
    for (int b0 = 0; b0 < nSB; b0 += 8) {
        int b = b0 + sub;
        if (b < nSB) {
            const unsigned int* blk = wr + b * 36;
            uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xffffu)));
            float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
            int scLo, mLo, scHi, mHi;
            q4kScaleMin(j << 1, hdr.y, hdr.z, hdr.w, &scLo, &mLo);
            q4kScaleMin((j << 1) | 1, hdr.y, hdr.z, hdr.w, &scHi, &mHi);
            const uint4* qv = reinterpret_cast<const uint4*>(blk + 4 + (j << 3));
            uint4 q0 = qv[0], q1 = qv[1];
            const int4* av = reinterpret_cast<const int4*>(a + (b << 6) + (j << 4));
            int4 l0 = av[0], l1 = av[1], h0 = av[2], h1 = av[3];
            int pLo = 0, pHi = 0;
            pLo = __dp4a((int)(q0.x & 0x0F0F0F0Fu), l0.x, pLo);
            pLo = __dp4a((int)(q0.y & 0x0F0F0F0Fu), l0.y, pLo);
            pLo = __dp4a((int)(q0.z & 0x0F0F0F0Fu), l0.z, pLo);
            pLo = __dp4a((int)(q0.w & 0x0F0F0F0Fu), l0.w, pLo);
            pLo = __dp4a((int)(q1.x & 0x0F0F0F0Fu), l1.x, pLo);
            pLo = __dp4a((int)(q1.y & 0x0F0F0F0Fu), l1.y, pLo);
            pLo = __dp4a((int)(q1.z & 0x0F0F0F0Fu), l1.z, pLo);
            pLo = __dp4a((int)(q1.w & 0x0F0F0F0Fu), l1.w, pLo);
            pHi = __dp4a((int)((q0.x >> 4) & 0x0F0F0F0Fu), h0.x, pHi);
            pHi = __dp4a((int)((q0.y >> 4) & 0x0F0F0F0Fu), h0.y, pHi);
            pHi = __dp4a((int)((q0.z >> 4) & 0x0F0F0F0Fu), h0.z, pHi);
            pHi = __dp4a((int)((q0.w >> 4) & 0x0F0F0F0Fu), h0.w, pHi);
            pHi = __dp4a((int)((q1.x >> 4) & 0x0F0F0F0Fu), h1.x, pHi);
            pHi = __dp4a((int)((q1.y >> 4) & 0x0F0F0F0Fu), h1.y, pHi);
            pHi = __dp4a((int)((q1.z >> 4) & 0x0F0F0F0Fu), h1.z, pHi);
            pHi = __dp4a((int)((q1.w >> 4) & 0x0F0F0F0Fu), h1.w, pHi);
            int g = (b << 3) + (j << 1);
            facc = __fmaf_rn((float)pLo, __fmul_rn(__fmul_rn(d, (float)scLo), aS[g]), facc);
            facc = __fmaf_rn((float)pHi, __fmul_rn(__fmul_rn(d, (float)scHi), aS[g + 1]), facc);
            facc = __fmaf_rn(-__fmul_rn(dmin, (float)mLo), asum[g], facc);
            facc = __fmaf_rn(-__fmul_rn(dmin, (float)mHi), asum[g + 1], facc);
        }
    }
    return facc;
}

// warpSum: the per-32 matvecs' lane reduction (shfl_down 16..1, the original kernels' order).
__device__ __forceinline__ float warpSum(float f) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) f += __shfl_down_sync(0xffffffffu, f, off);
    return f;
}

// Weight kinds a fused per-32 projection can carry (cuda/resident.go's wkCode).
#define WK_INT8 0
#define WK_INT4 1
#define WK_Q4K  2
#define WK_Q5K  3
#define WK_Q6K  4

// projRow: lane partial for row n of an N-row, K-wide projection of the given kind. sc is the int8 row scales
// (WK_INT8), the int4 f16 group scales (WK_INT4) or unused (the block kinds). N matters only to WK_Q6K, whose
// planes are laid out over the whole weight. Warp-uniform kind.
__device__ __forceinline__ float projRow(int kind, const unsigned int* __restrict__ W, const void* __restrict__ sc,
                                         int n, int N, int K, const int* __restrict__ a, const float* __restrict__ aS,
                                         const float* __restrict__ asum, int lane) {
    if (kind == WK_Q4K) return rowQ4K(W + (long)n * (K >> 8) * 36, a, aS, asum, K >> 8, lane);
    if (kind == WK_Q5K) {
        float f[1] = {0.f};
        rowsQ5K<1>(W, n, N, a, aS, asum, K >> 8, lane, f);
        return f[0];
    }
    if (kind == WK_Q6K) {
        float f[1] = {0.f};
        rowsQ6K<1>(reinterpret_cast<const unsigned char*>(W), n, N, K >> 8, a, aS, lane, f);
        return f[0];
    }
    if (kind == WK_INT4) return rowW4g32(W + (long)n * (K >> 3), a, (const __half*)sc + (long)n * (K >> 5), aS, K >> 3, lane);
    return rowW8g32((const int*)W + (long)n * (K >> 2), a, aS, K >> 2, lane);
}

// finishRow: the matching standalone kernel's epilogue (accum = 0).
__device__ __forceinline__ float finishRow(int kind, float sum, const void* __restrict__ sc,
                                           const float* __restrict__ bias, int n) {
    if (kind == WK_INT8) return __fmaf_rn(sum, ((const float*)sc)[n], (bias ? bias[n] : 0.f));
    return __fadd_rn(sum, (bias ? bias[n] : 0.f));
}

// gemv_w4a8_g32: gemv_w4a8_fwd with per-32 activation scales aS[Kgroups]; the epilogue drops the
// per-vector activation multiply: dst[n] = sum + bias. Same launch geometry as gemv_w4a8_fwd (8 rows
// per 256-thread block). Every multiply-accumulate is an explicit __fmaf_rn / __fmul_rn.
__global__ void gemv_w4a8_g32(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    const float* __restrict__ aS, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    float facc = warpSum(rowW4g32(W + (long)n * Kwords, a, gs + (long)n * Kgroups, aS, Kwords, lane));
    if (lane == 0) {
        float val = __fadd_rn(facc, (bias ? bias[n] : 0.f));
        dst[n] = accum ? dst[n] + val : val;
    }
}

// gemv_w8a8_g32: gemv_w8a8_fwd with per-32 activation scales aS[Kdiv4/8]; the per-row weight scale
// is applied in the epilogue: dst[n] = sum * wScale[n] + bias.
__global__ void gemv_w8a8_g32(
    const int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ wScale,
    const float* __restrict__ aS, const float* __restrict__ bias,
    int N, int Kdiv4, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    float facc = warpSum(rowW8g32(W + (long)n * Kdiv4, a, aS, Kdiv4, lane));
    if (lane == 0) {
        float val = __fmaf_rn(facc, wScale[n], (bias ? bias[n] : 0.f));
        dst[n] = accum ? dst[n] + val : val;
    }
}

// gemv_q4k_g32: GGUF Q4_K weights × per-32 int8 activations (rowQ4K). aS holds 2·Kgroups floats: the
// per-group activation scales, then aS·Σaq per group (quantG32). One warp per output row, 8 rows per
// 256-thread block (the per-row GEMVs' geometry).
__global__ void gemv_q4k_g32(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ aS,
    const float* __restrict__ bias, int N, int nSB, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    float facc = warpSum(rowQ4K(W + (long)n * nSB * 36, a, aS, aS + nSB * 8, nSB, lane));
    if (lane == 0) {
        float val = __fadd_rn(facc, (bias ? bias[n] : 0.f));
        dst[n] = accum ? dst[n] + val : val;
    }
}

// KQ_ROWS is the rows a warp of gemv_q5k_g32 / gemv_q6k_g32 (cuda/resident.go's kquantGemvRows, which must match).
#define KQ_ROWS 1

// gemv_q5k_g32 / gemv_q6k_g32: the file's own Q5_K blocks (rowsQ5K) and Q6_K planes (rowsQ6K) × per-32 int8 activations, for
// the int8 copies of a q4k load. aS holds scales then group sums (quantG32). KQ_ROWS rows a warp, 8 warps a 256-thread block;
// each row's arithmetic is projRow's, so a fused projection of these kinds equals the standalone one.
__global__ void gemv_q5k_g32(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ aS,
    const float* __restrict__ bias, int N, int nSB, float* __restrict__ dst, int accum)
{
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * KQ_ROWS;
    if (n0 >= N) return;
    float facc[KQ_ROWS];
    #pragma unroll
    for (int r = 0; r < KQ_ROWS; r++) facc[r] = 0.f;
    rowsQ5K<KQ_ROWS>(W, n0, N, a, aS, aS + nSB * 8, nSB, lane, facc);
    #pragma unroll
    for (int r = 0; r < KQ_ROWS; r++) {
        float s = warpSum(facc[r]);
        int n = n0 + r;
        if (lane == 0 && n < N) {
            float val = __fadd_rn(s, (bias ? bias[n] : 0.f));
            dst[n] = accum ? dst[n] + val : val;
        }
    }
}

__global__ void gemv_q6k_g32(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ aS,
    const float* __restrict__ bias, int N, int nSB, float* __restrict__ dst, int accum)
{
    int lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * KQ_ROWS;
    if (n0 >= N) return;
    float facc[KQ_ROWS];
    #pragma unroll
    for (int r = 0; r < KQ_ROWS; r++) facc[r] = 0.f;
    rowsQ6K<KQ_ROWS>(reinterpret_cast<const unsigned char*>(W), n0, N, nSB, a, aS, lane, facc);
    #pragma unroll
    for (int r = 0; r < KQ_ROWS; r++) {
        float s = warpSum(facc[r]);
        int n = n0 + r;
        if (lane == 0 && n < N) {
            float val = __fadd_rn(s, (bias ? bias[n] : 0.f));
            dst[n] = accum ? dst[n] + val : val;
        }
    }
}

// rmsQuantShared is rmsnorm_quant_g32's body into SHARED memory: sh = normed[H] | red[256] | aq[H/4] |
// aS[2·H/32]. Every block of a fused per-32 kernel runs it redundantly (x[H] is L2-resident), so the
// activation, its per-32 scales and its group sums never make a global round trip, and no single-block
// quantizer launch sits serially in the chain. Same passes, same reduction order and quantG32 itself,
// so the codes and scales equal rmsnorm_quant_g32's exactly. Block size must be 256 (the ladder).
__device__ __forceinline__ void rmsQuantShared(const float* __restrict__ x, const float* __restrict__ w,
                                               int H, float eps, int addOne, float* sh,
                                               int** aqOut, float** aSOut) {
    float* normed = sh;
    float* red = sh + H;
    int* aq = (int*)(red + 256);
    float* aS = (float*)(aq + H / 4);
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss = __fmaf_rn(x[k], x[k], ss);
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = 1.f / sqrtf(red[0] / H + eps); __syncthreads();
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; normed[k] = x[k] * g * rnorm; }
    __syncthreads();
    quantG32(normed, H / 32, aq, aS);
    __syncthreads();
    *aqOut = aq;
    *aSOut = aS;
}

// fused_rms_qkv_g32: rmsnorm + per-32 quant (rmsQuantShared) + this block's Q|K|V rows, one launch
// instead of four, for per-32 activations (fused_rms_qkv is per-vector and int4-only). Each projection
// has its own weight kind (WK_*) and scale pointer; a warp walks `rpw` consecutive rows, and a
// projection kind is uniform across a warp's row. Shared: (H + 256)·4 + H + 2·(H/32)·4 bytes.
// grid: ceil((qDim + 2·kvDim) / (8·rpw)) blocks × 256 threads.
__global__ void fused_rms_qkv_g32(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wq, const void* __restrict__ sq, const float* __restrict__ bq, int kq,
    const unsigned int* __restrict__ Wk, const void* __restrict__ sk, const float* __restrict__ bk, int kk,
    const unsigned int* __restrict__ Wv, const void* __restrict__ sv, const float* __restrict__ bv, int kv,
    int qDim, int kvDim, int rpw,
    float* __restrict__ qOut, float* __restrict__ kOut, float* __restrict__ vOut)
{
    extern __shared__ float sh[];
    int* aq;
    float* aS;
    rmsQuantShared(x, nrm, H, eps, addOne, sh, &aq, &aS);
    const float* asum = aS + H / 32;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nrows = qDim + 2 * kvDim;
    int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * rpw;
    for (int i = 0; i < rpw; i++) {
        int r = r0 + i;
        if (r >= nrows) return; // warp-uniform
        const unsigned int* W; const void* sc; const float* bias; float* dst; int row, kind;
        if (r < qDim)               { W = Wq; sc = sq; bias = bq; dst = qOut; row = r;                kind = kq; }
        else if (r < qDim + kvDim)  { W = Wk; sc = sk; bias = bk; dst = kOut; row = r - qDim;         kind = kk; }
        else                        { W = Wv; sc = sv; bias = bv; dst = vOut; row = r - qDim - kvDim; kind = kv; }
        float f = warpSum(projRow(kind, W, sc, row, r < qDim ? qDim : kvDim, H, aq, aS, asum, lane));
        if (lane == 0) dst[row] = finishRow(kind, f, sc, bias, row);
    }
}

// fused_rms_gu_g32: rmsnorm + per-32 quant (rmsQuantShared) + this block's gate|up rows (no bias).
// Same structure and shared layout as fused_rms_qkv_g32; rows 0..I-1 are gate, I..2I-1 up.
// grid: ceil(2·I / (8·rpw)) blocks × 256 threads.
__global__ void fused_rms_gu_g32(
    const float* __restrict__ x, const float* __restrict__ nrm, int H, float eps, int addOne,
    const unsigned int* __restrict__ Wg, const void* __restrict__ sg, int kg,
    const unsigned int* __restrict__ Wu, const void* __restrict__ su, int ku,
    int I, int rpw, float* __restrict__ gOut, float* __restrict__ uOut)
{
    extern __shared__ float sh[];
    int* aq;
    float* aS;
    rmsQuantShared(x, nrm, H, eps, addOne, sh, &aq, &aS);
    const float* asum = aS + H / 32;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * rpw;
    for (int i = 0; i < rpw; i++) {
        int r = r0 + i;
        if (r >= 2 * I) return; // warp-uniform
        bool gate = r < I;
        int row = gate ? r : r - I;
        const unsigned int* W = gate ? Wg : Wu;
        const void* sc = gate ? sg : su;
        int kind = gate ? kg : ku;
        float f = warpSum(projRow(kind, W, sc, row, I, H, aq, aS, asum, lane));
        if (lane == 0) (gate ? gOut : uOut)[row] = finishRow(kind, f, sc, (const float*)0, row);
    }
}

} // extern "C"
