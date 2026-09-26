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
// aScale[g], 8 packed ints into q[8g .. 8g+8). Called by all threads of the block after vals is
// complete and visible.
__device__ __forceinline__ void quantG32(const float* __restrict__ vals, int nG,
                                         int* __restrict__ q, float* __restrict__ aScale) {
    for (int g = threadIdx.x; g < nG; g += blockDim.x) {
        const float* v = vals + 32 * g;
        float ma = 0.f;
        for (int i = 0; i < 32; i++) ma = fmaxf(ma, fabsf(v[i]));
        float sc = ma / 127.f;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        aScale[g] = sc;
        for (int j = 0; j < 8; j++) {
            int packed = 0;
            for (int b = 0; b < 4; b++) {
                int x = __float2int_rn(v[4 * j + b] * inv);
                x = max(-127, min(127, x));
                packed |= (x & 0xff) << (8 * b);
            }
            q[8 * g + j] = packed;
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
    float rnorm = rsqrtf(red[0] / H + eps); __syncthreads();
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
            a = x / (1.f + __expf(-x));
        } else {
            a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x))));
        }
        dscratch[k] = a * u[uOff + k];
    }
    __syncthreads();
    quantG32(dscratch, I / 32, q, scale);
}

// gemv_w4a8_g32: gemv_w4a8_fwd with per-32 activation scales aS[Kgroups]. Each word's dp4a partial
// is scaled by weightScale(group) * aS[group] (group = word>>2) instead of the weight scale alone,
// and the epilogue drops the per-vector activation multiply: dst[n] = sum + bias. Same launch
// geometry as gemv_w4a8_fwd (8 rows per 256-thread block).
__global__ void gemv_w4a8_g32(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    const float* __restrict__ aS, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
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
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) {
        float val = __fadd_rn(facc, (bias ? bias[n] : 0.f));
        dst[n] = accum ? dst[n] + val : val;
    }
}

// gemv_w8a8_g32: gemv_w8a8_fwd with per-32 activation scales aS[Kdiv4/8]. The per-row kernel
// accumulates the whole row in one int32; here each 4-element word's dp4a partial is scaled by its
// group's activation scale (group = word>>3) and accumulated in f32, and the per-row weight scale
// is applied in the epilogue: dst[n] = sum * wScale[n] + bias.
__global__ void gemv_w8a8_g32(
    const int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ wScale,
    const float* __restrict__ aS, const float* __restrict__ bias,
    int N, int Kdiv4, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int* wr = W + (long)n * Kdiv4;
    float facc = 0.f;
    for (int k = lane; k < Kdiv4; k += 32) facc = __fmaf_rn((float)__dp4a(wr[k], a[k], 0), aS[k >> 3], facc);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffff, facc, o);
    if (lane == 0) {
        float val = __fmaf_rn(facc, wScale[n], (bias ? bias[n] : 0.f));
        dst[n] = accum ? dst[n] + val : val;
    }
}

} // extern "C"
