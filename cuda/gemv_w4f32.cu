// gemv_w4f32.cu — int4-weight GEMVs over f32 activations, for a decode that must not round its activations.
//
// The W4A8 kernels (aikit's gemv_w4a8_fwd, moe.cu's gemv_w4a8_moe / _wacc) quantize every activation to int8 with one
// scale per vector. On Gemma 4 26B-A4B that rounding makes the decode chaotic at some tokens: a difference of 1e-5 in the
// input swings the late layers and can land on a garbage digit, on the CPU's W4A8 path as on this one, while the same
// int4 weights over f32 activations stay at cosine 1.00000 (docs/tasks/task-multimodal-support-2026-10.md, "S19").
// These kernels keep the int4 weights and their layout and read the activation as f32.
//
// Layout (the W4A8 kernels'): W is [rows][Kwords] u32, 8 int4 per word, nibble-permuted at pack time so that byte b's low
// nibble is input k = 8w+b and its high nibble is k = 8w+4+b; a stored nibble is the value + 8. gs is [rows][Kgroups] f16,
// one scale per 32 inputs (word w's group is w>>2). One warp per output row, the W4A8 kernels' word walk (lane-strided,
// two words in flight, a 32-stride ragged tail) and their shuffle reduction.
//
// Every float MAC is an explicit intrinsic (the FMA lint): with f32 activations the products are not exact, so the order
// shows in the result, and it is fixed here as: per word, the eight products as one FMA chain in k order, then the word's
// sum times its group scale into the lane accumulator.
//
// Regenerate with:  ./build_ptx.sh gemv_w4f32

#include <cuda_fp16.h>

// w4f32_word is one word's eight int4 values against their eight f32 inputs (a8 points at input 8w).
__device__ __forceinline__ float w4f32_word(unsigned int word, const float* __restrict__ a8) {
    float4 lo = *reinterpret_cast<const float4*>(a8);
    float4 hi = *reinterpret_cast<const float4*>(a8 + 4);
    float p = 0.f;
    p = __fmaf_rn((float)((int)(word & 0xFu) - 8), lo.x, p);
    p = __fmaf_rn((float)((int)((word >> 8) & 0xFu) - 8), lo.y, p);
    p = __fmaf_rn((float)((int)((word >> 16) & 0xFu) - 8), lo.z, p);
    p = __fmaf_rn((float)((int)((word >> 24) & 0xFu) - 8), lo.w, p);
    p = __fmaf_rn((float)((int)((word >> 4) & 0xFu) - 8), hi.x, p);
    p = __fmaf_rn((float)((int)((word >> 12) & 0xFu) - 8), hi.y, p);
    p = __fmaf_rn((float)((int)((word >> 20) & 0xFu) - 8), hi.z, p);
    p = __fmaf_rn((float)((int)((word >> 28) & 0xFu) - 8), hi.w, p);
    return p;
}

// w4f32_row is one warp's dot product of an int4 row (wr, sr) with a; the full sum is in lane 0.
__device__ __forceinline__ float w4f32_row(const unsigned int* __restrict__ wr, const __half* __restrict__ sr,
                                           const float* __restrict__ a, int Kwords, int lane) {
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi0 = base + lane, wi1 = base + 32 + lane;
        unsigned int w0 = wr[wi0], w1 = wr[wi1];
        float p0 = w4f32_word(w0, a + (wi0 << 3));
        float p1 = w4f32_word(w1, a + (wi1 << 3));
        facc = __fmaf_rn(p0, __half2float(sr[wi0 >> 2]), facc);
        facc = __fmaf_rn(p1, __half2float(sr[wi1 >> 2]), facc);
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            float p = w4f32_word(wr[wi], a + (wi << 3));
            facc = __fmaf_rn(p, __half2float(sr[wi >> 2]), facc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc = __fadd_rn(facc, __shfl_down_sync(0xffffffffu, facc, off));
    return facc;
}

extern "C" {

// gemv_w4f32: dst[n] = sum_k dequant(W[n,k]) * a[k] + bias[n] (bias may be null); accum adds into dst. gemv_w4a8_fwd's
// arguments without the activation scale.
__global__ void gemv_w4f32(const unsigned int* __restrict__ W, const float* __restrict__ a, const __half* __restrict__ gs,
                           const float* __restrict__ bias, int N, int Kwords, int Kgroups, float* __restrict__ dst, int accum) {
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const unsigned int* wr = W + (long)n * Kwords;
    const __half* sr = gs + (long)n * Kgroups;
    float facc = w4f32_row(wr, sr, a, Kwords, lane);
    if (lane == 0) {
        float val = __fadd_rn(facc, bias ? bias[n] : 0.f);
        dst[n] = accum ? __fadd_rn(dst[n], val) : val;
    }
}

// gemv_w4f32_moe: gemv_w4f32 over a stacked-expert buffer, the weight row offset by the routed expert (or cache slot)
// idx[slot]: row idx[slot]*rowsPerExpert + n. gemv_w4a8_moe's arguments without the activation scale.
__global__ void gemv_w4f32_moe(const unsigned int* __restrict__ W, const float* __restrict__ a, const __half* __restrict__ gs,
                               const unsigned int* __restrict__ idx, int slot, int rowsPerExpert, int N, int Kwords, int Kgroups,
                               float* __restrict__ dst) {
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    long wrow = (long)idx[slot] * rowsPerExpert + n;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    float facc = w4f32_row(wr, sr, a, Kwords, lane);
    if (lane == 0) dst[n] = facc;
}

// gemv_w4f32_moe_wacc: gemv_w4f32_moe whose epilogue weight-accumulates, dst[n] += wgt[slot] * result: the expert combine.
__global__ void gemv_w4f32_moe_wacc(const unsigned int* __restrict__ W, const float* __restrict__ a, const __half* __restrict__ gs,
                                    const unsigned int* __restrict__ idx, const float* __restrict__ wgt, int slot, int rowsPerExpert,
                                    int N, int Kwords, int Kgroups, float* __restrict__ dst) {
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    long wrow = (long)idx[slot] * rowsPerExpert + n;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    float facc = w4f32_row(wr, sr, a, Kwords, lane);
    if (lane == 0) dst[n] = __fmaf_rn(wgt[slot], facc, dst[n]);
}

// glu_f32: dst[k] = act(g[gOff+k]) * u[uOff+k] as f32, glu_quant's activation (act takes glue.cu's values: 0 gelu-tanh,
// 1 SiLU) and product, without the int8 rounding that follows them there. One thread per element.
__global__ void glu_f32(const float* __restrict__ g, const float* __restrict__ u, int gOff, int uOff, int I, int act,
                        float* __restrict__ dst) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= I) return;
    float x = g[gOff + k], a;
    if (act == 1) { // SiLU
        a = __fdiv_rn(x, __fadd_rn(1.f, __expf(-x)));
    } else { // gelu-tanh: 0.7978845608028654 = sqrt(2/pi), as glu_quant and decoder/rmsnorm.go's geluTanh
        float inner = __fmul_rn(0.7978845608028654f, __fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x));
        a = __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, tanhf(inner)));
    }
    dst[k] = __fmul_rn(a, u[uOff + k]);
}

} // extern "C"
