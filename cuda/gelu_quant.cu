// gelu_quant.cu — GELU-tanh+int8-quantize for the resident SigLIP vision tower's plain
// (non-gated) 2-layer MLP (P6, docs/multimodal.md's "P6's other half"): h = FC2(GELU_tanh(FC1(x))).
// NOT glue.cu's glu_quant — that kernel computes act(gate)·up, a GATED product of TWO GEMV
// outputs, for the gated-SwiGLU/GeGLU MLP every text family here uses. SigLIP's own MLP has no
// gate: one hidden projection through the activation, straight back down. This kernel is glu_quant's
// single-input analogue, batched over M rows (rmsnorm_quant_batched's grid.x=M convention, which
// glu_quant itself does not use — it is called once per row already).
//
// GELU-tanh formula: the exact constant glue.cu's own glu_quant already uses (0.7978845608028654 =
// sqrt(2/pi)) — transcribed, not re-derived. Reference: aikit/vision/encoder.go's geluTanh
// (linalg.GELUTanhInto) — that comment's own contract ("not bit-identical; absolute error ≤1e-6")
// is the standard this kernel is held to as well, via the real-checkpoint gate's cosine/max-diff
// check (cuda/gemma3_vision_resident_real_test.go), not bit-identity.
//
// Own file — prefill_batched.ptx/glue.ptx stay untouched, same isolation reason as every other new
// kernel module this session.
//
// Regenerate with:  ./build_ptx.sh gelu_quant

extern "C" {

// gelu_quant_batched: M rows, one block per row (grid.x = M). x is FC1's raw output; q/scale are
// int8-quantized GELU_tanh(x), ready for FC2's GEMV.
__global__ void gelu_quant_batched(const float* __restrict__ x, int N,
                                   int* __restrict__ q, float* __restrict__ scale) {
    int m = blockIdx.x;
    const float* xr = x + (long)m * N;
    extern __shared__ float sm[];
    float* red = sm;                 // [blockDim]
    float* activated = sm + blockDim.x; // [N]
    int t = threadIdx.x, nt = blockDim.x;

    float amax = 0.f;
    for (int i = t; i < N; i += nt) {
        float v = xr[i];
        // 0.7978845608028654 = sqrt(2/pi), matching glue.cu's glu_quant ACT_GELU_TANH branch verbatim.
        float a = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(v, v), v), v))));
        activated[i] = a; amax = fmaxf(amax, fabsf(a));
    }
    // WARP-SHUFFLE MAXABS — same justification as rmsnorm_quant_batched/layernorm_quant_batched's
    // own: max is exact and order-independent.
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
    {
        int lane = t & 31, warp = t >> 5, nWarps = (nt + 31) >> 5;
        if (lane == 0) red[warp] = amax;
        __syncthreads();
        if (warp == 0) {
            float mv = (lane < nWarps) ? red[lane] : 0.f;
            for (int o = 16; o > 0; o >>= 1) mv = fmaxf(mv, __shfl_down_sync(0xffffffff, mv, o));
            if (lane == 0) red[0] = mv;
        }
        __syncthreads();
    }
    float sc = red[0] / 127.f; float invs = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) scale[m] = sc;
    int* qr = q + (long)m * (N / 4);
    for (int i4 = t; i4 < N / 4; i4 += nt) {
        int packed = 0;
        for (int bi = 0; bi < 4; bi++) {
            int qi = __float2int_rn(activated[i4 * 4 + bi] * invs);
            qi = max(-127, min(127, qi));
            packed |= (qi & 0xff) << (8 * bi);
        }
        qr[i4] = packed;
    }
}

} // extern "C"
