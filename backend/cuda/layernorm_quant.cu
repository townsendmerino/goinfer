// layernorm_quant.cu — LayerNorm+int8-quantize for the resident SigLIP vision tower (P6,
// docs/multimodal.md's "P6's other half"). No text family in this codebase uses LayerNorm (every
// one is RMSNorm — mean-free, weight-only), so this is a genuinely new primitive, not a widened
// copy of an existing kernel the way attn_img_prefill.cu/rope_mrope_prefill.cu are.
//
// Reference: aikit/vision/encoder.go's layerNormInto — mean and variance accumulated over the row,
// then out[d] = (x[d]-mean)*invstd*w[d] + b[d]. The CPU reference accumulates mean/variance in
// float64; this kernel accumulates in float32 (matching rmsnorm_quant_batched's own float32
// sum-of-squares, prefill_batched.cu) — NOT bit-identical to the CPU path (the vision tower's own
// GELU-tanh kernel already documents the same "not bit-identical, cosine/max-diff gated" contract,
// aikit/vision/encoder.go's geluTanh comment), only cosine/max-diff-close, which is what the
// real-checkpoint gate (cuda/gemma3_vision_resident_real_test.go) actually checks.
//
// Own file — prefill_batched.ptx/glue.ptx (every kernel every batched-prefill parity gate rests
// on) stay untouched, same isolation reason as every other new kernel module this session.
//
// Regenerate with:  ./build_ptx.sh layernorm_quant

extern "C" {

// layernorm_quant_batched: M rows, one block per row (grid.x = M). Two-pass reduction (mean, then
// variance+quantize) mirroring rmsnorm_quant_batched's shape (prefill_batched.cu) but with a THIRD
// reduction pass (mean) rmsnorm doesn't need, and a bias term rmsnorm doesn't have.
__global__ void layernorm_quant_batched(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ b, int N, float eps,
                                        int* __restrict__ q, float* __restrict__ scale) {
    int m = blockIdx.x;
    const float* xr = x + (long)m * N;
    extern __shared__ float sm[];
    float* red = sm;                  // [blockDim]
    float* normed = sm + blockDim.x;  // [N]
    int t = threadIdx.x, nt = blockDim.x;

    // Pass 1: mean.
    float s = 0.f;
    for (int i = t; i < N; i += nt) s += xr[i];
    red[t] = s; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float mean = red[0] / N; __syncthreads();

    // Pass 2: variance.
    float ss = 0.f;
    for (int i = t; i < N; i += nt) { float d = xr[i] - mean; ss = __fmaf_rn(d, d, ss); }
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = rsqrtf(red[0] / N + eps); __syncthreads();

    // Pass 3: normed + maxabs.
    float amax = 0.f;
    for (int i = t; i < N; i += nt) {
        float v = __fmaf_rn(__fmul_rn(xr[i] - mean, inv), w[i], b[i]);
        normed[i] = v; amax = fmaxf(amax, fabsf(v));
    }
    // WARP-SHUFFLE MAXABS — same justification as rmsnorm_quant_batched's own (prefill_batched.cu):
    // max is exact and order-independent, so the reduction tree may be restructured freely.
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
            int qi = __float2int_rn(normed[i4 * 4 + bi] * invs);
            qi = max(-127, min(127, qi));
            packed |= (qi & 0xff) << (8 * bi);
        }
        qr[i4] = packed;
    }
}

// layernorm_f32_batched: M rows (grid.y = m), plain (no quantize) LayerNorm — the tower's final
// post-LN, whose output feeds the projector/downstream consumer directly, never another GEMV.
__global__ void layernorm_f32_batched(float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ b, int N, float eps, int M) {
    int m = blockIdx.y; if (m >= M) return;
    float* xr = x + (long)m * N;
    extern __shared__ float red[]; // [blockDim]
    int t = threadIdx.x, nt = blockDim.x;

    float s = 0.f;
    for (int i = t; i < N; i += nt) s += xr[i];
    red[t] = s; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float mean = red[0] / N; __syncthreads();

    float ss = 0.f;
    for (int i = t; i < N; i += nt) { float d = xr[i] - mean; ss = __fmaf_rn(d, d, ss); }
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = rsqrtf(red[0] / N + eps); __syncthreads();

    for (int i = t; i < N; i += nt) xr[i] = __fmaf_rn(__fmul_rn(xr[i] - mean, inv), w[i], b[i]);
}

} // extern "C"
