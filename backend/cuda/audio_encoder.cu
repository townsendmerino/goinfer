// audio_encoder.cu — the two kernels the CUDA Whisper-style encoder (cuda/audio_encoder.go) adds to the tower base: the exact GELU and the
// k=3 im2col that turns each of the encoder's two Conv1d stems into a GEMM.
//
// audio_gelu_erf is aikit's linalg.GELUContractInto written out with the same constants, the same two erf branches and the same fused and
// unfused steps (linalg/exp_contract.go: geluF32Contract, erfF32Contract, expF32Contract), so a device result equals the CPU contract kernel bit for bit
// (TestAudioGeluErf_bitsMatchContract). It is the exact x·Φ(x), not the tanh form tower_base's GELU computes. Every float multiply and add is an
// explicit intrinsic (kernel_fma_lint_test.go): the one place the CPU contract fuses it says so (__fmaf_rn).
//
// A separate .cu from tower_base.cu for the reason in testdata/REGEN.md: a new kernel is a new file, and the audited artifacts are not regenerated.

__device__ __forceinline__ float audio_exp_contract(float x)
{
    float z = __fmul_rn(x, 1.4426950408889634f);
    float t = __fadd_rn(z, 12582912.0f);
    float kf = __fsub_rn(t, 12582912.0f);
    int k = (int)kf;
    float r = -__fmaf_rn(kf, 0.693145751953125f, -x);   // x - kf*ln2Hi, one rounding
    r = -__fmaf_rn(kf, 1.428606820309417e-06f, -r);     // r - kf*ln2Lo
    float p = 1.9875691500e-4f;
    p = __fmaf_rn(p, r, 1.3981999507e-3f);
    p = __fmaf_rn(p, r, 8.3334519073e-3f);
    p = __fmaf_rn(p, r, 4.1665795894e-2f);
    p = __fmaf_rn(p, r, 1.6666665459e-1f);
    p = __fmaf_rn(p, r, 5.0000001201e-1f);
    p = __fmaf_rn(__fmaf_rn(p, r, 1.0f), r, 1.0f);
    int e = k + 127;
    if (e <= 0) return 0.0f;
    if (e >= 255) return __fmul_rn(__fmul_rn(p, __int_as_float((e - 1) << 23)), 2.0f);
    return __fmul_rn(p, __int_as_float(e << 23));
}

__device__ __forceinline__ float audio_erf_contract(float x)
{
    float a = x;
    bool neg = false;
    if (a < 0.0f) { neg = true; a = -a; }
    float r;
    if (a < 1.0f) {
        float z = __fmul_rn(a, a);
        float p = 1.3122532963802806e-08f;
        p = __fmaf_rn(p, z, -1.4503852223150468e-07f);
        p = __fmaf_rn(p, z, 1.4589169000933706e-06f);
        p = __fmaf_rn(p, z, -1.3227513227513228e-05f);
        p = __fmaf_rn(p, z, 1.0683760683760684e-04f);
        p = __fmaf_rn(p, z, -7.5757575757575758e-04f);
        p = __fmaf_rn(p, z, 4.6296296296296294e-03f);
        p = __fmaf_rn(p, z, -2.3809523809523808e-02f);
        p = __fmaf_rn(p, z, 1.0000000000000001e-01f);
        p = __fmaf_rn(p, z, -3.3333333333333331e-01f);
        p = __fmaf_rn(p, z, 1.0f);
        r = __fmul_rn(__fmul_rn(1.1283791670955126f, a), p);
    } else {
        float t = __fdiv_rn(1.0f, __fmaf_rn(0.3275911f, a, 1.0f));
        float q = 1.061405429f;
        q = __fmaf_rn(q, t, -1.453152027f);
        q = __fmaf_rn(q, t, 1.421413741f);
        q = __fmaf_rn(q, t, -0.284496736f);
        q = __fmaf_rn(q, t, 0.254829592f);
        float e = -__fmul_rn(a, a);
        if (e < -104.0f) e = -104.0f;
        r = __fmaf_rn(-__fmul_rn(q, t), audio_exp_contract(e), 1.0f);
    }
    return neg ? -r : r;
}

// x[i] = x[i] * Phi(x[i]) over n values, in place.
extern "C" __global__ void audio_gelu_erf(float* __restrict__ x, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float h = __fmul_rn(0.5f, v);
    x[i] = __fmul_rn(h, __fadd_rn(1.0f, audio_erf_contract(__fmul_rn(0.7071067811865476f, v))));
}

// dst[t, ch*3 + k] = src[ch*chStride + (t*stride - 1 + k)*tStride], zero where t*stride-1+k falls outside [0, Tin): a k=3, pad=1 Conv1d as a matrix, so
// out[Tout, Cout] = dst[Tout, C*3] · W[Cout, C*3]ᵀ with W in the checkpoint's [Cout][C][3] layout. n = Tout*C*3.
extern "C" __global__ void audio_im2col_k3(const float* __restrict__ src, float* __restrict__ dst, int Tin, int stride, int C,
                                           long chStride, long tStride, long n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    long row = g / (3L * C);
    int rem = (int)(g % (3L * C));
    int ch = rem / 3, k = rem % 3;
    long ti = row * stride - 1 + k;
    dst[g] = (ti < 0 || ti >= Tin) ? 0.0f : src[(long)ch * chStride + ti * tStride];
}
