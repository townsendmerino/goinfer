// rope_pairwise.cu — the GPT-J PAIRWISE rotary kernels: Cohere/Command-R, Cohere2/Command-R7B,
// Aya and GLM-OCR (every arch whose decoder.Architecture.ropeInterleave is set, except MLA, which
// carries its own interleave inside mla.cu).
//
// WHY THIS FILE EXISTS. rope_kv (gemv_fwd.cu), rope_kv_batched (prefill_batched.cu) and
// rope_kv_mrope_batched (rope_mrope_prefill.cu) rotate the NeoX HALF-SPLIT pairs (d, d+half).
// Cohere's HF rotary is rope_gptj: the rotated pairs are ADJACENT dims (2d, 2d+1), sharing frequency
// d. Same theta_d = pos * invFreq[d]; only which two elements form a pair differs. Running a
// pairwise family through the half-split kernels is exact at position 0 (rotation is the identity
// there) and wrong at every later position, with no error and fluent-looking logits. Measured
// 2026-10-01 on real Command-R7B and Aya-expanse-8B at int4 (resident vs CPU, same quant): worst
// per-position cosine -0.075 / -0.041 on a 48-token prompt, and the committed flat-weight gate
// could not see it (docs/measurements/cuda-pairwise-rope-2026-10-01.md).
//
// THE REFERENCE is decoder.applyRoPEInterleaved (decoder/rope.go), and for the m-RoPE variant
// decoder.applyMRoPEPairwise:
//   x1 = v[2d], x2 = v[2d+1];   out[2d] = x1*c - x2*s;   out[2d+1] = x2*c + x1*s
//   c, s = cos/sin(theta_d) * mscale  (YaRN attention_factor folded into cos/sin, as in the NeoX kernels)
// Partial rotary (rotaryDim < headDim) rotates the first 2*rhalf dims and passes the tail through to
// the KV cache, exactly as the NeoX kernels do.
//
// EVERY KERNEL HERE KEEPS ITS NeoX COUNTERPART'S ARGUMENT LIST AND LAUNCH GEOMETRY (grid, block,
// per-thread index space), so the Go side dispatches by swapping which Pipeline it binds for the
// model, and no launch site changes. The rotation arithmetic is the NeoX kernels' own
// (__fmaf_rn / __fmul_rn, same operand order); only the element indices differ.
//
// WHY ITS OWN .cu / .ptx. glue.ptx and gemv_fwd.ptx are AUDITED artifacts (NVRTC 12.6.85) and every
// other family's decode rests on them; they must stay byte-identical. Same isolation reason as
// rope_mrope_prefill.cu. Not reachable from here: the scalar `rope` kernel in glue.cu (bound by
// nothing in production, see TestPipelineLint_boundKernelsAreLaunched) has no pairwise twin.
//
// Regenerate with (NVRTC 12.6.85, see cuda/testdata/REGEN.md):  ./build_ptx.sh rope_pairwise

extern "C" {

// rope_kv_pw: rope_kv (gemv_fwd.cu) with pairwise pairs. One thread per (head, d < rhalf) of Q, then
// of K (which also stores K/V to the cache), then per un-rotated tail element.
__global__ void rope_kv_pw(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int pos, int ropePos, int rhalf, float mscale, float qTempScale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tail = hd - 2 * rhalf;
    int qn = nH * rhalf, kn = nKV * rhalf, tn = nKV * tail;
    int kvDim = nKV * hd;
    if (idx < qn) {
        int h = idx / rhalf, d = idx % rhalf;
        float ang = ropePos * invFreq[d];
        float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = q + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        base[2 * d] = __fmul_rn(__fmaf_rn(a, c, -__fmul_rn(b, s)), qTempScale);
        base[2 * d + 1] = __fmul_rn(__fmaf_rn(a, s, __fmul_rn(b, c)), qTempScale);
    } else if (idx < qn + kn) {
        int j = idx - qn;
        int h = j / rhalf, d = j % rhalf;
        float ang = ropePos * invFreq[d];
        float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = k + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        float r0 = __fmaf_rn(a, c, -__fmul_rn(b, s)), r1 = __fmaf_rn(a, s, __fmul_rn(b, c));
        base[2 * d] = r0;
        base[2 * d + 1] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + 2 * d] = r0;
        kc[o + 2 * d + 1] = r1;
        vc[o + 2 * d] = v[h * hd + 2 * d];
        vc[o + 2 * d + 1] = v[h * hd + 2 * d + 1];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn;
        int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = k[h * hd + t]; // un-rotated: pass through, but it MUST still be cached
        vc[o + t] = v[h * hd + t];
    }
}

// rope_kv_batched_pw: rope_kv_batched (prefill_batched.cu) with pairwise pairs. M tokens on grid.y,
// token m at absolute position startPos+m.
__global__ void rope_kv_batched_pw(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int startPos, int rhalf, int M, float mscale,
    const float* __restrict__ qTempRows)
{
    int m = blockIdx.y; if (m >= M) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = startPos + m;
    int tail = hd - 2 * rhalf;
    int qn = nH * rhalf, kn = nKV * rhalf, tn = nKV * tail;
    int qDim = nH * hd, kvDim = nKV * hd;
    float* qm = q + (long)m * qDim;
    float* km = k + (long)m * kvDim;
    const float* vm = v + (long)m * kvDim;
    if (idx < qn) {
        float qTempScale = qTempRows ? qTempRows[m] : 1.0f;
        int h = idx / rhalf, d = idx % rhalf;
        float ang = pos * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = qm + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        base[2 * d] = __fmul_rn(__fmaf_rn(a, c, -__fmul_rn(b, s)), qTempScale);
        base[2 * d + 1] = __fmul_rn(__fmaf_rn(a, s, __fmul_rn(b, c)), qTempScale);
    } else if (idx < qn + kn) {
        int j = idx - qn; int h = j / rhalf, d = j % rhalf;
        float ang = pos * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = km + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        float r0 = __fmaf_rn(a, c, -__fmul_rn(b, s)), r1 = __fmaf_rn(a, s, __fmul_rn(b, c));
        base[2 * d] = r0; base[2 * d + 1] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + 2 * d] = r0; kc[o + 2 * d + 1] = r1;
        vc[o + 2 * d] = vm[h * hd + 2 * d]; vc[o + 2 * d + 1] = vm[h * hd + 2 * d + 1];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn; int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = km[h * hd + t]; vc[o + t] = vm[h * hd + t];
    }
}

// rope_kv_mrope_batched_pw: rope_kv_mrope_batched (rope_mrope_prefill.cu) with pairwise pairs — GLM-OCR's
// image-block prefill. Frequency index d takes its position component from the CONTIGUOUS section
// blocks (d < sec0 temporal, d < sec1 height, else width), decoder.applyMRoPEPairwise's rule; pos
// still drives only the KV-cache store index. For a text row (pT == pH == pW) this is bit-identical
// to rope_kv_batched_pw at that position.
__global__ void rope_kv_mrope_batched_pw(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int startPos, int rhalf, int M, float mscale,
    const int* __restrict__ posT, const int* __restrict__ posH, const int* __restrict__ posW,
    int sec0, int sec1)
{
    int m = blockIdx.y; if (m >= M) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = startPos + m; // KV-CACHE STORE index — unchanged, row-sequential
    int pT = posT[m], pH = posH[m], pW = posW[m]; // this row's OWN (t,h,w) rotation triple
    int tail = hd - 2 * rhalf;
    int qn = nH * rhalf, kn = nKV * rhalf, tn = nKV * tail;
    int qDim = nH * hd, kvDim = nKV * hd;
    float* qm = q + (long)m * qDim;
    float* km = k + (long)m * kvDim;
    const float* vm = v + (long)m * kvDim;
    if (idx < qn) {
        int h = idx / rhalf, d = idx % rhalf;
        int p = (d < sec0) ? pT : (d < sec1 ? pH : pW);
        float ang = p * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = qm + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        base[2 * d] = __fmaf_rn(a, c, -__fmul_rn(b, s)); base[2 * d + 1] = __fmaf_rn(a, s, __fmul_rn(b, c));
    } else if (idx < qn + kn) {
        int j = idx - qn; int h = j / rhalf, d = j % rhalf;
        int p = (d < sec0) ? pT : (d < sec1 ? pH : pW);
        float ang = p * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = km + h * hd;
        float a = base[2 * d], b = base[2 * d + 1];
        float r0 = __fmaf_rn(a, c, -__fmul_rn(b, s)), r1 = __fmaf_rn(a, s, __fmul_rn(b, c));
        base[2 * d] = r0; base[2 * d + 1] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + 2 * d] = r0; kc[o + 2 * d + 1] = r1;
        vc[o + 2 * d] = vm[h * hd + 2 * d]; vc[o + 2 * d + 1] = vm[h * hd + 2 * d + 1];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn; int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = km[h * hd + t]; vc[o + t] = vm[h * hd + t];
    }
}

} // extern "C"
