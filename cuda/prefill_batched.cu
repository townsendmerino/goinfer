#include <cuda_fp16.h>

// prefill_batched.cu — the batched (M=len) counterparts of the per-token glue kernels, for the
// weight-stationary prefill path. Each is the existing M=1 kernel with an M dimension added; the
// per-row math is copied VERBATIM so each row is bit-identical to its sequential counterpart. The
// weight matmuls are gemv_w4a8_batched (its own file); these cover norms, rope+kv-store, attention,
// swiglu, and the residual. Own file — moe.ptx and the audited PTX (glue/gemv_fwd) are untouched.
//
// The two correctness seams that only exist at M>1 live in attn_batched:
//   - CAUSALITY: query row m (absolute position startPos+m) attends to keys [0, startPos+m] only —
//     nKeys_m = startPos+m+1, computed per block from blockIdx.y. Sequentially this was implicit
//     (future K/V didn't exist); batched, they do, so the per-row nKeys is the mask.
//   - SLIDING WINDOW: winStart_m = (window>0 && nKeys_m>window) ? nKeys_m-window : 0 — per row,
//     relative to that row's own position. A short prompt never exercises it.

extern "C" {

// rmsnorm_quant_batched: M rows, one block per row (grid.x = M). Copy of rmsnorm_quant per row.
__global__ void rmsnorm_quant_batched(const float* __restrict__ x, const float* __restrict__ w,
                                      int N, float eps, int addOne, int* __restrict__ q,
                                      float* __restrict__ scale) {
    int m = blockIdx.x;
    const float* xr = x + (long)m * N;
    extern __shared__ float sm[];
    float* red = sm;             // [blockDim]
    float* normed = sm + blockDim.x; // [N]
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int i = t; i < N; i += nt) ss = __fmaf_rn(xr[i], xr[i], ss);
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = rsqrtf(red[0] / N + eps); __syncthreads();
    // pass 2: normed + maxabs — multiply order x*g*rnorm copied verbatim (IEEE rounds per step).
    float amax = 0.f;
    for (int i = t; i < N; i += nt) {
        float g = addOne ? (1.f + w[i]) : w[i];
        float v = xr[i] * g * inv;
        normed[i] = v; amax = fmaxf(amax, fabsf(v));
    }
    red[t] = amax; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float s = red[0] / 127.f; float invs = s > 0.f ? 1.f / s : 0.f;
    if (t == 0) scale[m] = s;
    int* qr = q + (long)m * (N / 4);
    for (int i4 = t; i4 < N / 4; i4 += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) {
            int qi = __float2int_rn(normed[i4 * 4 + b] * invs);
            qi = max(-127, min(127, qi));
            packed |= (qi & 0xff) << (8 * b);
        }
        qr[i4] = packed;
    }
}

// rmsnorm_f32_batched: M rows (grid.y = m), plain in-place RMSNorm of a [H] sublayer output — no
// quant, lands straight in the f32 residual stream. A per-token copy of rmsnorm_f32 (glue.cu) for
// Gemma's sandwich post-attn / post-MLP norms; float reduction + (1+w) addOne byte-for-byte the M=1
// kernel's, so batched sandwich norm is bit-identical to the sequential normF32 applied per row.
extern "C" __global__ void rmsnorm_f32_batched(float* __restrict__ x, const float* __restrict__ w,
                                               int H, float eps, int addOne, int M) {
    int m = blockIdx.y; if (m >= M) return;
    extern __shared__ float rednf[];
    int t = threadIdx.x, nt = blockDim.x;
    float* xm = x + (long)m * H;
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss = __fmaf_rn(xm[k], xm[k], ss);
    rednf[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) rednf[t] += rednf[t + o]; __syncthreads(); }
    float rnorm = rsqrtf(rednf[0] / H + eps); __syncthreads();
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; xm[k] = xm[k] * rnorm * g; }
}

// qk_norm_batched: M tokens (grid.y = m, grid.x = head in [0, nH+nKV)). Per-head Q/K RMSNorm over
// hd, in place, BEFORE rope_kv_batched — a per-token copy of qk_norm (fused_qkv.cu). The DOUBLE
// sum-of-squares reduction and the (1+w) addOne path are byte-for-byte the M=1 kernel's, so batched
// qk-norm is bit-identical to the sequential decode qk-norm applied token by token (Qwen3 parity).
extern "C" __global__ void qk_norm_batched(
    float* __restrict__ q, float* __restrict__ k,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm,
    int nH, int nKV, int hd, float eps, int addOne, int M)
{
    int m = blockIdx.y; if (m >= M) return;
    extern __shared__ double qkred[];
    int h = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    long qDim = (long)nH * hd, kvDim = (long)nKV * hd;
    float* base;
    const float* w;
    if (h < nH) { base = q + (long)m * qDim + (long)h * hd;         w = qNorm; }
    else        { base = k + (long)m * kvDim + (long)(h - nH) * hd; w = kNorm; }

    double ss = 0.0;
    for (int i = t; i < hd; i += nt) ss += (double)base[i] * (double)base[i];
    qkred[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) qkred[t] += qkred[t + o]; __syncthreads(); }
    float inv = (float)(1.0 / sqrt(qkred[0] / (double)hd + (double)eps));
    for (int i = t; i < hd; i += nt) {
        float g = addOne ? (1.f + w[i]) : w[i];
        base[i] = (base[i] * inv) * g;
    }
}

// rope_kv_batched: M tokens (grid.y = m, grid.x*blockDim over the per-token index space). Rotates
// q[m]/k[m] in place and stores K/V at absolute position startPos+m — copy of rope_kv per token.
__global__ void rope_kv_batched(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int startPos, int rhalf, int M)
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
        int h = idx / rhalf, d = idx % rhalf;
        float ang = pos * invFreq[d]; float c = cosf(ang), s = sinf(ang);
        float* base = qm + h * hd;
        float a = base[d], b = base[d + rhalf];
        base[d] = __fmaf_rn(a, c, -__fmul_rn(b, s)); base[d + rhalf] = __fmaf_rn(a, s, __fmul_rn(b, c));
    } else if (idx < qn + kn) {
        int j = idx - qn; int h = j / rhalf, d = j % rhalf;
        float ang = pos * invFreq[d]; float c = cosf(ang), s = sinf(ang);
        float* base = km + h * hd;
        float a = base[d], b = base[d + rhalf];
        float r0 = __fmaf_rn(a, c, -__fmul_rn(b, s)), r1 = __fmaf_rn(a, s, __fmul_rn(b, c));
        base[d] = r0; base[d + rhalf] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + d] = r0; kc[o + d + rhalf] = r1;
        vc[o + d] = vm[h * hd + d]; vc[o + d + rhalf] = vm[h * hd + d + rhalf];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn; int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = km[h * hd + t]; vc[o + t] = vm[h * hd + t];
    }
}

// attn_batched: grid.x = head, grid.y = query row m. Each block runs the M=1 attention math for
// query m with nKeys = startPos+m+1 (CAUSAL) and its own window. Shared mem sized to the largest
// row's window by the host (maxNWin + blockDim). Bit-identical to attention() per row.
// `sinks` (gpt-oss) is a per-head logit with no key and no value: it joins the softmax MAX
// and the DENOMINATOR, never the numerator. NULL elsewhere, leaving the kernel bit-identical.
// It must match splitkv_softmax exactly — a sink applied at decode but not at prefill would
// present as drift partway through a sequence rather than as a missing term.
__global__ void attn_batched(const float* __restrict__ q, const float* __restrict__ kc,
                             const float* __restrict__ vc, int nH, int nKV, int hd, int startPos,
                             float scale, int window, int M, float* __restrict__ ctx,
                             const float* __restrict__ sinks) {
    int h = blockIdx.x; if (h >= nH) return;
    int m = blockIdx.y; if (m >= M) return;
    int nKeys = startPos + m + 1;
    int winStart = (window > 0 && nKeys > window) ? nKeys - window : 0;
    int nWin = nKeys - winStart;
    extern __shared__ float sm[];
    float* sc = sm;            // [nWin]
    float* red = sm + nWin;    // [blockDim]
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    int qDim = nH * hd;
    const float* qh = q + (long)m * qDim + h * hd;
    int t = threadIdx.x, nt = blockDim.x;
    float lm = -1e30f;
    for (int s = winStart + t; s < nKeys; s += nt) {
        const float* ks = kc + (long)s * kvDim + kvh * hd;
        float dot = 0.f;
        // float4 the K read: threads split over KEYS, so a scalar ks[d] read is stride-kvDim across the
        // warp → only ~22% of each 32-byte L1TEX sector used (ncu), and this kernel is L1TEX-throughput-
        // saturated. The float4 loads 16 contiguous bytes/lane → fills the sector, ~4.5× fewer sectors.
        // The four adds stay SEPARATE (dot += each), preserving attention()'s exact d-order — bit-identical
        // (only the load width changes, like the GEMV int2). Scalar remainder for hd not a multiple of 4.
        int d4 = hd >> 2;
        const float4* q4 = (const float4*)qh;
        const float4* k4 = (const float4*)ks;
        for (int i = 0; i < d4; i++) {
            float4 qq = q4[i], kk = k4[i];
            dot = __fmaf_rn(qq.x, kk.x, dot);
            dot = __fmaf_rn(qq.y, kk.y, dot);
            dot = __fmaf_rn(qq.z, kk.z, dot);
            dot = __fmaf_rn(qq.w, kk.w, dot);
        }
        for (int d = d4 << 2; d < hd; d++) dot = __fmaf_rn(qh[d], ks[d], dot);
        dot *= scale; sc[s - winStart] = dot; lm = fmaxf(lm, dot);
    }
    red[t] = lm; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sink = 0.f; bool hasSink = (sinks != nullptr);
    if (hasSink) { sink = sinks[h]; mx = fmaxf(mx, sink); }  // the sink competes for the max
    float ls = 0.f;
    for (int s = winStart + t; s < nKeys; s += nt) { float e = __expf(sc[s - winStart] - mx); sc[s - winStart] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    // Added once, after the reduction, to the DENOMINATOR only — the sink has no value vector.
    float denom = red[0]; if (hasSink) denom += __expf(sink - mx);
    float inv = 1.f / denom; __syncthreads();
    for (int d = t; d < hd; d += nt) {
        float acc = 0.f;
        for (int s = winStart; s < nKeys; s++) acc = __fmaf_rn(sc[s - winStart], vc[(long)s * kvDim + kvh * hd + d], acc);
        ctx[(long)m * qDim + h * hd + d] = acc * inv;
    }
}

#define ACT_GELU_TANH 0
#define ACT_SILU      1
// glu_quant_batched: M rows, one block per row. act(g)*u then symmetric int8 quant. Copy of glu_quant.
__global__ void glu_quant_batched(const float* __restrict__ g, const float* __restrict__ u,
                                  int gOff, int uOff, int I, int act, int* __restrict__ q,
                                  float* __restrict__ scale, float* __restrict__ dscr, int M) {
    int m = blockIdx.x;
    const float* gr = g + (long)m * I + gOff;
    const float* ur = u + (long)m * I + uOff;
    float* dr = dscr + (long)m * I;
    extern __shared__ float sm[];
    float* red = sm;
    int t = threadIdx.x, nt = blockDim.x;
    float amax = 0.f;
    for (int i = t; i < I; i += nt) {
        float x = gr[i], a;
        if (act == ACT_SILU) a = x / (1.f + __expf(-x));
        else a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x))));
        float d = a * ur[i]; dr[i] = d; amax = fmaxf(amax, fabsf(d));
    }
    red[t] = amax; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float s = red[0] / 127.f; float invs = s > 0.f ? 1.f / s : 0.f;
    if (t == 0) scale[m] = s;
    int* qr = q + (long)m * (I / 4);
    for (int i4 = t; i4 < I / 4; i4 += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) {
            int v = __float2int_rn(dr[i4 * 4 + b] * invs);
            v = max(-127, min(127, v));
            packed |= (v & 0xff) << (8 * b);
        }
        qr[i4] = packed;
    }
}

// quant_vec_batched: M rows, one block per row. Symmetric int8 quant of a vector — copy of quant_vec.
__global__ void quant_vec_batched(const float* __restrict__ x, int N, int* __restrict__ q,
                                  float* __restrict__ scale, int M) {
    int m = blockIdx.x;
    const float* xr = x + (long)m * N;
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < N; k += nt) ma = fmaxf(ma, fabsf(xr[k]));
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) scale[m] = sc;
    int* qr = q + (long)m * (N / 4);
    for (int j = t; j < N / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) {
            int v = __float2int_rn(xr[4 * j + b] * inv); v = max(-127, min(127, v));
            packed |= (v & 0xff) << (8 * b);
        }
        qr[j] = packed;
    }
}

// residual_batched: x[M,N] += y[M,N], flat grid over M*N.
__global__ void residual_batched(float* __restrict__ x, const float* __restrict__ y, int MN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MN) x[i] += y[i];
}

} // extern "C"
