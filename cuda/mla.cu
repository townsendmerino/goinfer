// mla.cu — DeepSeek / Kimi Multi-head Latent Attention (MLA) resident kernels.
//
// Gated by FeatMLA on the cgo-free CUDA backend.
// Replaces per-head materialized K/V with a single compressed-KV latent row per position:
// [kv_lora_rank (rank) | qk_rope_head_dim (qkRope)]
//
// 1. mla_latent_store: RMSNorm(kvDown[:rank]) + decoupled RoPE(kvDown[rank:]) -> latCache[pos*latDim]
// 2. mla_head_matvec: per-head block-diagonal matvec for W_UK absorb and W_UV lift
// 3. mla_q_rope: decoupled RoPE on query's qkRope slice -> qAbs
// 4. mla_attn: causal attention over latDim keys, value accumulation over rank prefix ONLY.

#include <cuda_runtime.h>

extern "C" __global__ void mla_latent_store(const float* __restrict__ kvDown,
                                            const float* __restrict__ normW,
                                            const float* __restrict__ invFreq,
                                            float* __restrict__ latCache,
                                            int rank, int qkRope, int pos,
                                            float eps, int base, float ropeScale,
                                            int interleave) {
    int t = threadIdx.x;
    // RMSNorm over the rank latent -> cn
    float ss = 0.0f;
    for (int i = t; i < rank; i += 64) {
        float v = kvDown[i];
        ss = __fmaf_rn(v, v, ss);
    }
    __shared__ float sh[64];
    sh[t] = ss;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    float inv = rsqrtf(__fadd_rn(__fmul_rn(sh[0], 1.0f / (float)rank), eps));
    __syncthreads();
    for (int i = t; i < rank; i += 64) {
        latCache[base + i] = __fmul_rn(__fmul_rn(kvDown[i], inv), normW[i]);
    }
    // Decoupled RoPE the qkRope key -> krj, written directly after the rank block
    int half = qkRope / 2;
    for (int i = t; i < half; i += 64) {
        float theta = __fmul_rn((float)pos, invFreq[i]);
        float c = __fmul_rn(cosf(theta), ropeScale);
        float s = __fmul_rn(sinf(theta), ropeScale);
        float a, b;
        if (interleave == 1) {
            a = kvDown[rank + 2 * i];
            b = kvDown[rank + 2 * i + 1];
        } else {
            a = kvDown[rank + i];
            b = kvDown[rank + half + i];
        }
        latCache[base + rank + i]        = __fmaf_rn(a, c, -__fmul_rn(b, s));
        latCache[base + rank + half + i] = __fmaf_rn(b, c, __fmul_rn(a, s));
    }
}

extern "C" __global__ void mla_head_matvec(const float* __restrict__ a,
                                           const float* __restrict__ w,
                                           float* __restrict__ dst,
                                           int nH, int N, int K,
                                           int aStride, int outStride) {
    int elem = blockIdx.x + blockIdx.y * 32768;
    if (elem >= nH * N) return;
    int h = elem / N;
    int n = elem % N;
    int t = threadIdx.x;
    int abase = h * aStride;
    int wbase = (h * N + n) * K;
    float acc = 0.0f;
    for (int k = t; k < K; k += 64) {
        acc = __fmaf_rn(a[abase + k], w[wbase + k], acc);
    }
    __shared__ float red[64];
    red[t] = acc;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    if (t == 0) {
        dst[h * outStride + n] = red[0];
    }
}

extern "C" __global__ void mla_q_rope(const float* __restrict__ q,
                                      const float* __restrict__ invFreq,
                                      float* __restrict__ qAbs,
                                      int nH, int qkHead, int qkNope,
                                      int qkRope, int rank, int latDim,
                                      int pos, int interleave, float ropeScale) {
    int half = qkRope / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nH * half) return;
    int h = idx / half;
    int i = idx % half;
    int qbase = h * qkHead + qkNope;
    float theta = __fmul_rn((float)pos, invFreq[i]);
    float c = __fmul_rn(cosf(theta), ropeScale);
    float s = __fmul_rn(sinf(theta), ropeScale);
    float a, b;
    if (interleave == 1) {
        a = q[qbase + 2 * i];
        b = q[qbase + 2 * i + 1];
    } else {
        a = q[qbase + i];
        b = q[qbase + half + i];
    }
    int obase = h * latDim + rank;
    qAbs[obase + i]        = __fmaf_rn(a, c, -__fmul_rn(b, s));
    qAbs[obase + half + i] = __fmaf_rn(b, c, __fmul_rn(a, s));
}

extern "C" __global__ void mla_attn(const float* __restrict__ qAbs,
                                    const float* __restrict__ latCache,
                                    float* __restrict__ wsum,
                                    int nH, int latDim, int rank,
                                    int startPos, float scale, int window,
                                    int M) {
    int h = blockIdx.x; if (h >= nH) return;
    int m = blockIdx.y; if (m >= M) return;
    int nKeys = startPos + m + 1;
    int winStart = (window > 0 && nKeys > window) ? nKeys - window : 0;
    int nWin = nKeys - winStart;
    extern __shared__ float sm[];
    float* sc = sm;            // [nWin]
    float* red = sm + nWin;    // [blockDim.x]
    int qDim = nH * latDim;
    const float* qh = qAbs + (long)m * qDim + h * latDim;
    int t = threadIdx.x, nt = blockDim.x;
    float lm = -1e30f;
    int d4 = latDim >> 2;
    const float4* q4 = (const float4*)qh;
    for (int s = winStart + t; s < nKeys; s += nt) {
        const float* ks = latCache + (long)s * latDim;
        float dot = 0.0f;
        const float4* k4 = (const float4*)ks;
        for (int i = 0; i < d4; i++) {
            float4 qq = q4[i], kk = k4[i];
            dot = __fmaf_rn(qq.x, kk.x, dot);
            dot = __fmaf_rn(qq.y, kk.y, dot);
            dot = __fmaf_rn(qq.z, kk.z, dot);
            dot = __fmaf_rn(qq.w, kk.w, dot);
        }
        for (int d = d4 << 2; d < latDim; d++) {
            dot = __fmaf_rn(qh[d], ks[d], dot);
        }
        dot = __fmul_rn(dot, scale);
        sc[s - winStart] = dot;
        lm = fmaxf(lm, dot);
    }
    red[t] = lm;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) red[t] = fmaxf(red[t], red[t + o]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();
    float ls = 0.0f;
    for (int s = winStart + t; s < nKeys; s += nt) {
        float e = __expf(__fsub_rn(sc[s - winStart], mx));
        sc[s - winStart] = e;
        ls += e;
    }
    red[t] = ls;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();
    for (int d = t; d < rank; d += nt) {
        float acc = 0.0f;
        for (int s = winStart; s < nKeys; s++) {
            acc = __fmaf_rn(sc[s - winStart], latCache[(long)s * latDim + d], acc);
        }
        wsum[(long)m * (nH * rank) + h * rank + d] = __fmul_rn(acc, inv);
    }
}

