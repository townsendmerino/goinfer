// gemv_fwd.cu — the LLM-SPECIFIC forward kernels: KV-cache store and fused RoPE+KV.
//
// The two GENERIC quantized GEMVs that used to live here (gemv_w4a8_fwd, gemv_w8a8_fwd)
// moved to aikit/gpu >= v0.4.0 (gemv_quant.cu) — the Phase-1b blob-split. They were only
// ever co-located with these, never entangled: no shared __device__ helpers, no common
// state. The decode path now loads them from gpu.QuantGEMVPTX (backend.go), so the
// quantized GEMV has ONE owner across both repos, exactly as linalg owns the CPU matmul.
//
// What stays here is what is genuinely about running an LLM rather than multiplying a
// quantized matrix: the KV cache layout and the rotary embedding.
#include <cuda_fp16.h>
extern "C" __global__ void kv_store(const float* __restrict__ src, float* __restrict__ cache, int pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cache[(long)pos * n + i] = src[i];
}
// rope_kv: fused rope(q) + rope(k) + kv_store(k) + kv_store(v) — 4 launches → 1.
// Same math AND same order as the separate kernels: each thread rotates its own (d, d+rhalf)
// pair and then stores exactly those elements, so there is no cross-thread dependency the
// launch barriers were providing.
//
// rhalf = rotaryDim/2 supports PARTIAL rotary (GLM/Phi: rotary_dim < head_dim), where only the
// first rotaryDim elements of each head rotate and the tail passes through unrotated. rhalf ==
// hd/2 is the full-rotary case and must stay BIT-IDENTICAL to the pre-partial kernel — the tail
// group below is then empty and every other thread does exactly what it did before.
//
// The decomposition is forced by two facts, and getting it wrong is silent:
//
//  - q must rotate IN PLACE (attention reads it), so q threads own a disjoint PAIR
//    (d, d+rhalf), d < rhalf. A thread-per-ELEMENT layout would race: thread d reads
//    base[d+rhalf] while thread d+rhalf writes it. Pairs never overlap; elements do.
//  - The pair threads only touch [0, 2*rhalf). When rhalf < hd/2 that leaves the tail
//    [2*rhalf, hd) NEVER STORED into the KV cache — the old kernel got full coverage only
//    because 2*(hd/2) == hd. Hence the third thread group, which copies the un-rotated tail of
//    k and v straight through. It reads only tail elements, which nobody rotates, so it cannot
//    race the k pair threads.
//
// Thread layout: [0,qn) rotate q | [qn,qn+kn) rotate k in place + store k,v for their pair |
// [qn+kn, +tn) store the un-rotated k,v tail. tn == 0 when rotary is full.
extern "C" __global__ void rope_kv(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int pos, int rhalf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tail = hd - 2 * rhalf; // un-rotated elements per head; 0 when rotary is full
    int qn = nH * rhalf, kn = nKV * rhalf, tn = nKV * tail;
    int kvDim = nKV * hd;
    if (idx < qn) {
        int h = idx / rhalf, d = idx % rhalf;
        float ang = pos * invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        float* base = q + h * hd;
        float a = base[d], b = base[d + rhalf];
        base[d] = a * c - b * s;
        base[d + rhalf] = a * s + b * c;
    } else if (idx < qn + kn) {
        int j = idx - qn;
        int h = j / rhalf, d = j % rhalf;
        float ang = pos * invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        float* base = k + h * hd;
        float a = base[d], b = base[d + rhalf];
        float r0 = a * c - b * s, r1 = a * s + b * c;
        base[d] = r0;
        base[d + rhalf] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + d] = r0;
        kc[o + d + rhalf] = r1;
        vc[o + d] = v[h * hd + d];
        vc[o + d + rhalf] = v[h * hd + d + rhalf];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn;
        int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = k[h * hd + t]; // un-rotated: pass through, but it MUST still be cached
        vc[o + t] = v[h * hd + t];
    }
}
