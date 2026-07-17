#include <cuda_fp16.h>
extern "C" __global__ void gemv_w4a8_fwd(
    const unsigned int* __restrict__ W, const int* __restrict__ a, const __half* __restrict__ gs,
    const float* __restrict__ aScalePtr, const float* __restrict__ bias,
    int N, int Kwords, int Kgroups, float* __restrict__ dst, int accum)
{
    // COALESCED + 2x ILP unroll: consecutive lanes read consecutive words (128B warp
    // transactions), two independent word loads in flight per lane to saturate the
    // byte-light int4 stream (43%->80% peak in isolation). even/odd + __vsub4 unpack on the
    // fast nibble-permuted layout (permuteFast at pack time); each word's int partial is
    // scaled by its group's f32 scale and float-accumulated (the group sum falls out of the
    // final warp reduce), so no per-word segmented reduction. 32-stride remainder tail.
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
        facc += (float)p0 * __half2float(sr[wi0 >> 2]);
        facc += (float)p1 * __half2float(sr[wi1 >> 2]);
    }
    // Tail: Kwords need NOT be a multiple of 32 (e.g. qwen2.5-0.5B H=896 → Kwords=112).
    // Guard per lane — the scale-per-word float accumulate has no cross-lane dependency, so
    // out-of-range lanes simply contribute nothing. Without this they read past the row.
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
            facc += (float)p * __half2float(sr[wi >> 2]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) {
        float val = facc * (*aScalePtr) + (bias ? bias[n] : 0.f);
        dst[n] = accum ? dst[n] + val : val;
    }
}
extern "C" __global__ void kv_store(const float* __restrict__ src, float* __restrict__ cache, int pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cache[(long)pos * n + i] = src[i];
}
// W8A8 forward GEMV: per-row f32 weight scale, on-device activation scale ptr, per-row bias.
// W is int8x4-packed (4 int8 per word, row-major), a is the same int8 activation (4/word).
extern "C" __global__ void gemv_w8a8_fwd(
    const int* __restrict__ W, const int* __restrict__ a, const float* __restrict__ wScale,
    const float* __restrict__ aScalePtr, const float* __restrict__ bias,
    int N, int Kdiv4, float* __restrict__ dst, int accum)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int* wr = W + (long)n * Kdiv4;
    int acc = 0;
    for (int k = lane; k < Kdiv4; k += 32) acc = __dp4a(wr[k], a[k], acc);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) {
        float val = (float)acc * wScale[n] * (*aScalePtr) + (bias ? bias[n] : 0.f);
        dst[n] = accum ? dst[n] + val : val;
    }
}

// rope_kv: fused rope(q) + rope(k) + kv_store(k) + kv_store(v) — 4 launches → 1.
// Same math AND same order as the separate kernels: each thread rotates its own (d, d+half)
// pair and then stores exactly those elements, so there is no cross-thread dependency the
// launch barriers were providing. Threads [0,nH*half) rotate q; threads [nH*half, +nKV*half)
// rotate k in place then store the rotated k and the (un-roped) v into the resident caches.
extern "C" __global__ void rope_kv(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int pos)
{
    int half = hd / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int qn = nH * half, kn = nKV * half;
    if (idx < qn) {
        int h = idx / half, d = idx % half;
        float ang = pos * invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        float* base = q + h * hd;
        float a = base[d], b = base[d + half];
        base[d] = a * c - b * s;
        base[d + half] = a * s + b * c;
    } else if (idx < qn + kn) {
        int j = idx - qn;
        int h = j / half, d = j % half;
        float ang = pos * invFreq[d];
        float c = cosf(ang), s = sinf(ang);
        float* base = k + h * hd;
        float a = base[d], b = base[d + half];
        float r0 = a * c - b * s, r1 = a * s + b * c;
        base[d] = r0;
        base[d + half] = r1;
        int kvDim = nKV * hd;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + d] = r0;
        kc[o + d + half] = r1;
        vc[o + d] = v[h * hd + d];
        vc[o + d + half] = v[h * hd + d + half];
    }
}
