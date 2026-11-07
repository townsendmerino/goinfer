// kv8_store.cu — the quantizing KV store of the int8 cache (docs/measurements/cuda-kv-i8-2026-11/scope.md).
//
// The int8 cache holds, per (position, KV head), hd signed bytes and ONE f32 scale for K and one for V: the scheme the CPU, Metal and WebGPU
// caches use (decoder/kvcache.go quantizeHeads, aikit linalg.QuantizeRowInt8): scale = maxabs/127 in f32, q = round-half-away(x * (1/scale)) clamped to
// [-127, 127], and an all-zero head gets scale 1 and zeros. The rope/store kernels cannot do this themselves (a head's scale needs the absmax over its hd
// values, which their per-pair threads do not see), so in an i8 build they leave K rotated in place in their f32 K buffer and V in its own, and this
// kernel quantizes both into the pools.
//
// grid (nKV, M), block 32 (one warp per (token, head)); k and v are [M][nKV*hd] f32; token m is stored at absolute position startPos + m through the block
// table kvt, and the scales at the pools' row-major [physical row][nKV + head], laid out like the cache (kvpage.cuh).
#include "kvpage.cuh"

extern "C" __global__ void kv8_store(const float* __restrict__ k, const float* __restrict__ v, signed char* __restrict__ kc, signed char* __restrict__ vc,
                                     float* __restrict__ kscl, float* __restrict__ vscl, const int* __restrict__ kvt, int nKV, int hd, int startPos, int M) {
    const int h = blockIdx.x, m = blockIdx.y;
    if (h >= nKV || m >= M) return;
    const int lane = threadIdx.x;
    const int kvDim = nKV * hd;
    const long row = KV_ROW(kvt, startPos + m);
    const long rowOff = row * kvDim, headOff = (long)h * hd, srcOff = (long)m * kvDim;
    for (int which = 0; which < 2; which++) {
        const float* src = (which == 0 ? k : v) + srcOff + headOff;
        signed char* dst = (which == 0 ? kc : vc) + rowOff + headOff;
        float ma = 0.f;
        for (int d = lane; d < hd; d += 32) ma = fmaxf(ma, fabsf(src[d]));
        for (int o = 16; o > 0; o >>= 1) ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, o));
        float sc = 1.f, inv = 0.f;
        if (ma != 0.f) { sc = ma / 127.f; inv = 1.f / sc; }
        for (int d = lane; d < hd; d += 32) {
            float x = roundf(src[d] * inv);
            x = fminf(127.f, fmaxf(-127.f, x));
            dst[d] = (signed char)(int)x;
        }
        if (lane == 0) (which == 0 ? kscl : vscl)[row * nKV + h] = sc;
    }
}
