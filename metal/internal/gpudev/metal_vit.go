//go:build darwin

package gpudev

import "fmt"

// metal_vit.go is the Metal mirror of cuda_vit.go — the transformer-encoder kernel
// set a ViT forward needs on top of the metal.go device layer (docs/task-native-gpu.md,
// Phase 3). The exported surface (ViT, NewViT, the kernel-name consts, ViTBlock) is
// identical to the CUDA side: they are build-tag mutually exclusive, so a consumer that
// only touches this surface stays platform-agnostic — the whole point of the substrate.
//
// Every kernel mirrors vision/encoder.go's arithmetic exactly, EXCEPT for the one thing
// that does not port: MSL has no `double`. The CUDA kernels accumulate LayerNorm
// mean/variance, softmax sums, and GELU in double to match the CPU tower's float64 math;
// here those reductions run in f32 with a pairwise (tree) reduction over threadgroup
// memory, which is the best f32 can do. That costs a little precision — see
// metal_vit_test.go for the achieved per-kernel tolerances and why they differ from the
// CUDA bars. The tanh-GELU formula, the max-subtract softmax, and the bidirectional
// 1/sqrt(hd) attention are otherwise identical.
//
// Metal ↔ CUDA divergences baked into these kernels (inverses of gpu/cuda.go's):
//   - dispatchThreads launches EXACTLY n threads, so no per-kernel bounds check.
//   - per-row / per-(head,query) kernels use one threadgroup per unit: the CUDA
//     blockIdx.x is threadgroup_position_in_grid, threadIdx.x is
//     thread_position_in_threadgroup, blockDim.x is threads_per_threadgroup.
//   - scalars bind as 1-element buffers (there is no by-value arg on Metal).

// vitMSL is the encoder kernel set, compiled at load by NewViT (Metal compiles MSL at
// runtime, so unlike the CUDA side there is no embedded PTX). LNBLOCK matches ViTBlock,
// and — like ViTBlock — is a BIT-IDENTITY dependency, not just an array size: it is the
// width of the cross-thread f32 sum reductions below (layernorm mean/variance, softmax
// sum), and f32 addition is not associative, so changing it changes the bits. Keep it
// equal to the host ViTBlock and do not sweep either for performance without
// re-baselining the parity gate.
const vitMSL = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define LNBLOCK 256

// quant_rows: per-row symmetric int8 quant, byte-identical to linalg.quantizeRowInt8.
// One threadgroup per row. round() is half-away-from-zero, matching Go's math.Round.
kernel void quant_rows(
    device const float* x [[buffer(0)]], device char* q [[buffer(1)]], device float* s [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];
    float m = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float a = abs(xr[i]); if (a > m) m = a; }
    sm[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) {
        if (tid < o && sm[tid + o] > sm[tid]) sm[tid] = sm[tid + o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxAbs = sm[0];
    float scale = maxAbs / 127.0f;
    if (tid == 0) s[r] = (maxAbs == 0.0f) ? 0.0f : scale;
    device char* qr = q + (uint)r * (uint)dim;
    if (maxAbs == 0.0f) { for (int i = tid; i < dim; i += tgsz) qr[i] = 0; return; }
    float inv = 1.0f / scale;
    for (int i = tid; i < dim; i += tgsz) {
        float v = round(xr[i] * inv);
        if (v > 127.0f) v = 127.0f; else if (v < -127.0f) v = -127.0f;
        qr[i] = (char)v;
    }
}

// gemm_w8a8: C[M,N] = (Aq[M,K].Bq[N,K]) * aScale[M] * bScale[N]. Exact int32 dot.
kernel void gemm_w8a8(
    device const char* A [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const char* B [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid, m = g / N, n = g % N;
    device const char* ar = A + (uint)m * (uint)K;
    device const char* br = B + (uint)n * (uint)K;
    int acc = 0;
    for (int k = 0; k < K; k++) acc += (int)ar[k] * (int)br[k];
    C[g] = (float)acc * aScale[m] * bScale[n];
}

// gemm_f32: C[M,N] = A[M,K].B[N,K] in f32 (B row-major = B-transposed).
kernel void gemm_f32(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid, m = g / N, n = g % N;
    device const float* ar = A + (uint)m * (uint)K;
    device const float* br = B + (uint)n * (uint)K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += ar[k] * br[k];
    C[g] = acc;
}

// add_bias: x[r,d] += bias[d].
kernel void add_bias(
    device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
    constant int& rows [[buffer(2)]], constant int& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    x[gid] += bias[(int)gid % dim];
}

// add_vec: x[i] += v[i].
kernel void add_vec(
    device float* x [[buffer(0)]], device const float* v [[buffer(1)]], constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    x[gid] += v[gid];
}

// layernorm: mean/variance LayerNorm (not RMS), one threadgroup per row, f32 pairwise
// reduction (the double the CUDA kernel uses does not exist in MSL). eps is added to the
// variance. rsqrt(var+eps) == 1/sqrt(var+eps).
kernel void layernorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& rows [[buffer(4)]], constant int& dim [[buffer(5)]], constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    device float* dst = out + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];

    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) acc += xr[i];
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean = sm[0] / (float)dim;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float d = xr[i] - mean; acc += d * d; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float var = sm[0] / (float)dim;
    float inv = rsqrt(var + eps);

    for (int i = tid; i < dim; i += tgsz) dst[i] = ((xr[i] - mean) * inv) * w[i] + b[i];
}

// gelu_tanh: the tanh approximation, in f32 (precise::tanh), matching geluTanh's formula.
kernel void gelu_tanh(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float v = x[gid];
    x[gid] = 0.5f * v * (1.0f + precise::tanh(c * (v + 0.044715f * v * v * v)));
}

// attention: bidirectional multi-head self-attention, ONE threadgroup per (head, query).
// The per-query score row is staged in dynamic threadgroup memory (sc, np floats, bound
// at index 0 by the caller); max and sum reduce through static threadgroup arrays.
// Softmax sum is f32 (the CUDA double does not port).
kernel void attention(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& np [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* sc [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    int h = (int)blk / np, i = (int)blk % np;
    int hidden = nH * hd, off = h * hd;
    device const float* qi = q + (uint)i * (uint)hidden + off;
    threadgroup float qs[128];
    if (hd <= 128) {
        for (int d = tid; d < hd; d += tgsz) qs[d] = qi[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tid; j < np; j += tgsz) {
            device const float* kj = k + (uint)j * (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qs[d] * kj[d];
            sc[j] = acc * scale;
        }
    } else {
        for (int j = tid; j < np; j += tgsz) {
            device const float* kj = k + (uint)j * (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
            sc[j] = acc * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float smax[LNBLOCK];
    threadgroup float ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int j = tid; j < np; j += tgsz) if (sc[j] > m) m = sc[j];
    smax[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o && smax[tid + o] > smax[tid]) smax[tid] = smax[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx = smax[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = 0.0f;
    for (int j = tid; j < np; j += tgsz) { float e = exp(sc[j] - mx); sc[j] = e; su += e; }
    ssum[tid] = su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) ssum[tid] += ssum[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / ssum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tid; j < np; j += tgsz) sc[j] = sc[j] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oi = out + (uint)i * (uint)hidden + off;
    for (int d = tid; d < hd; d += tgsz) {
        float acc = 0.0f;
        for (int j = 0; j < np; j++) acc += sc[j] * v[(uint)j * (uint)hidden + off + d];
        oi[d] = acc;
    }
}

// attention_tiled — query-tiled, online-softmax (flash-attention-style) bidirectional
// self-attention (M-15, audit-metal-2026-09-12.md). attention above re-reads the FULL
// K and V arrays from device memory once PER QUERY (one threadgroup per (head,query)) —
// ≈4.2 TB/image at so400m's np=4096 against a 37.7 MB/layer minimum. This groups
// AT_QTILE queries into ONE threadgroup so a K/V chunk, once staged into threadgroup
// memory, is reused across every query in the tile instead of being re-read from device
// memory once per query.
//
// A naive query tile would still need an AT_QTILE×np score matrix resident (staging K/V
// alone does not help if every query still needs its WHOLE row before it can normalize)
// — that is why attention is one threadgroup per query in the first place, and an
// AT_QTILE×np buffer does not fit threadgroup memory at ViT patch counts. The way past
// that is an ONLINE softmax: process K/V in AT_KTILE-sized chunks, keeping a running max
// (m), running sum (l), and running weighted-V accumulator (acc[hd]) PER QUERY, rescaling
// the running sum/accumulator by exp(old_max-new_max) whenever a new chunk raises the
// max. This RE-ASSOCIATES the softmax sum — the same normalized result mathematically,
// but not bit-identical to attention's single-pass max-subtract-whole-row reduction.
// See metal_vit_test.go for the (freshly derived, necessarily looser) tolerance this
// needs against the float64 CPU reference, and its own comment for why a fresh bar is
// the right response to a re-association rather than a red flag.
//
// One thread per query (not a cooperative reduction across the whole threadgroup, unlike
// attention): AT_QTILE threads per threadgroup, each running its own online-softmax
// recurrence serially over hd — simpler to get right than splitting one query's work
// across threads, at some cost to per-query parallelism; a documented, deliberately
// deferred tuning opportunity, not a correctness concern.
//
// AT_MAXHD bounds hd for the STATIC K/V staging arrays below (16 KiB total at
// AT_KTILE=16 — comfortably under the ~32 KiB Apple GPU threadgroup budget, so unlike
// the dynamic-memory kernels elsewhere in this file this needs no runtime budget check
// for the allocation itself). A tower with hd > AT_MAXHD cannot use this kernel at all —
// see AttentionTiledEligible below, which callers MUST check before dispatching this
// rather than after a silent wrong answer or an out-of-bounds write into Ks/Vs.
//
// Wired into gpu/visionmetal's and gpu/qwenmetal's attn calls via
// AttentionTiledEligible(hd) plus the AttentionTiledMinNP crossover (see cuda_vit.go);
// verified standalone (metal_vit_w8a8reg_test.go's sibling — see
// TestMetal_vitAttentionTiled).
#define AT_MAXHD 128
#define AT_KTILE 16
#define AT_QTILE 32

kernel void attention_tiled(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& np [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float Ks[AT_KTILE][AT_MAXHD];
    threadgroup float Vs[AT_KTILE][AT_MAXHD];

    int tilesPerHead = (np + AT_QTILE - 1) / AT_QTILE;
    int h = (int)blk / tilesPerHead;
    int tileIdx = (int)blk % tilesPerHead;
    int qStart = tileIdx * AT_QTILE;
    int i = qStart + (int)tid;
    int hidden = nH * hd, off = h * hd;
    bool active = i < np; // the last tile in a head is ragged when np % AT_QTILE != 0

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float acc[AT_MAXHD];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;

    for (int k0 = 0; k0 < np; k0 += AT_KTILE) {
        int chunk = min(AT_KTILE, np - k0);
        // Cooperative stage: every thread in the threadgroup participates regardless of
        // active, since the staged chunk serves every query in the tile.
        for (int idx = (int)tid; idx < chunk * hd; idx += (int)tgsz) {
            int kk = idx / hd, d = idx % hd;
            Ks[kk][d] = k[(uint)(k0 + kk) * (uint)hidden + (uint)off + (uint)d];
            Vs[kk][d] = v[(uint)(k0 + kk) * (uint)hidden + (uint)off + (uint)d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            device const float* qi = q + (uint)i * (uint)hidden + (uint)off;
            float s[AT_KTILE];
            float blockMax = -3.402823466e+38f;
            for (int kk = 0; kk < chunk; kk++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qi[d] * Ks[kk][d];
                float sv = dot * scale;
                s[kk] = sv;
                if (sv > blockMax) blockMax = sv;
            }
            float newMax = max(m, blockMax);
            float corr = exp(m - newMax);
            l *= corr;
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            float blockSum = 0.0f;
            for (int kk = 0; kk < chunk; kk++) {
                float e = exp(s[kk] - newMax);
                blockSum += e;
                for (int d = 0; d < hd; d++) acc[d] += e * Vs[kk][d];
            }
            l += blockSum;
            m = newMax;
        }
        // Barrier unconditional (even for inactive threads): the NEXT iteration's stage
        // above overwrites Ks/Vs, and every thread — active or not — participates in
        // that stage, so every thread must reach this point before any thread proceeds.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        device float* oi = out + (uint)i * (uint)hidden + (uint)off;
        float inv = 1.0f / l;
        for (int d = 0; d < hd; d++) oi[d] = acc[d] * inv;
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5-VL ViT additions (Phase 3) — the Metal mirror of vit.cu's Qwen section,
// the five kernels Qwen needs on top of the shared SigLIP set. Written against
// vision/qwen_encoder.go's arithmetic, same as the CUDA side.
//
// gelu_erf below is a DIFFERENT function from gelu_tanh above: Qwen's patch merger
// uses nn.GELU()'s exact erf form while SigLIP uses the tanh approximation. They
// differ by ~5e-4, far above any parity bar — shipping one for both is a silent
// numeric bug, so both are present and callers pick deliberately.
// ---------------------------------------------------------------------------

// rmsnorm: weight-only RMS normalization (no mean subtraction, no bias) —
// x * rsqrt(mean(x^2) + eps) * w, one threadgroup per row. The mean-square reduces
// in f32 (the double vit.cu uses does not port); the divide/rsqrt stay exact because
// the library is compiled fast-math OFF (CompileLibraryPrecise) — the same guard that
// kept the quant scale from drifting a ULP on the SigLIP set.
kernel void rmsnorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    device float* dst = out + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float vv = xr[i]; acc += vv * vv; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += tgsz) dst[i] = (xr[i] * inv) * w[i];
}

// rope_qk: NeoX rotate_half 2D rotary applied IN PLACE to the q and k thirds of a fused
// qkv buffer [seq, 3*hidden] (row layout [3, nH, hd]); v is untouched. One thread per
// (patch, head, d<hd/2), rotating both q and k. dispatchThreads launches exactly the
// seq*nH*(hd/2) threads, so no bounds check.
kernel void rope_qk(
    device float* qkv [[buffer(0)]], device const float* cos [[buffer(1)]], device const float* sin [[buffer(2)]],
    constant int& seq [[buffer(3)]], constant int& nH [[buffer(4)]], constant int& hd [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int hf = hd / 2; // named hf, not half: half is MSL's f16 type keyword.
    int g = (int)gid;
    int d = g % hf;
    int rest = g / hf;
    int head = rest % nH;
    int i = rest / nH;
    int hidden = nH * hd;
    device const float* co = cos + (uint)i * (uint)hd;
    device const float* si = sin + (uint)i * (uint)hd;
    uint qoff = (uint)i * 3u * (uint)hidden + (uint)head * (uint)hd;
    uint koff = qoff + (uint)hidden;
    float x = qkv[qoff + d], y = qkv[qoff + d + hf];
    qkv[qoff + d] = x * co[d] - y * si[d];
    qkv[qoff + d + hf] = y * co[d + hf] + x * si[d + hf];
    x = qkv[koff + d]; y = qkv[koff + d + hf];
    qkv[koff + d] = x * co[d] - y * si[d];
    qkv[koff + d + hf] = y * co[d + hf] + x * si[d + hf];
}

// attention_seg: bidirectional MHA restricted to each patch's segment (a window for
// most blocks, a whole image for the fullatt blocks). Reads q/k/v from the FUSED qkv
// buffer [seq, 3*hidden] at offsets 0/hidden/2*hidden; the per-query score row stages
// in dynamic threadgroup memory (sc, maxSeg floats, bound at index 0 by the caller).
// segStart/segEnd are PER-PATCH bounds, not cu_seqlens. One threadgroup per (head,
// query). Softmax sum is f32 (the vit.cu double does not port).
kernel void attention_seg(
    device const float* qkv [[buffer(0)]], device float* out [[buffer(1)]],
    device const int* segStart [[buffer(2)]], device const int* segEnd [[buffer(3)]],
    constant int& seq [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* sc [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    int h = (int)blk / seq, i = (int)blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    device const float* qi = qkv + (uint)i * 3u * (uint)hidden + off;
    threadgroup float qs[128];
    if (hd <= 128) {
        for (int d = tid; d < hd; d += tgsz) qs[d] = qi[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = tid; t < n; t += tgsz) {
            device const float* kj = qkv + (uint)(s0 + t) * 3u * (uint)hidden + (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qs[d] * kj[d];
            sc[t] = acc * scale;
        }
    } else {
        for (int t = tid; t < n; t += tgsz) {
            device const float* kj = qkv + (uint)(s0 + t) * 3u * (uint)hidden + (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
            sc[t] = acc * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float smax[LNBLOCK];
    threadgroup float ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = tid; t < n; t += tgsz) if (sc[t] > m) m = sc[t];
    smax[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o && smax[tid + o] > smax[tid]) smax[tid] = smax[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx = smax[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = 0.0f;
    for (int t = tid; t < n; t += tgsz) { float e = exp(sc[t] - mx); sc[t] = e; su += e; }
    ssum[tid] = su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) ssum[tid] += ssum[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / ssum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = tid; t < n; t += tgsz) sc[t] = sc[t] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oi = out + (uint)i * (uint)hidden + off;
    for (int d = tid; d < hd; d += tgsz) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) acc += sc[t] * qkv[(uint)(s0 + t) * 3u * (uint)hidden + 2u * (uint)hidden + off + d];
        oi[d] = acc;
    }
}

// attention_seg_tiled — query-tiled online-softmax attention restricted to per-patch
// cu_seqlens segment bounds [segStart[i], segEnd[i]].
//
// Unlike attention_seg which stages an entire segment score row in dynamic threadgroup
// memory (which exceeds Apple Silicon's 32 KiB threadgroup ceiling at maxSeg > 7000),
// this uses a constant 16 KiB threadgroup allocation (Ks and Vs tiles of AT_KTILE=16),
// lifting the threadgroup memory limit completely and allowing arbitrarily large images.
kernel void attention_seg_tiled(
    device const float* qkv [[buffer(0)]], device float* out [[buffer(1)]],
    device const int* segStart [[buffer(2)]], device const int* segEnd [[buffer(3)]],
    constant int& seq [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float Ks[AT_KTILE][AT_MAXHD];
    threadgroup float Vs[AT_KTILE][AT_MAXHD];

    int tilesPerHead = (seq + AT_QTILE - 1) / AT_QTILE;
    int h = (int)blk / tilesPerHead;
    int tileIdx = (int)blk % tilesPerHead;
    int qStart = tileIdx * AT_QTILE;
    int i = qStart + (int)tid;
    int hidden = nH * hd, off = h * hd;
    bool active = (i < seq);

    int s0 = active ? segStart[i] : 0;
    int s1 = active ? segEnd[i] : 0;

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float acc[AT_MAXHD];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;

    for (int k0 = 0; k0 < seq; k0 += AT_KTILE) {
        int chunk = min(AT_KTILE, seq - k0);
        for (int idx = (int)tid; idx < chunk * hd; idx += (int)tgsz) {
            int kk = idx / hd, d = idx % hd;
            int kj_idx = k0 + kk;
            Ks[kk][d] = qkv[(uint)kj_idx * 3u * (uint)hidden + (uint)hidden + (uint)off + (uint)d];
            Vs[kk][d] = qkv[(uint)kj_idx * 3u * (uint)hidden + 2u * (uint)hidden + (uint)off + (uint)d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            int chunkStart = k0;
            int chunkEnd = k0 + chunk;
            int validStart = max(s0, chunkStart);
            int validEnd = min(s1, chunkEnd);

            if (validStart < validEnd) {
                device const float* qi = qkv + (uint)i * 3u * (uint)hidden + off;
                float s[AT_KTILE];
                float blockMax = -3.402823466e+38f;
                for (int kk = 0; kk < chunk; kk++) {
                    int kIdx = k0 + kk;
                    if (kIdx >= s0 && kIdx < s1) {
                        float dot = 0.0f;
                        for (int d = 0; d < hd; d++) dot += qi[d] * Ks[kk][d];
                        float sv = dot * scale;
                        s[kk] = sv;
                        if (sv > blockMax) blockMax = sv;
                    }
                }
                if (blockMax > -3.402823466e+38f) {
                    float newMax = max(m, blockMax);
                    float corr = exp(m - newMax);
                    l *= corr;
                    for (int d = 0; d < hd; d++) acc[d] *= corr;
                    float blockSum = 0.0f;
                    for (int kk = 0; kk < chunk; kk++) {
                        int kIdx = k0 + kk;
                        if (kIdx >= s0 && kIdx < s1) {
                            float e = exp(s[kk] - newMax);
                            blockSum += e;
                            for (int d = 0; d < hd; d++) acc[d] += e * Vs[kk][d];
                        }
                    }
                    l += blockSum;
                    m = newMax;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        device float* oi = out + (uint)i * (uint)hidden + off;
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int d = 0; d < hd; d++) oi[d] = acc[d] * inv;
    }
}

// silu_mul: gate = silu(gate) * up, the gated-MLP activation. silu in f32 (vit.cu's
// double does not port), which is why the per-kernel bar is looser than CUDA's 1e-6.
kernel void silu_mul(
    device float* gate [[buffer(0)]], device const float* up [[buffer(1)]], constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = gate[gid];
    gate[gid] = (vv / (1.0f + exp(-vv))) * up[gid];
}

// erf_approx: Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7 — at the f32 floor.
// MSL's standard library has NO erf (unlike CUDA), so the EXACT GELU below supplies its
// own. This is what makes gelu_erf a genuine f32 approximation of the float64 reference,
// hence its looser (but stated) per-kernel bar; the discrimination from gelu_tanh
// (~5e-4) is unaffected and remains the proof of correct FORM.
inline float erf_approx(float x) {
    float s = sign(x);
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// gelu_erf: the EXACT (erf) GELU — nn.GELU()'s default, used by Qwen's patch merger.
// Distinct from gelu_tanh; see the header note. The divide matches vit.cu (exact under
// fast-math-off); erf is erf_approx above because MSL lacks a stdlib erf.
kernel void gelu_erf(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = x[gid];
    x[gid] = 0.5f * vv * (1.0f + erf_approx(vv / 1.4142135623730951f));
}

// ---------------------------------------------------------------------------
// TILED GEMMs (throughput) — the Metal mirror of vit.cu's tiled section. The untiled
// gemm_w8a8 / gemm_f32 above are correctness-first: one thread per output, each re-reading
// a whole A row and B row from global memory. These stage a TILE×TILE block of A and of B
// through threadgroup memory per K-chunk, so each element is read once per tile.
//
// Launched via Run2D (dispatchThreadgroups — UNIFORM whole threadgroups), so the edge
// tiles are full and every thread reaches the barrier; each thread then bounds-checks its
// own (m,n,k), exactly like the CUDA kernels. This is the ONE Metal kernel family that
// keeps the CUDA bounds checks, because its dispatch is uniform, not dispatchThreads.
//
// gemm_w8a8_tiled is expected BIT-IDENTICAL to the untiled kernel: the accumulator is
// int32 and integer addition is associative, so re-chunking K cannot change the sum (no
// overflow at ViT shapes: K·127² ≈ 1.9e7 ≪ 2³¹). gemm_f32_tiled cannot claim that (f32
// addition reassociates), so it is gated on a tight relative bound vs a float64 reference.
#define TILE 16

kernel void gemm_w8a8_tiled(
    device const char* A [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const char* B [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup char As[TILE][TILE];
    threadgroup char Bs[TILE][TILE];
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;
    int n = (int)tgpos.x * TILE + tx;
    int bRow = (int)tgpos.x * TILE + ty; // the B row this thread STAGES (not the one it uses)
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(uint)m * (uint)K + (uint)k] : (char)0;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(uint)bRow * (uint)K + (uint)k] : (char)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += (int)As[ty][kk] * (int)Bs[tx][kk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < M && n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc * aScale[m] * bScale[n];
}

// gemm_w8a8_reg — the register-blocked, packed-word-staged twin of gemm_w8a8_tiled
// (M-15, audit-metal-2026-09-12.md), porting CUDA's already-shipped gemm_w8a8_reg
// (vit.cu, audit M-12). gemm_w8a8_tiled computes ONE output per thread with
// byte-granular threadgroup staging — 32 threadgroup loads per 4 MACs, LSU-bound
// rather than MAC-bound, the exact shape CUDA's own M-12 measured at 7% of its dp4a
// roof and retired from the ANN path; the retirement never reached the ViT path.
//
// Two changes, both mirroring gemm_w8a8_reg's CUDA playbook:
//
//   PACKED-WORD STAGING. Threadgroup memory holds int (four packed int8 lanes), not
//   char, so one load feeds a whole 4-lane MAC instead of four char loads feeding a
//   quarter of one. A/B are read through an int* view — K/4 words per row — which is
//   why K%4 is required; Metal buffer allocations are word-aligned.
//
//   4×4 REGISTER BLOCKING. Each thread owns a 4×4 micro-tile, so the 8 shared words it
//   loads per k-word-step feed 16 4-lane MACs instead of 1 — the same ratio win CUDA's
//   dp4a version gets. MSL has no simdgroup_matrix int8 form and no hardware dp4a
//   intrinsic (unlike gemm_f32_sg/sg_big above, which use simdgroup_matrix — f32 only),
//   so dp4a_manual below unpacks each packed word via arithmetic right-shift
//   (sign-extending each byte) and sums the four lane products in scalar registers —
//   this still cuts threadgroup-memory TRAFFIC 4× per word loaded and amortizes each
//   load over 4 uses in each dimension via the register tile, even without a hardware
//   4-way MAC instruction. The shift-based unpack matches how the bytes were packed:
//   both CPU and Apple GPU are little-endian, so element 0 of the original int8 array
//   is the LOW byte of the reinterpreted word, same as CUDA's convention.
//
// K steps 16 elements (IBK=4 words), not 64: SigLIP-so400m's intermediate width (4304)
// is a multiple of 16 but not of 64, so a K%64 kernel would miss the MLP projections
// that are most of a ViT's work — the same reasoning gemm_w8a8_reg's own comment gives.
//
// The epilogue is byte-for-byte gemm_w8a8_tiled's: an exact int32 accumulator scaled by
// aScale[m]*bScale[n] in the same order. int32 addition is associative (no overflow at
// ViT shapes: K·127² ≈ 1.9e7 ≪ 2³¹), so re-chunking K into words/steps cannot change the
// sum — this kernel and gemm_w8a8_tiled agree bit-for-bit regardless of grouping, and
// GEMMW8A8Plan can route between them freely.
//
// Requires only K%16==0 (IBK words of four int8). M and N are free: edge tiles stage
// zeros and skip their stores, so the inner loop stays unguarded for any M, N — matters
// because an M%64/N%64 requirement would have excluded SigLIP-so400m's MLP projections
// (N=4304, not a multiple of 64).
#define IBM 64
#define IBN 64
#define IBK 4
#define IBTM 4
#define IBTN 4

inline int dp4a_manual(int a, int b, int acc) {
    int a0 = a << 24 >> 24, a1 = a << 16 >> 24, a2 = a << 8 >> 24, a3 = a >> 24;
    int b0 = b << 24 >> 24, b1 = b << 16 >> 24, b2 = b << 8 >> 24, b3 = b >> 24;
    return acc + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
}

kernel void gemm_w8a8_reg(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x; // 0..255
    int lr = tid >> 2, lc = tid & 3;          // 64 rows x 4 k-words per pass

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        // Edge tiles are handled by STAGING ZEROS, not by bounds-checking the inner
        // loop — see the kernel comment above for why.
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc[i][j] * aScale[m] * bScale[n];
        }
    }
}

// gemm_w8a8_bias — fused GEMM + bias addition.
// Writes C[m, n] = (acc * aScale * bScale) + bias[n], eliminating separate add_bias dispatch.
kernel void gemm_w8a8_bias(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* C [[buffer(5)]],
    constant int& M [[buffer(6)]], constant int& N [[buffer(7)]], constant int& K [[buffer(8)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
        }
    }
}

// gemm_w8a8_bias_add — fused GEMM + bias + residual addition.
// Writes residual[m, n] += (acc * aScale * bScale) + bias[n], eliminating both add_bias and add_vec.
kernel void gemm_w8a8_bias_add(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* residual [[buffer(5)]],
    constant int& M [[buffer(6)]], constant int& N [[buffer(7)]], constant int& K [[buffer(8)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) {
                uint idx = (uint)m * (uint)N + (uint)n;
                residual[idx] += (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
            }
        }
    }
}

kernel void gemm_f32_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;
    int n = (int)tgpos.x * TILE + tx;
    int bRow = (int)tgpos.x * TILE + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(uint)m * (uint)K + (uint)k] : 0.0f;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(uint)bRow * (uint)K + (uint)k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < M && n < N) C[(uint)m * (uint)N + (uint)n] = acc;
}

// gemm_f32_sg — the PRODUCTION f32 GEMM, on simdgroup_matrix (the Apple GPU's cooperative
// 8×8 matrix ALU), vs gemm_f32_tiled's scalar accumulation. Same C[m,n]=dot(A[m,:],B[n,:])
// with B row-major [N,K].
//
// Layout: each threadgroup computes a BM×BN = 32×32 output tile with 4 simdgroups (2×2),
// each owning a 16×16 sub-tile = 2×2 accumulator fragments. Per K-chunk (BK=8) the 128
// threads cooperatively stage A[32×8] and Bᵀ[8×32] into threadgroup memory, ZERO-PADDED on
// the M/N/K edges, so simdgroup_load never reads out of bounds and ANY shape is handled;
// each simdgroup then loads its A/B fragments from that staged tile and issues four
// simdgroup_multiply_accumulate. Results stage back through Cs and are written bounds-checked.
//
// B is loaded pre-transposed into Bs[k][n] (a plain scatter during staging, not a device
// transpose), so the multiply consumes an [m,k]·[k,n] pair directly. Launch via Run2D with
// grid = (ceil(N/32), ceil(M/32)) threadgroups of 128 threads (SGDims).
#define SGBM 32
#define SGBN 32
#define SGBK 8

kernel void gemm_f32_sg(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;      // 0..3
    int sgr = sg / 2, sgc = sg % 2; // 2×2 simdgroup grid → each covers 16×16

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) C[(uint)(m0 + r) * (uint)N + (uint)(n0 + c)] = Cs[r][c];
    }
}

// gemm_f32_bias — production f32 GEMM with fused bias addition on simdgroup_matrix.
// Writes C[m,n] = Cs[r,c] + bias[n0+c].
kernel void gemm_f32_bias(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* C [[buffer(3)]],
    constant int& M [[buffer(4)]], constant int& N [[buffer(5)]], constant int& K [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;
    int sgr = sg / 2, sgc = sg % 2;

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) {
            uint pos = (uint)(m0 + r) * (uint)N + (uint)(n0 + c);
            C[pos] = Cs[r][c] + bias[n0 + c];
        }
    }
}

// gemm_f32_bias_add — production f32 GEMM with fused bias and residual addition on simdgroup_matrix.
// Writes residual[m,n] += Cs[r,c] + bias[n0+c].
kernel void gemm_f32_bias_add(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* residual [[buffer(3)]],
    constant int& M [[buffer(4)]], constant int& N [[buffer(5)]], constant int& K [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;
    int sgr = sg / 2, sgc = sg % 2;

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) {
            uint pos = (uint)(m0 + r) * (uint)N + (uint)(n0 + c);
            residual[pos] += Cs[r][c] + bias[n0 + c];
        }
    }
}

// gemm_f32_sg_big — the ALIGNED fast path of gemm_f32_sg, same 32×32 tile / 4 simdgroups /
// 2×2 fragments (so the same low register footprint the general kernel already tunes well),
// but for shapes with M%32==0, N%32==0, K%8==0 (GEMMF32Plan enforces it, else falls back to
// gemm_f32_sg). The alignment buys the removal of the general kernel's per-K-step cost:
//   - A and B fragments load DIRECTLY from device (B with the transpose flag, since it is
//     stored [N,K]) — NO threadgroup staging and, crucially, NO two barriers per K-step;
//   - NO edge bounds checks and NO output staging through Cs — every access is in range.
// Wider register blocking was tried (64×64, 16 fragments/simdgroup) and lost badly to register
// spilling and low occupancy, so the win here is dropping the barriers, not enlarging the tile.
kernel void gemm_f32_sg_big(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]])
{
    int sgr = (int)sgid / 2, sgc = (int)sgid % 2; // 2×2 simdgroups, each 16×16
    int rm = (int)tgpos.y * 32 + sgr * 16;
    int cn = (int)tgpos.x * 32 + sgc * 16;

    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 8) {
        simdgroup_float8x8 a[2], b[2];
        for (int i = 0; i < 2; i++)
            simdgroup_load(a[i], A + (uint)(rm + i * 8) * (uint)K + (uint)k0, K);
        for (int j = 0; j < 2; j++)
            simdgroup_load(b[j], B + (uint)(cn + j * 8) * (uint)K + (uint)k0, K, ulong2(0, 0), true);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            simdgroup_store(acc[i][j], C + (uint)(rm + i * 8) * (uint)N + (uint)(cn + j * 8), N);
}`

// Kernel entry points in vitMSL (identical names to the CUDA side).
const (
	KernelQuantRows = "quant_rows"
	KernelGEMMW8A8  = "gemm_w8a8"
	KernelGEMMF32   = "gemm_f32"
	KernelAddBias   = "add_bias"
	KernelAddVec    = "add_vec"
	KernelLayerNorm = "layernorm"
	KernelGELUTanh  = "gelu_tanh"
	KernelAttention = "attention"

	// KernelAttentionTiled is the query-tiled online-softmax kernel (M-15). Dispatch via
	// AttentionTiledDispatch, and only when AttentionTiledEligible(hd) — see the kernel's
	// own doc comment.
	KernelAttentionTiled = "attention_tiled"

	// --- Qwen2.5-VL additions (identical names to the CUDA side) ---
	KernelRMSNorm           = "rmsnorm"
	KernelRopeQK            = "rope_qk"
	KernelAttentionSeg      = "attention_seg"
	KernelAttentionSegTiled = "attention_seg_tiled"
	KernelSiLUMul           = "silu_mul"
	KernelGELUErf           = "gelu_erf"

	// --- tiled GEMMs (throughput; identical names to the CUDA side) ---
	KernelGEMMW8A8Tiled = "gemm_w8a8_tiled"
	KernelGEMMF32Tiled  = "gemm_f32_tiled"

	// KernelGEMMW8A8Reg is the register-blocked, packed-word int8 GEMM (identical name
	// to the CUDA side's gemm_w8a8_reg). Pick it via GEMMW8A8Plan.
	KernelGEMMW8A8Reg = "gemm_w8a8_reg"

	// KernelGEMMF32SG is the production f32 GEMM on simdgroup_matrix (Metal-only; CUDA has
	// no analogue in this kernel set). Same signature as gemm_f32_tiled. Launch with SGDims.
	KernelGEMMF32SG = "gemm_f32_sg"

	// KernelGEMMF32SGBig is the aligned (M%32==0, N%32==0, K%8==0) fast path: 32×32 tile
	// (SGBigBlock), direct device loads, no staging/barriers/bounds. Pick it via GEMMF32Plan.
	KernelGEMMF32SGBig = "gemm_f32_sg_big"

	// KernelGEMMW8A8Bias is gemm_w8a8 with fused bias epilogue: C[m,n] = (Aq·Bq)*aScale*bScale + bias[n].
	KernelGEMMW8A8Bias = "gemm_w8a8_bias"

	// KernelGEMMW8A8BiasAdd is gemm_w8a8 with fused bias and residual addition:
	// residual[m,n] += (Aq·Bq)*aScale*bScale + bias[n].
	KernelGEMMW8A8BiasAdd = "gemm_w8a8_bias_add"

	// KernelGEMMF32Bias is gemm_f32 with fused bias epilogue: C[m,n] = A·B + bias[n].
	KernelGEMMF32Bias = "gemm_f32_bias"

	// KernelGEMMF32BiasAdd is gemm_f32 with fused bias and residual addition:
	// residual[m,n] += A·B + bias[n].
	KernelGEMMF32BiasAdd = "gemm_f32_bias_add"
)

// SGBlock is gemm_f32_sg's threadgroup output-tile width (must match vitMSL's SGBM/SGBN);
// SGThreads is its threadgroup size (4 simdgroups × 32).
const (
	SGBlock   = 32
	SGThreads = 128
)

// SGDims is gemm_f32_sg's uniform-threadgroup 2-D geometry for Run2D: one SGBlock×SGBlock
// output tile per threadgroup, SGThreads threads each.
func SGDims(M, N int) (gx, gy, tgx, tgy int) {
	return (N + SGBlock - 1) / SGBlock, (M + SGBlock - 1) / SGBlock, SGThreads, 1
}

// SGBigBlock is gemm_f32_sg_big's threadgroup output-tile width (must match the kernel's 32).
const SGBigBlock = 32

// GEMMF32Plan picks the f32 GEMM kernel and its Run2D geometry for a shape M×N×K. The aligned
// no-barrier kernel (gemm_f32_sg_big — same 32×32 tile, but direct device loads and no bounds
// checks) is chosen when M and N are multiples of 32 and K of 8 — which every large encoder /
// ViT GEMM satisfies — and the general gemm_f32_sg (any shape) otherwise. Both bind at buffers
// 0..5 (A, B, C, M, N, K), so the caller just sets its scalar buffers and calls Run2D.
func (v ViT) GEMMF32Plan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M%SGBigBlock == 0 && N%SGBigBlock == 0 && K%8 == 0 {
		return v.GEMMF32SGBig, N / SGBigBlock, M / SGBigBlock, SGThreads, 1
	}
	gx, gy, tgx, tgy = SGDims(M, N)
	return v.GEMMF32SG, gx, gy, tgx, tgy
}

// GEMMTile is the tiled GEMMs' tile width; it must match vitMSL's TILE, which sizes the
// threadgroup As/Bs staging arrays. Mirrors cuda_vit.go's GEMMTile.
const GEMMTile = 16

// TileDims is the tiled GEMMs' uniform-threadgroup 2-D geometry for Run2D: one
// GEMMTile×GEMMTile output tile per threadgroup. (The CUDA side returns a LaunchConfig
// from TileGrid; Metal's Run2D takes the grid/threadgroup extents directly, so this
// returns them as four ints instead.)
func TileDims(M, N int) (gx, gy, tgx, tgy int) {
	return (N + GEMMTile - 1) / GEMMTile, (M + GEMMTile - 1) / GEMMTile, GEMMTile, GEMMTile
}

// IntRegBlock/IntRegK are gemm_w8a8_reg's per-threadgroup output-tile width and K-step
// in ELEMENTS (IBK words of four int8); together they form the kernel's alignment
// requirement. Named to match cuda_vit.go's identical constants.
//
// 16 rather than 64 is deliberate: SigLIP-so400m's intermediate width is 4304, a
// multiple of 16 but NOT of 64, so a K%64 kernel would have missed the MLP projections
// that are most of a ViT's work.
const (
	IntRegBlock = 64
	IntRegK     = 16
)

// GEMMW8A8Plan picks the int8 GEMM kernel and its Run2D geometry for a shape M×N×K — the
// int8 analogue of GEMMF32Plan, and the Metal twin of cuda_vit.go's GEMMW8A8Plan (M-15,
// audit-metal-2026-09-12.md; ported from CUDA's already-shipped M-12 kernel).
//
// It needs only K%16==0: M and N edge tiles stage zeros and skip their stores, so the
// inner loop stays unguarded while any M and N are accepted — SigLIP-so400m's
// intermediate width (4304) is a multiple of 16 but not 64, and an M%64/N%64
// requirement would have excluded the MLP projections, which are most of a ViT's work.
// A K%16!=0 shape falls back to gemm_w8a8_tiled. Both bind identically (A, aScale, B,
// bScale, C, M, N, K) and produce identical bits (int32 addition is associative), so a
// caller only swaps the pipeline and Run2D geometry.
func (v ViT) GEMMW8A8Plan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8Reg, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	gx, gy, tgx, tgy = TileDims(M, N)
	return v.GEMMW8A8Tiled, gx, gy, tgx, tgy
}

// GEMMW8A8BiasPlan picks the fused int8 GEMM+bias kernel and Run2D geometry for M×N×K.
// Returns an empty pipeline if K%IntRegK != 0.
// Bind signature: (Aw [[0]], aScale [[1]], Bw [[2]], bScale [[3]], bias [[4]], C [[5]], M [[6]], N [[7]], K [[8]]).
func (v ViT) GEMMW8A8BiasPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8Bias, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMW8A8BiasAddPlan picks the fused int8 GEMM+bias+residual add kernel and Run2D geometry for M×N×K.
// Returns an empty pipeline if K%IntRegK != 0.
// Bind signature: (Aw [[0]], aScale [[1]], Bw [[2]], bScale [[3]], bias [[4]], residual [[5]], M [[6]], N [[7]], K [[8]]).
func (v ViT) GEMMW8A8BiasAddPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8BiasAdd, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMF32BiasPlan picks the fused f32 GEMM+bias kernel and Run2D geometry for M×N×K.
// Bind signature: (A [[0]], B [[1]], bias [[2]], C [[3]], M [[4]], N [[5]], K [[6]]).
func (v ViT) GEMMF32BiasPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M > 0 && N > 0 && K > 0 {
		gx, gy, tgx, tgy = SGDims(M, N)
		return v.GEMMF32Bias, gx, gy, tgx, tgy
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMF32BiasAddPlan picks the fused f32 GEMM+bias+residual add kernel and Run2D geometry for M×N×K.
// Bind signature: (A [[0]], B [[1]], bias [[2]], residual [[3]], M [[4]], N [[5]], K [[6]]).
func (v ViT) GEMMF32BiasAddPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M > 0 && N > 0 && K > 0 {
		gx, gy, tgx, tgy = SGDims(M, N)
		return v.GEMMF32BiasAdd, gx, gy, tgx, tgy
	}
	return Pipeline{}, 0, 0, 0, 0
}

// AttnTiledMaxHD/AttnTiledKTile/AttnTiledQTile mirror attention_tiled's AT_MAXHD/AT_KTILE/
// AT_QTILE (M-15) — kept in sync by hand, like GEMMTile/SGBlock above mirror their own
// kernels' #defines elsewhere in this file.
const (
	AttnTiledMaxHD = 128
	AttnTiledKTile = 16
	AttnTiledQTile = 32
)

// AttentionTiledEligible reports whether attention_tiled can serve a tower with this
// head dim: its K/V staging arrays are sized statically for hd <= AttnTiledMaxHD. A wider
// hd MUST use the untiled `attention` kernel instead — this is not a fallback the kernel
// itself performs, callers must check it themselves before dispatching.
func AttentionTiledEligible(hd int) bool { return hd > 0 && hd <= AttnTiledMaxHD }

// AttentionTiledMinNP mirrors cuda_vit.go's constant of the same name (that file is
// //go:build linux, so darwin needs its own copy) — it is CUDA's measured crossover, kept
// here only so the two files' exported surfaces match. Do NOT gate a Metal dispatch on it;
// see AttentionTiledEnabledOnMetal below.
const AttentionTiledMinNP = 3072

// AttentionTiledEnabledOnMetal is false: the CUDA crossover above does not transfer.
// BenchmarkMetalAttention on a real M1 Pro found untiled `attention` faster than
// `attention_tiled` at EVERY shape tested (np=729..4096, the full realistic ViT/Qwen
// range) — 1.46-1.85x, not close, and not narrowing at the high end the way a real
// crossover approaching from one side would. CUDA's dynamic-shared-memory occupancy
// penalty (attention_tiled's own doc comment above) that motivates the tiled kernel
// there does not describe Apple's unified-memory/threadgroup-memory architecture, so
// there is no reason to expect np=3072 — or any nearby number — to be Metal's crossover
// too. Callers MUST check this (not just AttentionTiledEligible) before dispatching
// attention_tiled. The kernel itself is correct and stays tested
// (TestMetal_vitAttentionTiled) for whenever a real Metal shape justifies flipping this.
const AttentionTiledEnabledOnMetal = false

// AttentionTiledDispatch returns the Run1D geometry (n, tg) for attention_tiled at a
// given (np, nH). Callers MUST check AttentionTiledEligible(hd) first — an ineligible hd
// dispatched anyway is an out-of-bounds write into the kernel's fixed-size Ks/Vs arrays,
// not a graceful decline.
func AttentionTiledDispatch(np, nH int) (n, tg int) {
	tilesPerHead := (np + AttnTiledQTile - 1) / AttnTiledQTile
	return nH * tilesPerHead * AttnTiledQTile, AttnTiledQTile
}

// AttentionSegTiledDispatch returns the Run1D geometry (n, tg) for attention_seg_tiled at a
// given (seq, nH). Callers MUST check AttentionTiledEligible(hd) first.
func AttentionSegTiledDispatch(seq, nH int) (n, tg int) {
	tilesPerHead := (seq + AttnTiledQTile - 1) / AttnTiledQTile
	return nH * tilesPerHead * AttnTiledQTile, AttnTiledQTile
}

// ViTBlock is the threadgroup width the per-row/attention kernels reduce at; it must
// match vitMSL's LNBLOCK (the static threadgroup reduction arrays are sized to it).
//
// It is PART OF THE BIT-IDENTITY CONTRACT, not just a performance knob. The layernorm
// and softmax kernels sum floats across `tgsz` (== this width) threads, and f32 addition
// is not associative, so the summation order — and therefore the exact bits — is fixed
// by this value. Precise-math compilation (CompileLibraryPrecise) removes the compiler's
// discretion over contraction/reassociation but NOT this: the width is chosen here, in
// host code. Do not sweep it for a speed win without re-baselining the ViT parity gate —
// a small consistent shift passes the tolerance-based maxAbsDiff check while the numbers
// have moved. Change ViTBlock and LNBLOCK together (see the cross-thread `+=` reductions
// in vitMSL, each tagged as bit-identity-coupled).
const ViTBlock = 256

// ViT holds the compiled encoder kernel pipelines — identical shape to the CUDA ViT.
type ViT struct {
	QuantRows Pipeline
	GEMMW8A8  Pipeline
	GEMMF32   Pipeline
	AddBias   Pipeline
	AddVec    Pipeline
	LayerNorm Pipeline
	GELUTanh  Pipeline
	Attention Pipeline

	// AttentionTiled is the query-tiled online-softmax kernel (M-15) — see its own doc
	// comment in vitMSL.
	AttentionTiled Pipeline

	// Qwen2.5-VL additions.
	RMSNorm           Pipeline
	RopeQK            Pipeline
	AttentionSeg      Pipeline
	AttentionSegTiled Pipeline
	SiLUMul           Pipeline
	GELUErf           Pipeline

	// Tiled GEMMs — same math, staged through threadgroup memory.
	GEMMW8A8Tiled Pipeline
	GEMMF32Tiled  Pipeline

	// GEMMW8A8Reg is the register-blocked, packed-word-staged int8 GEMM. Pick it via
	// GEMMW8A8Plan, never directly — it requires K%16==0 and has no bounds-checked
	// fallback path of its own.
	GEMMW8A8Reg Pipeline

	// Production f32 GEMM on simdgroup_matrix (Metal-only); SGBig is the aligned fast path.
	GEMMF32SG    Pipeline
	GEMMF32SGBig Pipeline

	// Fused GEMM epilogues (bias and residual addition)
	GEMMW8A8Bias    Pipeline
	GEMMW8A8BiasAdd Pipeline
	GEMMF32Bias     Pipeline
	GEMMF32BiasAdd  Pipeline
}

// NewViT compiles vitMSL on this device and builds every encoder pipeline. The library
// is tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewViT() (ViT, error) {
	lib, err := d.CompileLibraryPrecise(vitMSL, MSL3_1)
	if err != nil {
		return ViT{}, fmt.Errorf("metal: compile ViT library: %w", err)
	}
	var v ViT
	for _, bind := range []struct {
		name string
		dst  *Pipeline
	}{
		{KernelQuantRows, &v.QuantRows},
		{KernelGEMMW8A8, &v.GEMMW8A8},
		{KernelGEMMF32, &v.GEMMF32},
		{KernelAddBias, &v.AddBias},
		{KernelAddVec, &v.AddVec},
		{KernelLayerNorm, &v.LayerNorm},
		{KernelGELUTanh, &v.GELUTanh},
		{KernelAttention, &v.Attention},
		{KernelAttentionTiled, &v.AttentionTiled},
		{KernelRMSNorm, &v.RMSNorm},
		{KernelRopeQK, &v.RopeQK},
		{KernelAttentionSeg, &v.AttentionSeg},
		{KernelAttentionSegTiled, &v.AttentionSegTiled},
		{KernelSiLUMul, &v.SiLUMul},
		{KernelGELUErf, &v.GELUErf},
		{KernelGEMMW8A8Tiled, &v.GEMMW8A8Tiled},
		{KernelGEMMF32Tiled, &v.GEMMF32Tiled},
		{KernelGEMMF32SG, &v.GEMMF32SG},
		{KernelGEMMF32SGBig, &v.GEMMF32SGBig},
		{KernelGEMMW8A8Reg, &v.GEMMW8A8Reg},
		{KernelGEMMW8A8Bias, &v.GEMMW8A8Bias},
		{KernelGEMMW8A8BiasAdd, &v.GEMMW8A8BiasAdd},
		{KernelGEMMF32Bias, &v.GEMMF32Bias},
		{KernelGEMMF32BiasAdd, &v.GEMMF32BiasAdd},
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}
