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
// Callers gate on AttentionTiledEligible(hd) plus the AttentionTiledMinNP crossover (see cuda_vit.go);
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

