package kernels

// MLAKernels is DeepSeek and Kimi's multi-head latent attention on the resident decode path, ported from gpu/mla.go's absorbed
// form. The cache holds one latent row per position, [cn (rank) | krj (qkRope)]: the KV down-projection's latent, RMS-normed,
// then its rope key, rotated at store time. Per-head K and V are never built: the query is absorbed through W_UK into rank
// space, scored against each latent row, and the rank-space weighted sum is lifted through W_UV.
//
//   - mla_store: one threadgroup of 64 writes position pos's latent row at pos*latDim.
//   - mla_head_mv: a per-head matvec, one simdgroup per output element: W_UK's absorb (into qAbs, rank values a head at
//     stride latDim) and W_UV's lift.
//   - mla_qrope: q's rope slice of each head, rotated into qAbs after the head's rank values.
//   - mla_attn: one threadgroup of 128 per head, an online softmax over the nKeys latent rows, scores over all latDim values
//     and the weighted sum over the first rank.
//
// p[] packs the per-model shape so a dispatch binds one buffer for it: see each kernel's comment for its layout.
const MLAKernels = `
// mla_store p: [rank, qkRope, interleave]; pos is the cache position, ropePos the rotation's.
kernel void mla_store(device const float* kvDown [[buffer(0)]], device const float* normW [[buffer(1)]],
    device const float* invFreq [[buffer(2)]], device float* lat [[buffer(3)]], constant uint* p [[buffer(4)]],
    constant uint& pos [[buffer(5)]], constant uint& ropePos [[buffer(6)]], constant float& eps [[buffer(7)]],
    constant float& ropeScale [[buffer(8)]], uint t [[thread_position_in_threadgroup]]) {
    uint rank = p[0], qkRope = p[1], interleave = p[2];
    uint base = pos * (rank + qkRope);
    threadgroup float sh[64];
    float ss = 0.0f;
    for (uint i = t; i < rank; i += 64u) { float v = kvDown[i]; ss += v * v; }
    sh[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32u; s > 0u; s >>= 1u) {
        if (t < s) sh[t] += sh[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(sh[0] / float(rank) + eps);
    for (uint i = t; i < rank; i += 64u) lat[base + i] = kvDown[i] * inv * normW[i];
    uint hr = qkRope / 2u;
    for (uint i = t; i < hr; i += 64u) {
        float theta = float(ropePos) * invFreq[i];
        float c = cos(theta) * ropeScale, s = sin(theta) * ropeScale;
        float a, b;
        if (interleave == 1u) { a = kvDown[rank + 2u*i]; b = kvDown[rank + 2u*i + 1u]; }
        else { a = kvDown[rank + i]; b = kvDown[rank + hr + i]; }
        lat[base + rank + i] = a * c - b * s;
        lat[base + rank + hr + i] = b * c + a * s;
    }
}

// mla_head_mv p: [nH, N, K, aStride, outStride]: dst[h*outStride + n] = sum_k a[h*aStride + k] * w[(h*N + n)*K + k].
kernel void mla_head_mv(device const float* a [[buffer(0)]], device const float* w [[buffer(1)]], device float* dst [[buffer(2)]],
    constant uint* p [[buffer(3)]], uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nH = p[0], N = p[1], K = p[2];
    uint elem = tgid * 8u + sgid;
    if (elem >= nH * N) return;
    uint h = elem / N, n = elem % N;
    device const float* ar = a + h * p[3];
    device const float* wr = w + (h * N + n) * K;
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) acc += ar[k] * wr[k];
    acc = simd_sum(acc);
    if (lane == 0u) dst[h * p[4] + n] = acc;
}

// mla_qrope p: [nH, qkHead, qkNope, qkRope, rank, latDim, interleave].
kernel void mla_qrope(device const float* q [[buffer(0)]], device const float* invFreq [[buffer(1)]], device float* qAbs [[buffer(2)]],
    constant uint* p [[buffer(3)]], constant uint& ropePos [[buffer(4)]], constant float& ropeScale [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    uint nH = p[0], qkHead = p[1], qkNope = p[2], hr = p[3] / 2u, rank = p[4], latDim = p[5];
    if (idx >= nH * hr) return;
    uint h = idx / hr, i = idx % hr;
    uint qbase = h * qkHead + qkNope;
    float theta = float(ropePos) * invFreq[i];
    float c = cos(theta) * ropeScale, s = sin(theta) * ropeScale;
    float a, b;
    if (p[6] == 1u) { a = q[qbase + 2u*i]; b = q[qbase + 2u*i + 1u]; }
    else { a = q[qbase + i]; b = q[qbase + hr + i]; }
    uint obase = h * latDim + rank;
    qAbs[obase + i] = a * c - b * s;
    qAbs[obase + hr + i] = b * c + a * s;
}

// mla_attn p: [nH, latDim, rank]. One threadgroup of 128 per head; rank <= 1024 (8 values a lane).
kernel void mla_attn(device const float* qAbs [[buffer(0)]], device const float* lat [[buffer(1)]], device float* wsum [[buffer(2)]],
    constant uint* p [[buffer(3)]], constant uint& nKeys [[buffer(4)]], constant float& scale [[buffer(5)]],
    uint h [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint nH = p[0], latDim = p[1], rank = p[2];
    if (h >= nH) return;
    threadgroup float red[4];
    device const float* q = qAbs + h * latDim;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float m = -1e30f, l = 0.0f;
    for (uint j = 0u; j < nKeys; j++) {
        device const float* k = lat + j * latDim;
        float part = 0.0f;
        for (uint d = t; d < latDim; d += 128u) part += q[d] * k[d];
        part = simd_sum(part);
        if (lane == 0u) red[sgid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x = (red[0] + red[1] + red[2] + red[3]) * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mnew = max(m, x);
        float corr = exp(m - mnew), pe = exp(x - mnew);
        uint i = 0u;
        for (uint d = t; d < rank; d += 128u) { acc[i] = acc[i] * corr + pe * k[d]; i++; }
        l = l * corr + pe;
        m = mnew;
    }
    uint i = 0u;
    for (uint d = t; d < rank; d += 128u) { wsum[h * rank + d] = acc[i] / l; i++; }
}
`
