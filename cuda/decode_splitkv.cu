// decode_splitkv.cu — high-occupancy, BIT-IDENTICAL decode attention (Campaign A, split-KV).
//
// The single-block attn_batched(M=1) launches one block per query head (nH ≈ 12) → 12 blocks on a
// 40-SM card, occupancy-starved (A1-reprofile: Waves/SM 0.04, 11.9% occ, 93% no-eligible-warp). This
// splits the two O(nKeys·hd) passes across many blocks along their INDEPENDENT axes — scores over
// keys, V-sum over output dims — so every order-dependent softmax fold stays whole and in-order. The
// result is byte-for-byte the attn_batched(M=1) output (see docs/task-decode-splitkv-attention.md for
// the associativity argument). Decode-only (M=1). Own file — glue.ptx / moe.ptx untouched.
//
// Three launches per (head,token), with the implicit global barrier between kernel launches standing
// in for the cross-block sync CUDA can't give inside one grid:
//   1. splitkv_scores : grid nH × ⌈nWin/128⌉ — raw scores sc[s]=scale·Σ q·k (no reduction ⇒ trivially
//      bit-identical; the float4 K-read + separate adds are attn_batched's exact d-order).
//   2. splitkv_softmax: grid nH, block 128 (MUST match attn_batched) — max + exp + denom with the same
//      strided partition and reduction tree ⇒ byte-identical max/denominator. Writes sc←exp, inv=1/Σ.
//   3. splitkv_vsum   : grid nH × ⌈hd/Dtile⌉ — each thread owns one dim d and does the WHOLE sequential
//      fold Σ_s sc[s]·v[s][d] in ascending s (identical to attn_batched), then ·inv.

// splitkv_scores: one thread per key. scStore is [nH][nWin]; window-index i = s - winStart.
extern "C" __global__ void splitkv_scores(
    const float* __restrict__ q, const float* __restrict__ kc,
    int nH, int nKV, int hd, int winStart, int nKeys, float scale,
    float* __restrict__ scStore, int nWin) {
    int h = blockIdx.x; if (h >= nH) return;
    int i = blockIdx.y * blockDim.x + threadIdx.x;   // window index
    int s = winStart + i;
    if (i >= nWin || s >= nKeys) return;
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    const float* qh = q + (long)h * hd;
    const float* ks = kc + (long)s * kvDim + (long)kvh * hd;
    float dot = 0.f;
    // float4 K-read, separate adds — attn_batched's exact d-order (bit-identical; only load width).
    int d4 = hd >> 2;
    const float4* q4 = (const float4*)qh;
    const float4* k4 = (const float4*)ks;
    for (int j = 0; j < d4; j++) {
        float4 qq = q4[j], kk = k4[j];
        dot = __fmaf_rn(qq.x, kk.x, dot);
        dot = __fmaf_rn(qq.y, kk.y, dot);
        dot = __fmaf_rn(qq.z, kk.z, dot);
        dot = __fmaf_rn(qq.w, kk.w, dot);
    }
    for (int d = d4 << 2; d < hd; d++) dot = __fmaf_rn(qh[d], ks[d], dot);
    scStore[(long)h * nWin + i] = dot * scale;
}

// splitkv_softmax: grid nH, block 128 — replicates attn_batched's max+exp+denom EXACTLY (same strided
// partition thread t → {t, t+128, …}, same tree), so the max and denominator sum are byte-identical.
// In place: scStore[h][i] ← exp(sc-mx); invStore[h] = 1/Σ.
// `sinks` is gpt-oss's per-head learned attention sink: an extra logit with NO key and NO
// value. It joins the softmax MAX as well as the denominator —
//     m     = max(max_s score_s, sink_h)
//     denom = Sum_s exp(score_s - m) + exp(sink_h - m)
// — which is why it cannot be folded in afterwards as a denominator correction: shifting the
// max after the exp pass would need every exp recomputed. NULL for every other family, and
// the kernel is then bit-identical to before (the branch is uniform across the block, so it
// costs no divergence).
extern "C" __global__ void splitkv_softmax(
    float* __restrict__ scStore, int nH, int nWin, float* __restrict__ invStore,
    const float* __restrict__ sinks) {
    int h = blockIdx.x; if (h >= nH) return;
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float* sc = scStore + (long)h * nWin;
    float lm = -1e30f;
    for (int i = t; i < nWin; i += nt) lm = fmaxf(lm, sc[i]);
    red[t] = lm; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float sink = 0.f; bool hasSink = (sinks != nullptr);
    if (hasSink) { sink = sinks[h]; mx = fmaxf(mx, sink); }  // the sink competes for the max
    float ls = 0.f;
    for (int i = t; i < nWin; i += nt) { float e = __expf(sc[i] - mx); sc[i] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    // The sink's term is added ONCE, after the tree reduction, so it cannot be double-counted
    // across the strided partition — and it is added to the DENOMINATOR only: it has no value
    // vector, so splitkv_vsum's numerator is untouched.
    if (t == 0) { float denom = red[0]; if (hasSink) denom += __expf(sink - mx); invStore[h] = 1.f / denom; }
}

// splitkv_vsum: each thread owns one output dim d and folds Σ_s sc[s]·v[s][d] in ascending s — the
// WHOLE per-d reduction, identical to attn_batched. grid nH × ⌈hd/Dtile⌉ blocks fills the SMs.
//
// This is the bit-identity ceiling of the split (nH·hd threads, ncu: 3.8% occupancy — latency-bound).
// A manual ILP unroll (U=4, hoisting the independent v[s][d] loads) was tried and REFUTED: with clean
// advancing-pointer indexing it landed 158 vs the scalar 160 (no gain — nvcc already pipelines the
// scalar loop's independent loads), and a naive `(long)(s+k)*kvDim` unroll REGRESSED to 111 (extra
// 64-bit multiplies). Multiple accumulators are forbidden (they reorder the fold). So the scalar loop
// stands; further vsum gains would need a non-bit-identical reduction. See task-decode-splitkv-attention.
extern "C" __global__ void splitkv_vsum(
    const float* __restrict__ scStore, const float* __restrict__ vc,
    const float* __restrict__ invStore, int nH, int nKV, int hd, int winStart, int nKeys,
    int nWin, float* __restrict__ ctx) {
    int h = blockIdx.x; if (h >= nH) return;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= hd) return;
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    const float* sc = scStore + (long)h * nWin;
    float acc = 0.f;
    for (int s = winStart; s < nKeys; s++)
        acc = __fmaf_rn(sc[s - winStart], vc[(long)s * kvDim + (long)kvh * hd + d], acc);
    ctx[(long)h * hd + d] = acc * invStore[h];
}
