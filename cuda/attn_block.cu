// attn_block_full — the DFlash block drafter's NON-CAUSAL attention.
//
// A verbatim copy of prefill_batched.cu's attn_batched with ONE line changed: nKeys is
// startPos + M for every row, instead of startPos + m + 1. That is the whole difference between
// the target's causal verify and the drafter's bidirectional block:
//
//   decoder/dflash.go, blockTrunk.layer: "Non-causal: every block query attends every context
//   key AND every block key, including positions after itself. No mask, by construction."
//
// With the context occupying cache positions [0, startPos) and the block's own K/V written at
// [startPos, startPos+M), nKeys = startPos+M gives every block row exactly ctx ‖ block.
//
// WHY ITS OWN .cu / .ptx RATHER THAN A KERNEL ADDED TO prefill_batched.cu. Regenerating
// prefill_batched.ptx to add a kernel risks shifting codegen (register allocation, ordering) for
// the kernels already in it, and every batched-prefill parity gate rests on those. This repo
// already established the isolation pattern for exactly this reason — argmax_reduce was split
// out of glue.ptx, and router_f32 out of moe.ptx. Same reasoning, so no shipped PTX changes.
//
// Everything else is byte-identical to attn_batched, deliberately: the float4 K read with
// separate adds (attention()'s exact d-order), the two-pass softmax, the shared-memory layout.
// The window parameter is kept so the signature matches; the drafter passes 0.
//
// Regenerate with:  ./build_ptx.sh attn_block

extern "C" {

__global__ void attn_block_full(const float* __restrict__ q, const float* __restrict__ kc,
                             const float* __restrict__ vc, int nH, int nKV, int hd, int startPos,
                             float scale, int window, int M, float* __restrict__ ctx) {
    int h = blockIdx.x; if (h >= nH) return;
    int m = blockIdx.y; if (m >= M) return;
    int nKeys = startPos + M; // NON-CAUSAL: uniform for every row (the block is bidirectional)
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
    float ls = 0.f;
    for (int s = winStart + t; s < nKeys; s += nt) { float e = __expf(sc[s - winStart] - mx); sc[s - winStart] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = 1.f / red[0]; __syncthreads();
    for (int d = t; d < hd; d += nt) {
        float acc = 0.f;
        for (int s = winStart; s < nKeys; s++) acc = __fmaf_rn(sc[s - winStart], vc[(long)s * kvDim + kvh * hd + d], acc);
        ctx[(long)m * qDim + h * hd + d] = acc * inv;
    }
}

} // extern "C"
