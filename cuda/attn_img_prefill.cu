// attn_img_prefill.cu — Gemma 3's image-block bidirectional attention for the resident batched
// prefill path (decoder.ResidentImagePrefill / cuda.PrefillImageLast).
//
// A verbatim copy of prefill_batched.cu's attn_batched with the mask formula widened for a
// contiguous [imgStart,imgEnd) sub-range: a query row whose OWN position lies inside that range
// sees the whole block (image tokens attend each other mutually, per decoder/kvcache.go's
// SetImageBlocks/attendHi); every other row stays exactly causal. This is NOT attn_block.cu's
// pattern (that kernel widens nKeys uniformly for every row — the drafter's whole block is
// non-causal). Here only in-block rows widen; text before and after the block stays causal.
//
//   decoder/kvcache.go, attendHi(pos): "pos for a causal (text) position, or end-1 if pos lies
//   in a bidirectional image block [start,end)."
//
// THE ONE FORMULA THAT IS *NOT* A NAIVE PORT OF attn_batched'S OWN nKeys->winStart COUPLING.
// attn_batched derives its sliding-window start FROM its (possibly widened) nKeys:
//   winStart = (window>0 && nKeys>window) ? nKeys-window : 0
// The CPU reference does NOT do this for an image-block row: decoder/forwardn.go computes the
// window's lower bound from `cache.WindowStart(pos, global)` — a function of pos ALONE — and the
// upper bound from `cache.attendHi(pos)` SEPARATELY; the two are never coupled. Reusing
// attn_batched's coupled formula here with the widened nKeys would silently UNDER-SIZE the
// shared-memory window for any windowed (local) layer once the image block starts more than one
// window-length into the sequence — the window would start too late, dropping keys the CPU
// reference attends to, and (worse) the host's shared-memory size calculation
// (cuda/prefill.go's checkPrefillShmemImg) must size the buffer using the SAME decoupled formula
// or this under-sizes an allocation into an out-of-bounds kernel write, not just a wrong answer.
// So: nKeysCausal (the plain causal count) feeds winStart; nKeys (the possibly-widened count)
// only extends how many keys of the softmax are actually visited/summed.
//
// WHY ITS OWN .cu / .ptx RATHER THAN A KERNEL ADDED TO prefill_batched.cu. Same reasoning
// attn_block.cu's own header states: regenerating prefill_batched.ptx in place risks shifting
// codegen for kernels every batched-prefill parity gate rests on. No shipped PTX changes there.
//
// Everything else is byte-identical to attn_batched, deliberately: the float4 K read with
// separate adds (attention()'s exact d-order), the two-pass softmax, the shared-memory layout,
// the sinks handling (kept for signature parity with attn_batched even though Gemma 3 — the only
// family this kernel serves — never sets it; NULL leaves the kernel bit-identical to the
// no-sink path, same guarantee attn_batched's own doc comment makes).
//
// Regenerate with:  ./build_ptx.sh attn_img_prefill

extern "C" {

__global__ void attn_img_batched(const float* __restrict__ q, const float* __restrict__ kc,
                             const float* __restrict__ vc, int nH, int nKV, int hd, int startPos,
                             float scale, int window, int M, float* __restrict__ ctx,
                             const float* __restrict__ sinks, int imgStart, int imgEnd) {
    int h = blockIdx.x; if (h >= nH) return;
    int m = blockIdx.y; if (m >= M) return;
    int pos = startPos + m;
    int nKeysCausal = pos + 1;
    // IMAGE-BLOCK WIDENING: only a row whose OWN position lies inside [imgStart,imgEnd) sees the
    // whole block; every other row (before or after it) stays exactly causal. imgEnd<=imgStart
    // is the "no block" sentinel (matches this codebase's window==0 "no window" convention).
    int nKeys = (imgEnd > imgStart && pos >= imgStart && pos < imgEnd) ? imgEnd : nKeysCausal;
    // DECOUPLED from nKeys on purpose — see the file header. winStart is derived from the plain
    // CAUSAL count, never from the widened one.
    int winStart = (window > 0 && nKeysCausal > window) ? nKeysCausal - window : 0;
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
        // float4 the K read — see attn_batched's identical comment for the rationale.
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
    if (hasSink) { sink = sinks[h]; mx = fmaxf(mx, sink); }
    float ls = 0.f;
    for (int s = winStart + t; s < nKeys; s += nt) { float e = __expf(sc[s - winStart] - mx); sc[s - winStart] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float denom = red[0]; if (hasSink) denom += __expf(sink - mx);
    float inv = 1.f / denom; __syncthreads();
    for (int d = t; d < hd; d += nt) {
        float acc = 0.f;
        for (int s = winStart; s < nKeys; s++) acc = __fmaf_rn(sc[s - winStart], vc[(long)s * kvDim + kvh * hd + d], acc);
        ctx[(long)m * qDim + h * hd + d] = acc * inv;
    }
}

} // extern "C"
