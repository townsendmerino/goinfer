// rope_mrope_prefill.cu — Qwen2.5-VL's m-RoPE batched-prefill rotation
// (decoder.ResidentMRoPEPrefill / cuda.PrefillMRoPELast).
//
// A verbatim copy of prefill_batched.cu's rope_kv_batched with the rotation angle widened from a
// single row-sequential scalar to a PER-ROW, PER-FREQUENCY-SECTION lookup — decoder/rope.go's
// applyMRoPE, inlined: each frequency index d rotates by pos[comp(d)]·invFreq[d], where comp(d)
// picks temporal/height/width per the model's MRopeSection cumulative boundaries (sec0, sec1).
//
//   decoder/rope.go, applyMRoPE: "the len(invFreq) frequencies are partitioned into section[]
//   chunks assigned to the (temporal, height, width) position components, so each frequency d
//   rotates by pos[comp(d)]·invFreq[d]. ... For a TEXT token the three positions are equal, so
//   this reduces EXACTLY to applyRoPE(pos[0])."
//
// WHY EVERY ROW NEEDS THE PER-ROW LOOKUP, NOT JUST IMAGE-BLOCK ROWS. decoder/rope.go's
// mropePositions resumes scalar counting AFTER an image block from base+max(t,h,w) — a value
// COMPRESSED by the merged image grid, not the naive sequential count startPos+m. So even an
// ordinary TEXT row past an image block needs its own per-row position, not the row-index
// formula rope_kv_batched already uses. Only a prompt with NO image at all would ever have every
// row's mrope position equal startPos+m — and Qwen-VL prefill without an image never reaches this
// kernel (PrefillLast/rope_kv_batched already serve that case unmodified).
//
// pos vs (posT,posH,posW): the same split rope_kv's own header documents for decode (pos vs
// ropePos) — pos still drives ONLY the KV-cache STORE index (row-sequential, startPos+m,
// UNCHANGED); the three position arrays drive ONLY the rotation angle. Getting this backwards is
// silent: wrong rotation looks like a plausible but wrong token, wrong storage position corrupts
// an already-written KV row.
//
// Attention itself needs no change for Qwen — its image tokens attend CAUSALLY
// (decoder/generate_vl.go's own doc comment: "Qwen's bidirectionality is in the ViT, not the
// decoder"), so attn_batched serves it unmodified; only this kernel is new.
//
// WHY ITS OWN .cu / .ptx RATHER THAN A KERNEL ADDED TO prefill_batched.cu. Same reasoning
// attn_img_prefill.cu's header states: regenerating prefill_batched.ptx in place risks shifting
// codegen for kernels every batched-prefill parity gate rests on. No shipped PTX changes there.
//
// MODE (added 2026-10-06 for the Qwen3.5 hybrids' image prefill, docs/queue-performance.md P26): 0 is Qwen2.5-VL's contiguous
// section layout exactly as before; 1 is Qwen3-VL's interleaved layout (see mrope_pos). GLM-OCR's pairwise twin
// (rope_pairwise.cu's rope_kv_mrope_batched_pw) is a separate kernel with its own argument list and is untouched.
//
// Regenerate with:  ./build_ptx.sh rope_mrope_prefill   (NVRTC 12.9.86, which reproduces the previous PTX byte for byte)

// mrope_pos picks the position component for frequency index d.
//   mode 0 (Qwen2.5-VL, contiguous blocks; decoder.mropeComponent): sec0 = section[0], sec1 = section[0]+section[1];
//          d < sec0 temporal, d < sec1 height, else width.
//   mode 1 (Qwen3-VL / Qwen3.5, per-index interleave; decoder.mropeComponentInterleaved): sec0 = 3*section[1],
//          sec1 = 3*section[2]; d%3==1 && d<sec0 height, d%3==2 && d<sec1 width, else temporal (which also covers
//          the tail beyond the strided runs). Mode 0 is the original expression, unchanged.
__device__ __forceinline__ int mrope_pos(int d, int sec0, int sec1, int mode, int pT, int pH, int pW)
{
    if (mode == 0) return (d < sec0) ? pT : (d < sec1 ? pH : pW);
    if (d % 3 == 1 && d < sec0) return pH;
    if (d % 3 == 2 && d < sec1) return pW;
    return pT;
}

extern "C" {

__global__ void rope_kv_mrope_batched(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ invFreq, float* __restrict__ kc, float* __restrict__ vc,
    int nH, int nKV, int hd, int startPos, int rhalf, int M, float mscale,
    const int* __restrict__ posT, const int* __restrict__ posH, const int* __restrict__ posW,
    int sec0, int sec1, int mode)
{
    int m = blockIdx.y; if (m >= M) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = startPos + m; // KV-CACHE STORE index — unchanged, row-sequential (see file header).
    int pT = posT[m], pH = posH[m], pW = posW[m]; // this row's OWN (t,h,w) rotation triple
    int tail = hd - 2 * rhalf;
    int qn = nH * rhalf, kn = nKV * rhalf, tn = nKV * tail;
    int qDim = nH * hd, kvDim = nKV * hd;
    float* qm = q + (long)m * qDim;
    float* km = k + (long)m * kvDim;
    const float* vm = v + (long)m * kvDim;
    if (idx < qn) {
        int h = idx / rhalf, d = idx % rhalf;
        int p = mrope_pos(d, sec0, sec1, mode, pT, pH, pW); // mropeComponent / mropeComponentInterleaved(d, section), inlined
        float ang = p * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = qm + h * hd;
        float a = base[d], b = base[d + rhalf];
        base[d] = __fmaf_rn(a, c, -__fmul_rn(b, s)); base[d + rhalf] = __fmaf_rn(a, s, __fmul_rn(b, c));
    } else if (idx < qn + kn) {
        int j = idx - qn; int h = j / rhalf, d = j % rhalf;
        int p = mrope_pos(d, sec0, sec1, mode, pT, pH, pW);
        float ang = p * invFreq[d]; float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
        float* base = km + h * hd;
        float a = base[d], b = base[d + rhalf];
        float r0 = __fmaf_rn(a, c, -__fmul_rn(b, s)), r1 = __fmaf_rn(a, s, __fmul_rn(b, c));
        base[d] = r0; base[d + rhalf] = r1;
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + d] = r0; kc[o + d + rhalf] = r1;
        vc[o + d] = vm[h * hd + d]; vc[o + d + rhalf] = vm[h * hd + d + rhalf];
    } else if (idx < qn + kn + tn) {
        int j = idx - qn - kn; int h = j / tail, t = 2 * rhalf + (j % tail);
        long o = (long)pos * kvDim + (long)h * hd;
        kc[o + t] = km[h * hd + t]; vc[o + t] = vm[h * hd + t];
    }
}

} // extern "C"
