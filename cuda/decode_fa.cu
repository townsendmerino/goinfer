// decode_fa.cu — CUDA flash-decode lane (docs/tasks/red-october.md R6 step 2). NOT bit-identical to the
// shipped decode attention, opt-in (GOINFER_CUDA_FLASH_DECODE=S), and fidelity-gated by
// docs/measurements/attn-decode-fa-fidelity-PREREGISTERED.md.
//
// WHY. The exact split-KV path is three launches (scores, softmax, V-sum) that materialise the score row in
// global memory and keep every reduction whole and in order so the result is byte-identical to attn_batched.
// That costs ~447 us/layer on D7 at 8000 against a ~75 us traffic floor (4 kv heads x 8000 keys x 128 x 4 B x
// K and V = 32.8 MB at 448 GB/s); ncu says the V-sum is latency-bound at 9% occupancy and the scores kernel is
// issue-throttled by redundant q loads (docs/measurements/splitkv-stall-profile-2026-09-13.md). A flash-decode
// kernel drops the bit-identity constraint and with it the score row, the separate softmax, and the
// per-output-dim single-thread fold.
//
// SHAPE. One CTA (8 warps) per (kv head, key split); every query head of the GQA group is handled by the SAME
// CTA so each K/V row is read from DRAM once per group, not once per head. Keys are processed in blocks of 8:
//   scores  : 4-lane groups, one key per group (32 lanes = 8 keys). Lane `sub` reads float4 #(j*4+sub) of the
//             key row for j < HD/16, so the 4 lanes of a group read 64 contiguous bytes per step (full sectors),
//             dots it against the group's staged q, and the 4 lanes finish with 2 shuffles.
//   softmax : ONLINE, per head, warp-wide over the 8 keys of the block (3 shuffles for the max, 3 for the sum);
//             the running (m, l, acc) live in registers and are rescaled once per block, warp-uniformly.
//   P.V     : all 32 lanes on one key at a time, each lane owning HD/32 consecutive-by-float4 output dims, so the
//             V row is one coalesced read and there are no shuffles; the probabilities come from shared memory.
// Each WARP finishes with an unnormalised partial (O[HD], m, l) per head; a second kernel merges the
// nSplit*8 partials of a head in FIXED ASCENDING unit order (never atomics), so the result is deterministic
// run to run — the one property the lane must keep while giving up bit-identity to history.
//
// PARTITION. Split sp owns keys [winStart + sp*per, ...) with per = ceil(span/nSplit), a function of
// (nKeys, winStart, nSplit) only. Inside a split, warp w takes 8-key blocks w, w+8, w+16, ... so the 8 warps of
// a CTA sweep neighbouring memory together. Unit u = sp*8 + w indexes the partial buffer.
//
// SCOPE. hd in {64,128,256}, GQA group nH/nKV <= 8, no attention sink (the caller declines gpt-oss). f32 KV.
// Exponentials are __expf, as in splitkv_softmax. `#pragma` FMA: the dots use explicit __fmaf_rn, and
// kernel_fma_lint_test.go checks no other float multiply-add pair is left to the compiler.
#define FA_WARPS 8
#define FA_NEG (-1e30f)

template <int HD>
__device__ __forceinline__ void fa_partial_impl(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    constexpr int VEC = (HD >= 128) ? 4 : 2;      // output dims per load per lane
    constexpr int NCH = HD / (32 * VEC);          // loads per lane per V row (1 for 64 and 128, 2 for 256)
    constexpr int NQ = HD / 16;                   // float4 steps of the 4-lane score dot
    const int kvh = blockIdx.x, sp = blockIdx.y;
    if (kvh >= nKV || sp >= nSplit) return;
    const int G = nH / nKV;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = lane >> 2, sub = lane & 3;
    const int kvDim = nKV * HD;

    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                               // [G][HD]
    float* pb = smem + G * HD + warp * 64;          // this warp's [8 keys][8 heads] probabilities
    for (int t = threadIdx.x; t < G * HD; t += blockDim.x) qs[t] = q[(long)(kvh * G) * HD + t];
    __syncthreads();

    const int span = nKeys - winStart;
    const int per = (span + nSplit - 1) / nSplit;
    const int lo = winStart + sp * per;
    int hi = lo + per; if (hi > nKeys) hi = nKeys;

    float m[8], l[8], acc[8][NCH * VEC];
#pragma unroll
    for (int g = 0; g < 8; g++) {
        m[g] = FA_NEG; l[g] = 0.f;
#pragma unroll
        for (int c = 0; c < NCH * VEC; c++) acc[g][c] = 0.f;
    }

    for (int b0 = lo + warp * 8; b0 < hi; b0 += FA_WARPS * 8) {
        // ---- scores: one key per 4-lane group ----
        const int key = b0 + grp;
        const bool valid = key < hi;
        float part[8];
#pragma unroll
        for (int g = 0; g < 8; g++) part[g] = 0.f;
        if (valid) {
            const float4* k4 = (const float4*)(kc + (long)key * kvDim + (long)kvh * HD);
#pragma unroll
            for (int j = 0; j < NQ; j++) {
                const float4 kk = k4[j * 4 + sub];
#pragma unroll
                for (int g = 0; g < 8; g++) {
                    if (g < G) {
                        const float4 qq = ((const float4*)qs)[g * (HD / 4) + j * 4 + sub];
                        part[g] = __fmaf_rn(qq.x, kk.x, part[g]);
                        part[g] = __fmaf_rn(qq.y, kk.y, part[g]);
                        part[g] = __fmaf_rn(qq.z, kk.z, part[g]);
                        part[g] = __fmaf_rn(qq.w, kk.w, part[g]);
                    }
                }
            }
        }
        float s[8], alpha[8], p[8];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            float d = part[g];
            d += __shfl_xor_sync(0xffffffff, d, 1);
            d += __shfl_xor_sync(0xffffffff, d, 2);
            s[g] = valid ? d * scale : FA_NEG;
        }
        // ---- online softmax over the block's 8 keys, per head, warp-uniform ----
#pragma unroll
        for (int g = 0; g < 8; g++) {
            if (g < G) {
                float mx = s[g];
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16));
                const float mn = fmaxf(m[g], mx);
                alpha[g] = __expf(m[g] - mn);
                p[g] = valid ? __expf(s[g] - mn) : 0.f;
                float ps = p[g];
                ps += __shfl_xor_sync(0xffffffff, ps, 4);
                ps += __shfl_xor_sync(0xffffffff, ps, 8);
                ps += __shfl_xor_sync(0xffffffff, ps, 16);
                l[g] = __fmaf_rn(l[g], alpha[g], ps);
                m[g] = mn;
                if (sub == 0) pb[grp * 8 + g] = p[g];
            }
        }
        __syncwarp();
        // ---- P.V: all lanes on one key at a time ----
#pragma unroll
        for (int g = 0; g < 8; g++) {
            if (g < G) {
#pragma unroll
                for (int c = 0; c < NCH * VEC; c++) acc[g][c] *= alpha[g];
            }
        }
        const int nb = min(8, hi - b0);
        for (int kk = 0; kk < nb; kk++) {
            const float* vp = vc + (long)(b0 + kk) * kvDim + (long)kvh * HD;
            float v[NCH * VEC];
#pragma unroll
            for (int c = 0; c < NCH; c++) {
                if (VEC == 4) {
                    const float4 t = ((const float4*)vp)[c * 32 + lane];
                    v[c * 4 + 0] = t.x; v[c * 4 + 1] = t.y; v[c * 4 + 2] = t.z; v[c * 4 + 3] = t.w;
                } else {
                    const float2 t = ((const float2*)vp)[lane];
                    v[0] = t.x; v[1] = t.y;
                }
            }
#pragma unroll
            for (int g = 0; g < 8; g++) {
                if (g < G) {
                    const float pv = pb[kk * 8 + g];
#pragma unroll
                    for (int c = 0; c < NCH * VEC; c++) acc[g][c] = __fmaf_rn(pv, v[c], acc[g][c]);
                }
            }
        }
        __syncwarp();
    }

    // ---- unnormalised per-warp partial: O[HD], m, l ----
    const int U = nSplit * FA_WARPS, u = sp * FA_WARPS + warp;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        if (g < G) {
            float* out = partial + ((long)(kvh * G + g) * U + u) * (HD + 4);
#pragma unroll
            for (int c = 0; c < NCH; c++) {
                if (VEC == 4) {
                    ((float4*)out)[c * 32 + lane] = make_float4(acc[g][c * 4], acc[g][c * 4 + 1], acc[g][c * 4 + 2], acc[g][c * 4 + 3]);
                } else {
                    ((float2*)out)[lane] = make_float2(acc[g][0], acc[g][1]);
                }
            }
            if (lane == 0) { out[HD] = m[g]; out[HD + 1] = l[g]; }
        }
    }
}

extern "C" __global__ void fa_partial_64(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<64>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}
extern "C" __global__ void fa_partial_128(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<128>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}
extern "C" __global__ void fa_partial_256(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    int nH, int nKV, int winStart, int nKeys, float scale, int nSplit, float* __restrict__ partial) {
    fa_partial_impl<256>(q, kc, vc, nH, nKV, winStart, nKeys, scale, nSplit, partial);
}

// fa_combine: grid nH, block hd. Merges the U = nSplit*8 unit partials of a head in fixed ascending order.
// A unit with no keys carries m = FA_NEG, l = 0 and contributes nothing; the global max is over units that
// saw a key (m > FA_NEG/2), and at least one always did (the caller guarantees nKeys > winStart).
extern "C" __global__ void fa_combine(const float* __restrict__ partial, int hd, int U, float* __restrict__ ctx) {
    const int h = blockIdx.x, d = threadIdx.x;
    if (d >= hd) return;
    const float* base = partial + (long)h * U * (hd + 4);
    float M = FA_NEG;
    for (int u = 0; u < U; u++) M = fmaxf(M, base[(long)u * (hd + 4) + hd]);
    float num = 0.f, den = 0.f;
    for (int u = 0; u < U; u++) {
        const float* pu = base + (long)u * (hd + 4);
        const float mu = pu[hd];
        if (mu > FA_NEG * 0.5f) {
            const float w = __expf(mu - M);
            num = __fmaf_rn(w, pu[d], num);
            den = __fmaf_rn(w, pu[hd + 1], den);
        }
    }
    ctx[(long)h * hd + d] = num / den;
}
