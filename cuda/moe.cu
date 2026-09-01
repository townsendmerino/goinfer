// moe.cu — sparse mixture-of-experts decode kernels for the cgo-free CUDA backend.
//
// Mirrors the design already validated on WebGPU (gpu/moe.go) and Metal (metal/moe.go): the
// router runs ON the GPU and writes idx[k]/wgt[k]; the experts are ROW-STACKED into one
// buffer per projection, so an expert is selected by INDEXING (idx[slot]*rowsPerExpert + row)
// rather than by binding a different buffer per token. That keeps the launch geometry fixed
// and the dispatch chain static, which is what the resident runner needs.
//
// Compiled to PTX offline by build_ptx.sh (NVRTC compute_75), go:embed'd, driver-JIT'd.

#include <cuda_fp16.h>

// NVRTC compiles without the host <math.h>, so INFINITY is not defined. __int_as_float is the
// CUDA-native spelling and is what the fp32 +inf bit pattern means anyway.
#define MOE_NEG_INF (-__int_as_float(0x7f800000))

// MOE_MAX_E bounds moe_route's per-thread scratch (score[]/sel[] below). Raised 256 → 512 so
// Kimi-K2 (384 routed experts) stops declining to CPU on this backend; DeepSeek-V4-Pro's 384 is
// covered by the same number. It stops SHORT of Kimi-K3's 896 on purpose — that family is not built,
// and an unbuilt model should not set a validated limit.
//
// CHANGING THIS REQUIRES REGENERATING THE AUDITED moe.ptx. That is allowed, but only at the PINNED
// SAME toolchain (NVRTC 12.6.85) and only with the byte-identical rebuild-unchanged control run
// first — see cuda/testdata/REGEN.md, and the expanded policy note in router_f32.cu. The 256→512
// regen's entire PTX diff was moe_route's stack depot (2368 → 4416 B = 2×512×4 score/sel + 64×4
// gscore + 64 keep) plus three stack offsets; gemv_f32_a8 / gemv_w4a8_moe / gemv_w4a8_moe_wacc /
// shared_gate_combine came out byte-identical.
//
// Cost when nE is SMALL: the depot is sized unconditionally, so every MoE model now carries a 4416-B
// per-thread local frame instead of 2368. moe_route runs ONE thread once per layer per token, so
// that is ~2 KB of extra local memory on a single thread — measured no decode regression on a
// mixtral-class model at 8 experts (see the CHANGELOG entry for the A/B).
#define MOE_MAX_E 512 // router cap; buildMoE rejects larger
#define MOE_MAX_G 64  // group-limited routing cap

extern "C" {

// moe_route: score -> select top-k -> weights. ONE thread: nE is <=MOE_MAX_E and this runs once per
// layer per token, so the parallel version would cost more in launch/reduction overhead than it
// saves. Semantics mirror decoder's CPU router exactly (see gpu/moe.go, metal/moe.go).
//
//   sigmoid=0 -> softmax over logits; sigmoid=1 -> per-expert sigmoid (DeepSeek/GLM).
//   sel = score + bias   (bias is zeros when the arch has none)
//   nGroup>1 -> DeepSeek group-limited: partition into nGroup contiguous groups, score each by
//               the SUM OF ITS TOP-2 sel values, keep the best topkGroup groups, mask the rest.
//   top-k on sel (strict > so ties take the LOWEST index — matches the CPU), but the emitted
//   weight is score[best], NOT sel[best]: the bias steers SELECTION only, never the weight.
//   norm=1 -> renormalize the k weights to sum 1; scale != 0,1 -> routed_scaling_factor.
__global__ void moe_route(const float* __restrict__ logits, const float* __restrict__ bias,
                          unsigned int* __restrict__ outIdx, float* __restrict__ outWgt,
                          int nE, int k, int sigmoid, int norm, float scale,
                          int nGroup, int topkGroup) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float score[MOE_MAX_E];
    float sel[MOE_MAX_E];
    if (sigmoid) {
        for (int i = 0; i < nE; i++) score[i] = 1.0f / (1.0f + __expf(-logits[i]));
    } else {
        float mx = logits[0];
        for (int i = 1; i < nE; i++) mx = fmaxf(mx, logits[i]);
        float sum = 0.f;
        for (int i = 0; i < nE; i++) { float e = __expf(logits[i] - mx); score[i] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int i = 0; i < nE; i++) score[i] *= inv;
    }
    for (int i = 0; i < nE; i++) sel[i] = score[i] + bias[i];

    if (nGroup > 1) {
        int gsz = nE / nGroup;
        float gscore[MOE_MAX_G];
        for (int g = 0; g < nGroup; g++) {
            float t1 = MOE_NEG_INF, t2 = MOE_NEG_INF;
            for (int i = g * gsz; i < (g + 1) * gsz; i++) {
                float v = sel[i];
                if (v > t1) { t2 = t1; t1 = v; }
                else if (v > t2) { t2 = v; }
            }
            gscore[g] = t1 + t2;
        }
        bool keep[MOE_MAX_G];
        for (int g = 0; g < nGroup; g++) keep[g] = false;
        for (int j = 0; j < topkGroup; j++) {
            int bg = 0; float bv = MOE_NEG_INF; bool found = false;
            for (int g = 0; g < nGroup; g++) if (!keep[g] && gscore[g] > bv) { bv = gscore[g]; bg = g; found = true; }
            if (found) keep[bg] = true;
        }
        for (int g = 0; g < nGroup; g++)
            if (!keep[g])
                for (int i = g * gsz; i < (g + 1) * gsz; i++) sel[i] = MOE_NEG_INF;
    }

    float wsum = 0.f;
    for (int j = 0; j < k; j++) {
        int best = 0; float bv = MOE_NEG_INF;
        for (int i = 0; i < nE; i++) if (sel[i] > bv) { bv = sel[i]; best = i; }
        outIdx[j] = (unsigned int)best;
        outWgt[j] = score[best]; // the WEIGHT is the raw score; bias only steered selection
        wsum += score[best];
        sel[best] = MOE_NEG_INF;
    }
    if (norm && wsum > 0.f) for (int j = 0; j < k; j++) outWgt[j] /= wsum;
    if (scale != 0.f && scale != 1.f) for (int j = 0; j < k; j++) outWgt[j] *= scale;
}

// gemv_f32_a8: the ROUTER projection — f32 weights [nE, H] against the int8-quantized
// activation. The router stays f32 because its output steers a DISCRETE choice: quantizing it
// risks flipping an expert near a tie, and a flipped expert is a cliff, not a small error
// (goinfer's own SSM work traced a 66%-agreement wall to exactly that class of discrete flip).
// One block per output row.
__global__ void gemv_f32_a8(const float* __restrict__ W, const int* __restrict__ a,
                            const float* __restrict__ aScalePtr, int N, int K,
                            float* __restrict__ dst) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* wr = W + (long)n * K;
    float aScale = *aScalePtr;
    int t = threadIdx.x, nt = blockDim.x;
    float acc = 0.f;
    // a packs 4 int8 per int, matching the dense path's activation layout.
    for (int k4 = t; k4 < K / 4; k4 += nt) {
        int packed = a[k4];
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int q = (int)(char)((packed >> (8 * b)) & 0xff);
            acc += wr[4 * k4 + b] * (float)q;
        }
    }
    extern __shared__ float red[];
    red[t] = acc; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    if (t == 0) dst[n] = red[0] * aScale;
}

// gemv_w4a8_moe: an INDEXED stacked-expert W4A8 GEMV. Identical math to gemv_w4a8_fwd, except
// the weight row is offset by the ROUTED expert: idx[slot]*rowsPerExpert + row. That is the
// whole trick — one buffer holds every expert's rows, so selecting an expert is arithmetic on
// the row index and the launch geometry never changes with the routing.
__global__ void gemv_w4a8_moe(const unsigned int* __restrict__ W, const int* __restrict__ a,
                              const __half* __restrict__ gs, const float* __restrict__ aScalePtr,
                              const unsigned int* __restrict__ idx, int slot, int rowsPerExpert,
                              int N, int Kwords, int Kgroups, float* __restrict__ dst) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= N) return;
    int lane = threadIdx.x & 31;
    long wrow = (long)idx[slot] * rowsPerExpert + row;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    float aScale = *aScalePtr;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi = base + lane, wj = wi + 32;
        unsigned int w0 = wr[wi], w1 = wr[wj];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj + 1], p1);
        facc += (float)p0 * __half2float(sr[wi >> 2]) + (float)p1 * __half2float(sr[wj >> 2]);
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) { // ragged tail: Kwords need NOT be a multiple of 32
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
            facc += (float)p * __half2float(sr[wi >> 2]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) dst[row] = facc * aScale;
}

// gemv_w4a8_moe_wacc: the same indexed GEMV, but the epilogue WEIGHT-ACCUMULATES into dst:
//   dst[row] += wgt[slot] * result
// This is the expert combine. Accumulating in the epilogue means the k experts sum straight
// into the residual stream with no separate combine pass and no per-expert scratch.
__global__ void gemv_w4a8_moe_wacc(const unsigned int* __restrict__ W, const int* __restrict__ a,
                                   const __half* __restrict__ gs, const float* __restrict__ aScalePtr,
                                   const unsigned int* __restrict__ idx, const float* __restrict__ wgt,
                                   int slot, int rowsPerExpert, int N, int Kwords, int Kgroups,
                                   float* __restrict__ dst) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= N) return;
    int lane = threadIdx.x & 31;
    long wrow = (long)idx[slot] * rowsPerExpert + row;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    float aScale = *aScalePtr;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi = base + lane, wj = wi + 32;
        unsigned int w0 = wr[wi], w1 = wr[wj];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj + 1], p1);
        facc += (float)p0 * __half2float(sr[wi >> 2]) + (float)p1 * __half2float(sr[wj >> 2]);
    }
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
    if (lane == 0) dst[row] += wgt[slot] * (facc * aScale);
}

// gemv_w4a8_moe_wacc_bias: gemv_w4a8_moe_wacc plus gpt-oss's PER-EXPERT DOWN-PROJECTION BIAS.
//
//   dst[row] += wgt[slot] * (Down_e · act + downBias[e][row])
//
// THE BIAS IS INSIDE THE PARENTHESES AND THAT IS THE WHOLE POINT. The router weight scales the
// expert's OUTPUT, and the bias is part of that output — decoder/forward_gptoss.go computes
// `dst = Down·h + downBias` per expert and only then `out += w·dst`. Adding it after the scale
// (dst += w*(Down·h) + bias) would be wrong by a factor of w, and wrong AGAIN because every one
// of the k routed experts would add it rather than each contributing its own scaled copy.
//
// TWO INDEX ARRAYS, NOT ONE, and they differ only when expert caching is on:
//   idx[slot]  WHERE THE WEIGHTS LIVE   → a SLOT id under caching (the weight row moved)
//   bidx[slot] WHICH EXPERT IS RUNNING  → always the EXPERT id (this table never moves)
// Binding one for the other is the defect fixed in d9829ce for the gate‖up bias table; it is
// silent, because the two coincide whenever caching is off. bias == nullptr disables the term,
// so a family without per-expert down biases can share this kernel.
__global__ void gemv_w4a8_moe_wacc_bias(const unsigned int* __restrict__ W, const int* __restrict__ a,
                                        const __half* __restrict__ gs, const float* __restrict__ aScalePtr,
                                        const unsigned int* __restrict__ idx, const unsigned int* __restrict__ bidx,
                                        const float* __restrict__ bias, const float* __restrict__ wgt,
                                        int slot, int rowsPerExpert, int N, int Kwords, int Kgroups,
                                        float* __restrict__ dst) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= N) return;
    int lane = threadIdx.x & 31;
    long wrow = (long)idx[slot] * rowsPerExpert + row;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    float aScale = *aScalePtr;
    float facc = 0.f;
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi = base + lane, wj = wi + 32;
        unsigned int w0 = wr[wi], w1 = wr[wj];
        int p0 = 0, p1 = 0;
        p0 = __dp4a((int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p0);
        p0 = __dp4a((int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p0);
        p1 = __dp4a((int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj], p1);
        p1 = __dp4a((int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wj + 1], p1);
        facc += (float)p0 * __half2float(sr[wi >> 2]) + (float)p1 * __half2float(sr[wj >> 2]);
    }
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
        float b = bias ? bias[(long)bidx[slot] * N + row] : 0.f;
        dst[row] += wgt[slot] * (facc * aScale + b);
    }
}

// shared_gate_combine: the always-on shared expert's contribution.
//   gated   (Qwen-MoE):     dst[i] += sigmoid(gl[0]) * shDown[i]
//   ungated (GLM/DeepSeek): dst[i] += shDown[i]
__global__ void shared_gate_combine(float* __restrict__ dst, const float* __restrict__ shDown,
                                    const float* __restrict__ gl, int N, int ungated) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = ungated ? 1.0f : (1.0f / (1.0f + __expf(-gl[0])));
    dst[i] += g * shDown[i];
}

} // extern "C"
