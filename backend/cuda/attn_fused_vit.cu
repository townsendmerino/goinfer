// attn_fused_vit.cu -- R8 phase A (docs/measurements/vision-tower-attn-PREREGISTERED.md): the attn_fused.cu kernel shape
// (mma.sync m16n8k8 f16 operands, f32 accumulate, online softmax, K/V streamed through shared memory in BN-key tiles, the
// softmax weights P fed to the second matmul straight from the score registers) specialised for the SigLIP vision tower:
//
//   * NON-CAUSAL, no window, no sinks: every query row attends all M keys. The only mask is the key-tile tail.
//   * head dim HDR (72) zero-padded to HDP (80, a multiple of the mma k-step 8). Padded lanes of Q, K and V are 0, so they
//     add nothing to any dot product or output; output lanes >= HDR are not stored. Global rows keep the REAL stride HDR.
//   * query-tile height BM is a template parameter (64 and 128 are built; BM does not change any row's arithmetic order here,
//     so the arms are bit-identical to each other).
//
// Numerics are attn_fused.cu's, deliberately: Q, K, V and P are rounded to f16, everything accumulates in f32. attn_img_batched
// (the kernel this replaces in cuda/vision_encoder.go) is f32 throughout; that difference is what the tower-level gate measures.
//
// attn_fused.cu's long header (why K/V are staged as f16, why V is stored transposed, why the hd template is constexpr and
// the register-array reasoning) applies unchanged and is not repeated. Regenerate with:  ./build_ptx.sh attn_fused_vit

#include <cuda_fp16.h>

#define BN 64
#define KPAD 8

__device__ __forceinline__ void mma_m16n8k8_f32(float &d0, float &d1, float &d2, float &d3,
                                                unsigned int a0, unsigned int a1, unsigned int b0,
                                                float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <int HDP, int HDR, int BM>
__device__ __forceinline__ void attn_vit_impl(const float* __restrict__ q, const float* __restrict__ kc,
                                              const float* __restrict__ vc, int nH, int nKV, float scale,
                                              int M, float* __restrict__ ctx) {
    constexpr int OT = HDP / 8;
    constexpr int NG = BN / 8;

    const int h = blockIdx.x;
    if (h >= nH) return;
    const int qTile = blockIdx.y * BM;
    if (qTile >= M) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qRow0 = lane >> 2;
    const int qRow1 = qRow0 + 8;
    const int colBase = (lane & 3) * 2;

    const int kvDim = nKV * HDR, group = nH / nKV, kvh = h / group;
    const int qDim = nH * HDR;

    extern __shared__ __half sh[];
    __half* Ksh = sh;                                   // [BN][HDP + KPAD]
    __half* Vtsh = sh + (size_t)BN * (HDP + KPAD);      // [HDP][BN + KPAD]
    constexpr int kStride = HDP + KPAD;
    constexpr int vStride = BN + KPAD;

    const int mBase = qTile + warp * 16;
    const int m0abs = mBase + qRow0, m1abs = mBase + qRow1;
    const bool has0 = m0abs < M, has1 = m1abs < M;

    unsigned int qFrag[OT][2];
#pragma unroll
    for (int c = 0; c < OT; c++) {
        const int d = c * 8 + colBase;
        float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
        if (d < HDR) {   // HDR is a multiple of 8, so d and d+1 are both in range or both padding
            if (has0) { const float* qh = q + (long)m0abs * qDim + h * HDR; v00 = qh[d]; v01 = qh[d + 1]; }
            if (has1) { const float* qh = q + (long)m1abs * qDim + h * HDR; v10 = qh[d]; v11 = qh[d + 1]; }
        }
        __half2 p0 = __floats2half2_rn(v00, v01);
        __half2 p1 = __floats2half2_rn(v10, v11);
        qFrag[c][0] = *reinterpret_cast<unsigned int*>(&p0);
        qFrag[c][1] = *reinterpret_cast<unsigned int*>(&p1);
    }

    float mRun0 = -1e30f, mRun1 = -1e30f, lRun0 = 0.f, lRun1 = 0.f;
    float acc[OT][4];
#pragma unroll
    for (int t = 0; t < OT; t++) { acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f; }

    for (int s0 = 0; s0 < M; s0 += BN) {
        const int nk = (M - s0 < BN) ? M - s0 : BN;
        __syncthreads();
        for (int idx = threadIdx.x; idx < nk * HDP; idx += blockDim.x) {
            const int kk = idx / HDP, dd = idx - kk * HDP;
            float kv = 0.f, vv = 0.f;
            if (dd < HDR) {
                kv = kc[(long)(s0 + kk) * kvDim + kvh * HDR + dd];
                vv = vc[(long)(s0 + kk) * kvDim + kvh * HDR + dd];
            }
            Ksh[kk * kStride + dd] = __float2half_rn(kv);
            Vtsh[dd * vStride + kk] = __float2half_rn(vv);
        }
        __syncthreads();

        float s[NG][4];
#pragma unroll
        for (int g = 0; g < NG; g++) { s[g][0] = 0.f; s[g][1] = 0.f; s[g][2] = 0.f; s[g][3] = 0.f; }
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int kIdx = g * 8 + (lane >> 2);
#pragma unroll
            for (int c = 0; c < OT; c++) {
                unsigned int b0 = 0;
                if (kIdx < nk) b0 = *reinterpret_cast<const unsigned int*>(&Ksh[kIdx * kStride + c * 8 + colBase]);
                mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3], qFrag[c][0], qFrag[c][1], b0,
                                s[g][0], s[g][1], s[g][2], s[g][3]);
            }
        }

        // Non-causal: the only masked keys are the tail of the last tile. Every row therefore has a real score in the
        // first tile, so the running max is finite from tile 0 on and exp(-1e30 - m) underflows to 0 for masked columns
        // (attn_fused.cu's "live" special case, needed for causal rows that have seen no key yet, cannot arise here).
        float tileMax0 = -1e30f, tileMax1 = -1e30f;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int kA = g * 8 + colBase, kB = kA + 1;
            const float a0 = (kA < nk) ? __fmul_rn(s[g][0], scale) : -1e30f;
            const float a1 = (kB < nk) ? __fmul_rn(s[g][1], scale) : -1e30f;
            const float b0 = (kA < nk) ? __fmul_rn(s[g][2], scale) : -1e30f;
            const float b1 = (kB < nk) ? __fmul_rn(s[g][3], scale) : -1e30f;
            s[g][0] = a0; s[g][1] = a1; s[g][2] = b0; s[g][3] = b1;
            tileMax0 = fmaxf(tileMax0, fmaxf(a0, a1));
            tileMax1 = fmaxf(tileMax1, fmaxf(b0, b1));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffffu, tileMax0, off));
            tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffffu, tileMax1, off));
        }
        const float mNew0 = fmaxf(mRun0, tileMax0), mNew1 = fmaxf(mRun1, tileMax1);
        const float alpha0 = __expf(mRun0 - mNew0), alpha1 = __expf(mRun1 - mNew1);

        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const float p0 = __expf(s[g][0] - mNew0);
            const float p1 = __expf(s[g][1] - mNew0);
            const float p2 = __expf(s[g][2] - mNew1);
            const float p3 = __expf(s[g][3] - mNew1);
            s[g][0] = p0; s[g][1] = p1; s[g][2] = p2; s[g][3] = p3;
            sum0 = __fadd_rn(sum0, __fadd_rn(p0, p1));
            sum1 = __fadd_rn(sum1, __fadd_rn(p2, p3));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            sum0 = __fadd_rn(sum0, __shfl_xor_sync(0xffffffffu, sum0, off));
            sum1 = __fadd_rn(sum1, __shfl_xor_sync(0xffffffffu, sum1, off));
        }
        lRun0 = __fmaf_rn(alpha0, lRun0, sum0);
        lRun1 = __fmaf_rn(alpha1, lRun1, sum1);
        mRun0 = mNew0; mRun1 = mNew1;

#pragma unroll
        for (int t = 0; t < OT; t++) {
            acc[t][0] = __fmul_rn(acc[t][0], alpha0);
            acc[t][1] = __fmul_rn(acc[t][1], alpha0);
            acc[t][2] = __fmul_rn(acc[t][2], alpha1);
            acc[t][3] = __fmul_rn(acc[t][3], alpha1);
        }
#pragma unroll
        for (int g = 0; g < NG; g++) {
            __half2 pa = __floats2half2_rn(s[g][0], s[g][1]);
            __half2 pb = __floats2half2_rn(s[g][2], s[g][3]);
            const unsigned int a0 = *reinterpret_cast<unsigned int*>(&pa);
            const unsigned int a1 = *reinterpret_cast<unsigned int*>(&pb);
            const int kIdx = g * 8 + colBase;
#pragma unroll
            for (int t = 0; t < OT; t++) {
                const int dIdx = t * 8 + (lane >> 2);
                unsigned int b0 = 0;
                if (kIdx + 1 < nk) {
                    b0 = *reinterpret_cast<const unsigned int*>(&Vtsh[dIdx * vStride + kIdx]);
                } else if (kIdx < nk) {
                    __half2 hv = __halves2half2(Vtsh[dIdx * vStride + kIdx], __float2half_rn(0.f));
                    b0 = *reinterpret_cast<unsigned int*>(&hv);
                }
                mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, b0,
                                acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
            }
        }
    }

    const float inv0 = (lRun0 > 0.f) ? 1.f / lRun0 : 0.f;
    const float inv1 = (lRun1 > 0.f) ? 1.f / lRun1 : 0.f;
#pragma unroll
    for (int t = 0; t < OT; t++) {
        const int dcol = t * 8 + colBase;
        if (dcol < HDR) {
            if (has0) {
                float* o = ctx + (long)m0abs * qDim + h * HDR;
                o[dcol]     = __fmul_rn(acc[t][0], inv0);
                o[dcol + 1] = __fmul_rn(acc[t][1], inv0);
            }
            if (has1) {
                float* o = ctx + (long)m1abs * qDim + h * HDR;
                o[dcol]     = __fmul_rn(acc[t][2], inv1);
                o[dcol + 1] = __fmul_rn(acc[t][3], inv1);
            }
        }
    }
}

extern "C" __global__ void attn_vit_hd72_bm64(const float* __restrict__ q, const float* __restrict__ kc,
                                              const float* __restrict__ vc, int nH, int nKV, float scale,
                                              int M, float* __restrict__ ctx) {
    attn_vit_impl<80, 72, 64>(q, kc, vc, nH, nKV, scale, M, ctx);
}

extern "C" __global__ void attn_vit_hd72_bm128(const float* __restrict__ q, const float* __restrict__ kc,
                                               const float* __restrict__ vc, int nH, int nKV, float scale,
                                               int M, float* __restrict__ ctx) {
    attn_vit_impl<80, 72, 128>(q, kc, vc, nH, nKV, scale, M, ctx);
}
