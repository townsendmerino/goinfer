// moe_split.cu: the routed-expert combine taken apart, for layers where the host computes some experts
// (docs/tasks/task-multimodal-support-2026-10.md, S20 step 4). gemv_w4a8_moe_wacc computes an expert's down product and
// adds weight * product into the residual in one launch, so the k experts must run in order. Here a device expert stores its
// product (gemv_w4a8_moe_res), a host expert's product is uploaded beside it, and moe_combine replays the k accumulates in
// expert order. Every float operation is an explicit _rn intrinsic, in moe.ptx's order, so res is bit-identical to the
// product inside gemv_w4a8_moe_wacc and the combine to its chain of epilogues, whatever NVRTC builds this file.
#include <cuda_fp16.h>

extern "C" {

// gemv_w4a8_moe_res: gemv_w4a8_moe_wacc's arguments without wgt; res[slot * N + row] = aScale * (row sum).
__global__ void gemv_w4a8_moe_res(const unsigned int* __restrict__ W, const int* __restrict__ a,
                                  const __half* __restrict__ gs, const float* __restrict__ aScalePtr,
                                  const unsigned int* __restrict__ idx, int slot, int rowsPerExpert, int N,
                                  int Kwords, int Kgroups, float* __restrict__ res) {
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
        facc = __fadd_rn(facc, __fmaf_rn(__half2float(sr[wi >> 2]), __int2float_rn(p0),
                                         __fmul_rn(__half2float(sr[wj >> 2]), __int2float_rn(p1))));
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) {
            unsigned int word = wr[wi];
            int p = 0;
            p = __dp4a((int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi], p);
            p = __dp4a((int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u), a[2 * wi + 1], p);
            facc = __fmaf_rn(__half2float(sr[wi >> 2]), __int2float_rn(p), facc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc = __fadd_rn(facc, __shfl_down_sync(0xffffffffu, facc, off));
    if (lane == 0) res[(long)slot * N + row] = __fmul_rn(aScale, facc);
}

// moe_combine: dst[row] = fma(res[j][row], wgt[j], dst[row]) for j = 0 .. k-1 in order, the value k gemv_w4a8_moe_wacc
// launches leave in dst.
__global__ void moe_combine(const float* __restrict__ res, const float* __restrict__ wgt, int k, int N,
                            float* __restrict__ dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float acc = dst[row];
    for (int j = 0; j < k; j++) acc = __fmaf_rn(res[(long)j * N + row], wgt[j], acc);
    dst[row] = acc;
}

// moe_act_to_host: the expert input (n packed int32, four int8 each, and its scale) into mapped host memory, for the host's
// experts to read once an event after this launch completes.
__global__ void moe_act_to_host(const int* __restrict__ q, const float* __restrict__ sc, int n, int* __restrict__ hq,
                                float* __restrict__ hsc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hq[i] = q[i];
    if (i == 0) *hsc = *sc;
}

} // extern "C"
