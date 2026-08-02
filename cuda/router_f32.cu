// router_f32.cu — pure-f32 router projection for the Gemma-4 MoE router.
//
// A SEPARATE .cu from moe.cu on purpose. moe.ptx ships built at CUDA 12.6; this box's NVRTC is 12.9,
// so regenerating moe.ptx to add a kernel would rewrite EVERY shipped kernel's codegen (a toolchain
// bump masquerading as a kernel add). A fresh file adds only router_f32.ptx and leaves the audited
// 12.6 moe.ptx untouched.
//
// WHY f32xf32 and not gemv_f32_a8 (the shared int8-activation router GEMV). Gemma-4's router is the
// one DISCRETE-failure path: a quantization error near a top-k tie picks a DIFFERENT expert — a
// cliff, not a small error. gemv_f32_a8 quantizes the activation to int8 (~1e-2), which can flip a
// decision near a tie; safe only when the margin is wide. The gemma4-moe-tiny fixture's 0.12 routing
// margin was CONSTRUCTED by least-squares to be wide, so a "no flip" result there is circular for a
// TRAINED 128-expert/top-8 router whose 8th-vs-9th boundary is far tighter. Quantizing NOTHING
// removes the perturbation entirely: this kernel is bit-exact to the CPU f32 router modulo f32
// reduction order (~1e-6), so routing cannot flip from activation quant at ANY expert count. That
// retires routing from the resident MoE suspect list permanently. One block per output row.

extern "C" {
__global__ void gemv_f32_f32(const float* __restrict__ W, const float* __restrict__ a,
                             int N, int K, float* __restrict__ dst) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* wr = W + (long)n * K;
    int t = threadIdx.x, nt = blockDim.x;
    float acc = 0.f;
    for (int k = t; k < K; k += nt) acc += wr[k] * a[k];
    extern __shared__ float red[];
    red[t] = acc;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        __syncthreads();
    }
    if (t == 0) dst[n] = red[0];
}
}
