// router_f32.cu — pure-f32 router projection for the Gemma-4 MoE router.
//
// A SEPARATE .cu from moe.cu on purpose. moe.ptx ships built at CUDA 12.6; this box's NVRTC is 12.9,
// so regenerating moe.ptx AT 12.9 would rewrite EVERY shipped kernel's codegen (a toolchain bump
// masquerading as a kernel add). A fresh file adds only router_f32.ptx and leaves the audited
// 12.6 moe.ptx untouched.
//
// READ THE RULE PRECISELY: it is "never regenerate at a DIFFERENT toolchain", NOT "never regenerate".
// A regen at the PINNED, IDENTICAL version (NVRTC 12.6.85) is the sanctioned path for editing a
// kernel that already lives in moe.ptx, because it is provably a no-op on every unrelated kernel —
// establish that by rebuilding UNCHANGED and confirming the artifact is byte-identical BEFORE making
// the edit. That control plus a per-kernel diff is what makes the change auditable. Procedure and
// exact wheel version: cuda/testdata/REGEN.md (used to raise MOE_MAX_E 256→512).
//
// So: add a NEW kernel → new file (this one). Change an EXISTING moe.ptx kernel → pinned regen.
// Do not reach for a duplicate second implementation to dodge the regen; two routers that must agree
// is a worse failure mode than one auditable artifact.
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
    for (int k = t; k < K; k += nt) acc = __fmaf_rn(wr[k], a[k], acc); // explicit fma: pin the router GEMV (audit R-04)
    extern __shared__ float red[];
    red[t] = acc;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        __syncthreads();
    }
    if (t == 0) dst[n] = red[0];
}

// scale_wgt_by_expert: fold Gemma-4's learned per-expert scale into the routed weights, AFTER
// moe_route's top-k + renormalize (CPU: wts[j] = (topv[j]/sum) * perExpertScale[idx[j]]). idx lives
// in a device buffer (moe_route's output), so this must run ON-GPU: a host fold would read idx/wgt
// back per token, reintroducing exactly the per-token sync the on-device router exists to avoid.
// K lanes (K = top_k, tiny), one dispatch. In router_f32's module so it stays off the audited
// moe.ptx — same reason as gemv_f32_f32.
__global__ void scale_wgt_by_expert(float* __restrict__ wgt, const unsigned int* __restrict__ idx,
                                    const float* __restrict__ perExpertScale, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    wgt[k] *= perExpertScale[idx[k]];
}

// rmsnorm_nw: weightless RMSNorm, OUT-OF-PLACE (src → dst). The Gemma-4 MoE router norms the RAW
// residual h WITHOUT mutating it — h still feeds the parallel dense branch, the expert branch, and
// the final residual add, so the in-place rmsnorm_f32 can't be used here. dst[i] = src[i] *
// rsqrt(mean(src^2)+eps). The learned routerScale and hidden^-0.5 are folded into the router weight
// columns at build time, so nothing else is applied here. One block.
__global__ void rmsnorm_nw(const float* __restrict__ src, float* __restrict__ dst, int H, float eps) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss = __fmaf_rn(src[k], src[k], ss); // explicit fma: pin the rmsnorm reduction (audit R-04)
    red[t] = ss;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        __syncthreads();
    }
    float rnorm = rsqrtf(red[0] / H + eps);
    for (int k = t; k < H; k += nt) dst[k] = src[k] * rnorm;
}

// scale_vec: x[i] *= s. Gemma-4's per-layer output scalar (out = (h + combined) * layerScalar) —
// applied to the residual after the joint post-norm. One dispatch over hidden.
__global__ void scale_vec(float* __restrict__ x, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= s;
}
}
