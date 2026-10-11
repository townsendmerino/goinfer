// moe_fold.cu — the rank-order fold of the expert-major MoE prefill (cuda/moe_expert_major.go, moe_expert_major_gemma4.go).
//
// The per-row MoE path adds each routed expert into its accumulator inside gemv_w4a8_moe_wacc (moe.cu), whose epilogue
// is ONE fused multiply-add in the shipped moe.ptx: dst = fma(val, wgt, dst), with val = facc*aScale already rounded.
// Expert-major computes the experts in another order, so it stores each term's val unweighted (gemv_w4a8_moe, the same
// reduction with a plain store) and folds here, in rank order, with that same single fused operation. Writing wgt*val
// to a scratch and adding it afterwards rounds the product on its own and is not the per-row arithmetic.
//
// A separate .cu from moe.cu for router_f32.cu's reason: moe.ptx is an audited artifact and is not regenerated for a
// new kernel (cuda/testdata/REGEN.md). FMA-linted (kernel_fma_lint_test.go): the one MAC here is an explicit __fmaf_rn.

extern "C" {

// moe_fold_wacc: x[m,i] = fma(val[m,i], wgt[m*topK + rank], x[m,i]) for M rows of N, flat grid over MN = M*N.
// One launch per rank, in rank order on one stream, so each row's terms are added in the per-row path's order.
__global__ void moe_fold_wacc(float* __restrict__ x, const float* __restrict__ val, const float* __restrict__ wgt,
                              int rank, int topK, int N, int MN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int m = i / N;
    x[i] = __fmaf_rn(val[i], wgt[m * topK + rank], x[i]);
}

} // extern "C"
