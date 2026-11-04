// mamba.cu — the Mamba-2 decode mixer (Nemotron-H, Granite-4.0-H) on CUDA, ported kernel for kernel from metal/kernels/mamba_kernels_msl.go (itself
// gpu/mamba.go's three shaders), so decoder.SetMambaCapHook gates the same chain on every backend. The decode step is a bounded per-token
// recurrence: a conv window and the SSM state are updated in place, one token at a time. A new module with its own PTX, so no existing PTX moves.
//
//   - mamba_conv:  the depthwise causal conv over xBC, then SiLU. One thread owns channel c of the window, so the in-place shift is race-free.
//                  The window starts zeroed, which reproduces the CPU's early-token zero padding. K (DConv) is at most 8.
//   - mamba_ssm:   the selective state update. One thread owns the state row ssm[head, pi, 0:N]. aExp[head] = -exp(aLog[head]) is computed on the
//                  host in f64; only the per-token exp(dth·aExp) is f32, as on WebGPU and Metal. The row is walked in n order, the order the CPU sums in.
//   - mamba_gnorm: the gated RMSNorm, y·silu(z) normalized per group (Nemotron-H: nGroups groups; Granite-4.0-H: one group over the inner width).
//                  One block per group; the sum of squares accumulates in f64, like decoder.gatedRMSNormGrouped.
//
// Every float multiply and add is an explicit round-to-nearest intrinsic and none is fused: the CPU reference (decoder/mamba2.go) does not fuse them on
// amd64, and TestKernelFMALint holds this file to the rule.

extern "C" {

__device__ __forceinline__ float mb_silu(float x) { return x / __fadd_rn(1.f, expf(-x)); }
__device__ __forceinline__ float mb_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// conv: [convDim] out. xBC is read at xbcBase (the in_proj output is [z | xBC | dt] in one buffer). win is [(K-1)*convDim], oldest tap first.
__global__ void mamba_conv(const float* __restrict__ xBC, const float* __restrict__ convW, const float* __restrict__ convB,
                           float* __restrict__ win, float* __restrict__ conv, int convDim, int K, int xbcBase) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convDim) return;
    float xc = xBC[xbcBase + c];
    float w[8];
    float s = __fadd_rn(convB[c], __fmul_rn(convW[c * K + (K - 1)], xc));
    for (int j = 0; j < K - 1; j++) {
        float wv = win[j * convDim + c];
        w[j] = wv;
        s = __fadd_rn(s, __fmul_rn(convW[c * K + j], wv));
    }
    conv[c] = mb_silu(s);
    for (int j = 0; j + 1 < K - 1; j++) win[j * convDim + c] = w[j + 1];
    win[(K - 2) * convDim + c] = xc;
}

// ssm: one thread per (head, pi). headP is [nHeads*3] = {aExp, dtBias, D} per head. dtRaw is read at dtBase. conv is [x (dInner) | B (gSize) | C (gSize)].
__global__ void mamba_ssm(const float* __restrict__ conv, const float* __restrict__ dtRaw, const float* __restrict__ headP,
                          float* __restrict__ ssm, float* __restrict__ yout,
                          int nHeads, int hp, int dn, int repeat, int gSize, int dInner, int dtBase) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nHeads * hp) return;
    int head = t / hp, pi = t % hp, g = head / repeat;
    float aExp = headP[head * 3 + 0], dtb = headP[head * 3 + 1], dW = headP[head * 3 + 2];
    float dth = mb_softplus(__fadd_rn(dtRaw[dtBase + head], dtb));
    float dA = expf(__fmul_rn(dth, aExp));
    float xv = conv[head * hp + pi];
    float dx = __fmul_rn(dth, xv);
    int sBase = (head * hp + pi) * dn;
    int bBase = dInner + g * dn;
    int cBase = dInner + gSize + g * dn;
    float acc = 0.f;
    for (int n = 0; n < dn; n++) {
        float v = __fadd_rn(__fmul_rn(ssm[sBase + n], dA), __fmul_rn(dx, conv[bBase + n]));
        ssm[sBase + n] = v;
        acc = __fadd_rn(acc, __fmul_rn(v, conv[cBase + n]));
    }
    yout[head * hp + pi] = __fadd_rn(acc, __fmul_rn(dW, xv));
}

// gnorm: grid = number of groups, block = 256 (a power of two; the reduction below assumes it). gated = weight * (yin * silu(z)) * rsqrt(mean + eps) per group.
// z is read at zBase (the in_proj output's gate slice).
__global__ void mamba_gnorm(const float* __restrict__ yin, const float* __restrict__ z, const float* __restrict__ weight,
                            float* __restrict__ gated, int groupSize, float eps, int zBase) {
    __shared__ double sh[256];
    int g = blockIdx.x, t = threadIdx.x;
    int base = g * groupSize;
    double ss = 0.0;
    for (int i = t; i < groupSize; i += 256) {
        float v = __fmul_rn(yin[base + i], mb_silu(z[zBase + base + i]));
        gated[base + i] = v;
        ss = __fma_rn((double)v, (double)v, ss);
    }
    sh[t] = ss;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) sh[t] = __dadd_rn(sh[t], sh[t + stride]);
        __syncthreads();
    }
    float inv = (float)(1.0 / sqrt(__ddiv_rn(sh[0], (double)groupSize) + (double)eps));
    for (int i = t; i < groupSize; i += 256) gated[base + i] = __fmul_rn(__fmul_rn(weight[base + i], gated[base + i]), inv);
}

} // extern "C"
