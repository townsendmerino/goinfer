package kernels

// MambaKernels is the Mamba-2 decode mixer (Nemotron-H, Granite-4.0-H), ported from gpu/mamba.go's three shaders, kernel for
// kernel, so the CPU capture hook (decoder.SetMambaCapHook) gates the same chain on both backends. The decode step is a bounded
// per-token recurrence: a conv window and the SSM state are updated in place, one token at a time.
//
//   - mamba_conv: the depthwise causal conv over xBC. One thread owns channel c of the window, so the in-place shift is
//     race-free. The window starts zeroed, which reproduces the CPU's early-token zero padding. K (DConv) is at most 8.
//   - mamba_ssm: the selective state update. One thread owns the state row ssm[head, pi, 0:N]. Aexp[head] = -exp(aLog[head]) is
//     precomputed on the host in f64; only the per-token exp(dth·Aexp) is f32, as on WebGPU.
//   - mamba_gnorm: the gated RMSNorm, y·silu(z) normalized per group. One threadgroup of 64 per group, dispatched as groups×64
//     threads: Nemotron-H normalizes per group (NGroups groups), Granite-4.0-H over the full inner width (one group).
const MambaKernels = `
inline float mb_silu(float x) { return x / (1.0f + exp(-x)); }
inline float mb_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

kernel void mamba_conv(device const float* xBC [[buffer(0)]], device const float* convW [[buffer(1)]],
    device const float* convB [[buffer(2)]], device float* win [[buffer(3)]], device float* conv [[buffer(4)]],
    constant uint& convDim [[buffer(5)]], constant uint& K [[buffer(6)]], constant uint& xbcBase [[buffer(7)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= convDim) return;
    float xc = xBC[xbcBase + c];
    float w[8];
    float s = convB[c] + convW[c * K + (K - 1u)] * xc;
    for (uint j = 0u; j < K - 1u; j++) {
        float wv = win[j * convDim + c];
        w[j] = wv;
        s = s + convW[c * K + j] * wv;
    }
    conv[c] = mb_silu(s);
    for (uint j = 0u; j + 1u < K - 1u; j++) { win[j * convDim + c] = w[j + 1u]; }
    win[(K - 2u) * convDim + c] = xc;
}

kernel void mamba_ssm(device const float* conv [[buffer(0)]], device const float* dtRaw [[buffer(1)]],
    device const float* headP [[buffer(2)]], device float* ssm [[buffer(3)]], device float* yout [[buffer(4)]],
    constant uint& nHeads [[buffer(5)]], constant uint& hp [[buffer(6)]], constant uint& dn [[buffer(7)]],
    constant uint& repeat [[buffer(8)]], constant uint& gSize [[buffer(9)]], constant uint& dInner [[buffer(10)]],
    constant uint& dtBase [[buffer(11)]], uint t [[thread_position_in_grid]]) {
    if (t >= nHeads * hp) return;
    uint head = t / hp, pi = t % hp, g = head / repeat;
    float Aexp = headP[head * 3u + 0u], dtb = headP[head * 3u + 1u], Dw = headP[head * 3u + 2u];
    float dth = mb_softplus(dtRaw[dtBase + head] + dtb);
    float dA = exp(dth * Aexp);
    float xv = conv[head * hp + pi];
    float dx = dth * xv;
    uint sBase = (head * hp + pi) * dn;
    uint bBase = dInner + g * dn;
    uint cBase = dInner + gSize + g * dn;
    float acc = 0.0f;
    for (uint n = 0u; n < dn; n++) {
        float v = ssm[sBase + n] * dA + dx * conv[bBase + n];
        ssm[sBase + n] = v;
        acc = acc + v * conv[cBase + n];
    }
    yout[head * hp + pi] = acc + Dw * xv;
}

kernel void mamba_gnorm(device const float* yin [[buffer(0)]], device const float* z [[buffer(1)]],
    device const float* weight [[buffer(2)]], device float* gated [[buffer(3)]], constant uint& groupSize [[buffer(4)]],
    constant float& eps [[buffer(5)]], constant uint& zBase [[buffer(6)]],
    uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sh[64];
    uint base = g * groupSize;
    float ss = 0.0f;
    for (uint i = t; i < groupSize; i += 64u) {
        float v = yin[base + i] * mb_silu(z[zBase + base + i]);
        gated[base + i] = v;
        ss = ss + v * v;
    }
    sh[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride > 0u; stride /= 2u) {
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(sh[0] / float(groupSize) + eps);
    for (uint i = t; i < groupSize; i += 64u) {
        gated[base + i] = weight[base + i] * gated[base + i] * inv;
    }
}
`
