// gptoss_act.cu — gpt-oss's clamped interleaved-SwiGLU expert epilogue.
//
// WHY ITS OWN FILE. The equivalent for every other family is glu_quant in glue.cu, and
// glue.ptx is an AUDITED artifact — as is moe.ptx, where the expert GEMVs live. Adding a
// family-specific activation to either would drag a re-audit into what is otherwise a
// self-contained kernel. decode_splitkv.cu set this precedent for exactly the same reason
// ("Own file — glue.ptx / moe.ptx untouched"), so this follows it: a new module, a new PTX,
// nothing audited modified.
//
// THE MATH, from decoder/forward_gptoss.go's gptOssExpert (which is itself matched to HF's
// GptOssExperts):
//
//     gate = clamp(Gate·h + gateBias, max = limit)        // UPPER bound only
//     up   = clamp(Up·h   + upBias,   [-limit, limit])    // BOTH bounds
//     glu  = gate · sigmoid(alpha · gate)
//     d    = (up + 1) · glu
//
// THE ASYMMETRY IS REAL AND EASY TO GET WRONG: the gate branch is clamped only from above,
// the up branch from both sides. Clamping both symmetrically looks tidier, changes nothing
// on most inputs, and is wrong precisely where the activation saturates — which is where a
// clamp exists to matter at all. The gate test drives both branches past ±limit for that
// reason.
//
// The +1 on the linear branch and the alpha-scaled sigmoid are likewise not decoration: with
// alpha = 1.702, sigmoid(alpha·g) approximates GELU's gate, and the +1 keeps the linear
// branch centred on 1 rather than 0.
//
// Everything after the activation — the max-reduction, the symmetric int8 quantization and
// the packed store — is glu_quant's, unchanged, so the down-projection GEMV consumes an
// identical format.

// THE BIASES ARE PER-EXPERT and selected on the DEVICE, so they cannot be passed as a row
// pointer: which expert runs is decided by the router, and the launch geometry must stay
// identical regardless (that is what makes the MoE path graph-static). So the kernel takes
// the whole [nExpert][2*I] table plus the routing index and does the same arithmetic
// gemv_w4a8_moe does — biasRow = idx[slot] * 2*I — with gate at +0 and up at +I, matching
// the gate‖up packing the weights already use. Passing biasGU == nullptr disables biases.
extern "C" __global__ void glu_quant_gptoss(const float* __restrict__ g, const float* __restrict__ u,
                                 int gOff, int uOff, int I,
                                 const float* __restrict__ biasGU, const int* __restrict__ idx, int slot,
                                 float alpha, float limit,
                                 int* __restrict__ q, float* __restrict__ scale,
                                 float* __restrict__ dscratch) {
    const float* gateBias = nullptr; const float* upBias = nullptr;
    if (biasGU) { const float* b = biasGU + (long)idx[slot] * 2 * I; gateBias = b; upBias = b + I; }
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < I; k += nt) {
        float gx = g[gOff + k];
        float ux = u[uOff + k];
        if (gateBias) gx += gateBias[k];
        if (upBias)   ux += upBias[k];
        // gate: upper clamp ONLY. up: both sides.
        gx = fminf(gx, limit);
        ux = fminf(fmaxf(ux, -limit), limit);
        float glu = gx / (1.f + __expf(-alpha * gx));   // gx * sigmoid(alpha*gx)
        float d = (ux + 1.f) * glu;
        dscratch[k] = d; ma = fmaxf(ma, fabsf(d));
    }
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) *scale = sc;
    for (int j = t; j < I / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) { int v = __float2int_rn(dscratch[4 * j + b] * inv); v = max(-127, min(127, v)); packed |= (v & 0xff) << (8 * b); }
        q[j] = packed;
    }
}
