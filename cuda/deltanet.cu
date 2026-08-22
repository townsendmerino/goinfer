// deltanet.cu — the Gated-DeltaNet decode mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8).
//
// WHY ITS OWN FILE. moe.ptx and glue.ptx are audited artifacts; decode_splitkv.cu and
// gptoss_act.cu set the precedent that a self-contained kernel gets its own module rather than
// dragging a re-audit into unrelated work. This is more self-contained than either: CUDA has no
// recurrent state at all today — nothing in this directory carries a persistent per-layer buffer,
// a conv ring, or a scan — so none of it belongs anywhere existing.
//
// WHAT IS NEW HERE VS THE WEBGPU PORT. The WebGPU backend reused its Mamba-2 engine for the causal
// conv and the state plumbing, so only the delta rule was new. CUDA has neither, so delta_conv
// below is a port of the Mamba-2 conv rather than a reuse — same ring-buffer discipline, same
// SiLU, and bias-free (DeltaNet's conv has no bias; passing a null bias makes that explicit rather
// than requiring the caller to allocate a zero vector).
//
// THE MATH is decoder/deltanet.go's gatedDeltaNetStep, split at the same four seams the WebGPU
// kernels use, because those seams are where the CPU reference can be captured and compared
// (deltaCapHook):
//
//   1. delta_conv   [q|k|v] = silu(depthwise causal conv over the last K mixed_qkv vectors)
//   2. delta_gates  beta = sigmoid(b);  gt = exp(negExpA · softplus(a + dt_bias))
//   3. delta_norm   q, k l2-normalized PER KEY HEAD; q additionally scaled by 1/sqrt(hk)
//   4. delta_rule   the recurrence itself, state updated in place
//   5. delta_gnorm  out = core/rms(core) · normW · silu(z)
//
// THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, and that is the design, not an artifact.
// decoder/deltanet.go holds S as [hk, hv] and walks it COLUMN-wise with stride hv, in two passes.
// Here S is [hv, hk], so thread (headV, vd) OWNS the contiguous row S[headV][vd][0:hk] and reads
// it with stride 1 — no atomics, no cross-thread sharing, and coalesced across the vd of a warp.
//
// STEP 5 IS NOT THE MAMBA GATED NORM, and the two are easy to confuse because they spell the same
// words. Mamba normalizes the GATED PRODUCT (g = y·silu(z); out = w·g/rms(g)); DeltaNet normalizes
// the recurrence output and gates AFTERWARDS. Substituting one for the other measures cosine 0.986
// with a 12× RMS error on the WebGPU side — a plausible tensor of the right shape and the wrong
// values. Whoever reuses a norm here should re-check the order, not the shape.
//
// EVERY float multiply-accumulate is an explicit intrinsic (__fmaf_rn / __fmul_rn / __fadd_rn),
// per the bit-identity rule TestKernelFMALint enforces: a bare MAC lets the compiler choose
// fma-vs-mul+add, and a recurrence is the worst place to leave that to chance — the difference
// compounds over every token rather than washing out.

__device__ __forceinline__ float dn_silu(float x) { return x / (1.f + __expf(-x)); }

// softplus with torch's threshold=20 linear branch. Without it exp(a) overflows to +inf for large
// a and the decay becomes NaN; the CPU reference (softplusf) has the same branch, so matching it
// is not a robustness nicety but a parity requirement.
__device__ __forceinline__ float dn_softplus(float x) {
    return x > 20.f ? x : __logf(1.f + __expf(x));
}

// delta_conv: depthwise causal conv over [q|k|v] with SiLU, plus the ring-window slide.
//
// One thread per channel c of convDim. win holds the last K-1 mixed vectors, oldest first, and is
// updated in place after the read — the same ring discipline the WebGPU mambaConv uses.
//
// NO PER-THREAD ARRAY, deliberately. The obvious spelling caches the K-1 window taps in a
// `float w[K]` local so the shift can reuse them, and K is a runtime argument, so the array is
// dynamically indexed and lands in LOCAL MEMORY. That is not merely slow: a kernel with per-thread
// local memory makes the driver reserve a backing store sized by occupancy, which A9 measured at
// 138 MB for a single kernel on this card — a cost paid at first launch, on a device where this
// family is already fighting for VRAM. TestKernelLocalMemoryCensus exists to catch exactly this,
// and the first draft of this kernel tripped it (32 B/thread, __local_depot0).
//
// So the shift re-reads the window instead. Ascending j writes slot j from slot j+1, which has not
// been written yet, so the in-place slide is safe without a copy; the re-reads hit L1/L2 having
// just been read by the dot above. K-1 extra cached loads per channel against a 138 MB reservation
// is not a close call.
extern "C" __global__ void delta_conv(const float* __restrict__ mixed,
                                      const float* __restrict__ convW,   // [convDim*K]
                                      float* __restrict__ win,           // [(K-1)*convDim], in place
                                      float* __restrict__ conv,          // [convDim]
                                      int convDim, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convDim) return;
    float xc = mixed[c];
    // tap j=K-1 is the current token; taps 0..K-2 come from the window, oldest first.
    float s = __fmul_rn(convW[c * K + (K - 1)], xc);
    for (int j = 0; j < K - 1; j++) s = __fmaf_rn(convW[c * K + j], win[j * convDim + c], s);
    conv[c] = dn_silu(s);
    for (int j = 0; j + 1 < K - 1; j++) win[j * convDim + c] = win[(j + 1) * convDim + c];
    win[(K - 2) * convDim + c] = xc;
}

// delta_gates: the two per-value-head scalars the recurrence consumes, packed interleaved as
// (beta, gt) so the rule kernel reads one aligned pair per head.
//
// negExpA is ALREADY −exp(A_log) — precomputed at load (negExpAFromLog), and the GGUF path stores
// it that way natively. Re-exponentiating here would be silently wrong on both loaders.
extern "C" __global__ void delta_gates(const float* __restrict__ bt,      // [nv]
                                       const float* __restrict__ at,      // [nv]
                                       const float* __restrict__ dtBias,  // [nv]
                                       const float* __restrict__ negExpA, // [nv]
                                       float* __restrict__ headP,         // [nv*2]
                                       int nv) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    float sp = dn_softplus(__fadd_rn(at[h], dtBias[h]));
    headP[h * 2 + 0] = 1.f / (1.f + __expf(-bt[h]));
    headP[h * 2 + 1] = __expf(__fmul_rn(negExpA[h], sp));
}

// delta_norm: l2-normalize the per-head q and k slices of the conv output and apply the query
// scale. ONE BLOCK PER KEY HEAD, reducing in shared memory — unlike the WebGPU version, which
// gives each key head a single thread, because hk is 128 on every released model and a serial
// 128-element pass per head wastes a CUDA block.
//
// sqrt(1/(ss+eps)), not rsqrtf(ss): the epsilon is FLA's 1e-6 and it is NOT a precision knob. On a
// head whose conv output is all zero — reachable, since silu(0) is exactly 0 — rsqrtf(0) is +inf
// and every downstream state entry becomes NaN, where the reference yields a finite 1e3 scale on a
// zero vector. rsqrtf is also only ~2 ULP where sqrtf and divide are correctly rounded, and this
// feeds a recurrence.
extern "C" __global__ void delta_norm(const float* __restrict__ conv,  // [q|k|v]
                                      float* __restrict__ qn,          // [nk*hk]
                                      float* __restrict__ kn,          // [nk*hk]
                                      int nk, int hk, int keyDim, float qScale) {
    extern __shared__ float red[];   // [2*blockDim]
    int h = blockIdx.x;
    if (h >= nk) return;
    int t = threadIdx.x, nt = blockDim.x, base = h * hk;
    float qs = 0.f, ks = 0.f;
    for (int i = t; i < hk; i += nt) {
        float qv = conv[base + i], kv = conv[keyDim + base + i];
        qs = __fmaf_rn(qv, qv, qs);
        ks = __fmaf_rn(kv, kv, ks);
    }
    red[t] = qs; red[nt + t] = ks;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o) { red[t] += red[t + o]; red[nt + t] += red[nt + t + o]; }
        __syncthreads();
    }
    float qi = __fmul_rn(sqrtf(1.f / (red[0] + 1e-6f)), qScale);
    float ki = sqrtf(1.f / (red[nt] + 1e-6f));
    __syncthreads();
    for (int i = t; i < hk; i += nt) {
        qn[base + i] = __fmul_rn(conv[base + i], qi);
        kn[base + i] = __fmul_rn(conv[keyDim + base + i], ki);
    }
}

// delta_rule: the recurrence. One thread per (headV, vd); each owns a contiguous state row.
//
//   S[kd] *= gt                      // decay
//   kv     = sum_kd S[kd]·k[kd]
//   delta  = (v[vd] − kv)·beta
//   S[kd] += k[kd]·delta ; o += S[kd]·q[kd]
//
// The decay and the kv dot are fused into ONE pass over the row, and the update and the output dot
// into a second — two passes over hk floats, against the CPU reference's two strided passes.
//
// vBase lets the caller point at the v slice of the whole post-conv [q|k|v] buffer instead of
// copying it out.
extern "C" __global__ void delta_rule(const float* __restrict__ qn,    // [nk*hk] normalized
                                      const float* __restrict__ kn,    // [nk*hk] normalized
                                      const float* __restrict__ v,     // [nv*hv] at vBase
                                      const float* __restrict__ headP, // [nv*2] (beta, gt)
                                      float* __restrict__ state,       // [nv*hv*hk], in place, [hv,hk]
                                      float* __restrict__ yout,        // [nv*hv]
                                      int nv, int hk, int hv, int rep, int vBase) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nv * hv) return;
    int headV = t / hv;
    int vd = t % hv;
    int headK = headV / rep;              // GVA: rep value heads share one key head
    float beta = headP[headV * 2 + 0];
    float gt = headP[headV * 2 + 1];
    float* S = state + (size_t)(headV * hv + vd) * hk;
    const float* k = kn + headK * hk;
    const float* q = qn + headK * hk;

    // UNROLL 8-WIDE, SINGLE ACCUMULATOR. The fold order is unchanged: kvdot is still one chain of
    // __fmaf_rn in ascending kd, which is what keeps this bit-identical. Multiple accumulators would
    // be the obvious way to expose ILP and are FORBIDDEN here — they reorder a float sum. What the
    // unroll buys is independent LOADS (S[kd], k[kd]) in flight while the dependent FMA chain
    // retires, which costs nothing numerically because the explicit intrinsics stop the compiler
    // reassociating in either direction.
    float kvdot = 0.f;
#pragma unroll 8
    for (int kd = 0; kd < hk; kd++) {
        float s = __fmul_rn(S[kd], gt);
        S[kd] = s;
        kvdot = __fmaf_rn(s, k[kd], kvdot);
    }
    float delta = __fmul_rn(__fadd_rn(v[vBase + headV * hv + vd], -kvdot), beta);
    float o = 0.f;
#pragma unroll 8
    for (int kd = 0; kd < hk; kd++) {
        float s = __fmaf_rn(k[kd], delta, S[kd]);
        S[kd] = s;
        o = __fmaf_rn(s, q[kd], o);
    }
    yout[headV * hv + vd] = o;
}

// delta_gnorm: DeltaNet's gated RMSNorm — normalize the recurrence output, THEN gate.
// One block per value head. normW is [hv], shared across heads and indexed by vd, so unlike a
// mamba-style [dInner] weight it needs no per-head tiling at load.
extern "C" __global__ void delta_gnorm(const float* __restrict__ core,  // [nv*hv]
                                       const float* __restrict__ z,     // [nv*hv] pre-silu gate
                                       const float* __restrict__ normW, // [hv]
                                       float* __restrict__ out,         // [nv*hv]
                                       int nv, int hv, float eps) {
    extern __shared__ float red[];
    int h = blockIdx.x;
    if (h >= nv) return;
    int t = threadIdx.x, nt = blockDim.x, base = h * hv;
    float ss = 0.f;
    for (int i = t; i < hv; i += nt) { float c = core[base + i]; ss = __fmaf_rn(c, c, ss); }
    red[t] = ss;
    __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = rsqrtf(red[0] / hv + eps);
    __syncthreads();
    for (int i = t; i < hv; i += nt) {
        float g = __fmul_rn(__fmul_rn(core[base + i], inv), normW[i]);
        out[base + i] = __fmul_rn(g, dn_silu(z[base + i]));
    }
}

// The SOFTMAX layers of this family are not ordinary GQA either: with attn_output_gate, q_proj
// emits [query ‖ gate] PER HEAD at double width and the context is scaled by sigmoid(gate) before
// o_proj. Interleaved per head, NOT two concatenated blocks — reading it as two blocks yields
// plausible logits from the wrong tensor (measured cosine 0.90 on the WebGPU side).
//
// The split is on the ACTIVATION, not the weight, because the weight is quantized and slicing rows
// out of an int4 bundle with its per-group scales is real surgery.
extern "C" __global__ void delta_qsplit(const float* __restrict__ qg,  // [nH*2*hd]
                                        float* __restrict__ q,         // [nH*hd]
                                        float* __restrict__ gate,      // [nH*hd]
                                        int n, int hd) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int h = t / hd, d = t % hd;
    int base = h * 2 * hd + d;
    q[t] = qg[base];
    gate[t] = qg[base + hd];
}

// delta_attn_gate: ctx *= sigmoid(gate), in place, after attention and before o_proj.
extern "C" __global__ void delta_attn_gate(float* __restrict__ ctx,
                                           const float* __restrict__ gate, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    ctx[t] = ctx[t] / (1.f + __expf(-gate[t]));
}
