//go:build darwin

package metal

// deltaNetKernels — the Gated-DeltaNet decode mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8),
// ported verbatim from cuda/deltanet.cu (read its header comment first; this file mirrors it
// kernel-for-kernel and keeps the same five-stage split so the CPU capture hook
// (decoder.deltaCapHook) gates each stage the same way on both backends).
//
// WHAT IS NEW HERE VS THE OTHER TWO PORTS. WebGPU reused its Mamba-2 engine for the causal conv
// and the state plumbing, so only the delta rule was new there. Metal has no recurrent-state
// kernel of any kind before this file (same starting point CUDA was in), so every kernel below is
// new code, gated from its own input via the capture hook's mixed slot — not just delta_rule.
//
// THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, same as CUDA. decoder/deltanet.go holds S
// as [hk, hv] and walks it column-wise with stride hv, in two passes. Here S is [hv, hk], so
// thread (headV, vd) owns a contiguous row S[headV][vd][0:hk] and reads it stride-1.
//
// delta_gnorm IS NOT the Mamba gated norm. Mamba normalizes the gated product; DeltaNet
// normalizes the recurrence output and gates AFTERWARDS. Substituting one for the other measured
// cosine 0.986 with a 12x RMS error on the WebGPU side — a plausible tensor of the right shape
// and the wrong values.
//
// THE L2-NORM EPSILON (delta_norm) IS A ZERO-GUARD, NOT A PRECISION KNOB: sqrt(1/(ss+1e-6)),
// never rsqrt(ss). silu(0) is exactly 0, so an all-zero head is reachable, and rsqrt(0) is +inf,
// poisoning every downstream state entry with NaN where the reference yields a finite scale on a
// zero vector. This does not show in the chained-drift gate (1e-6 is below f32 resolution at
// normal magnitudes) — TestDeltaNorm_zeroHead exists specifically to catch it.
//
// FMA DISCIPLINE. Metal has no per-operation explicit-rounding intrinsics the way CUDA's
// __fmaf_rn/__fmul_rn/__fadd_rn does; the whole kernel library compiles under one library-wide
// fast-math setting (metal/model.go's preciseMathCompile). fma() below is used ONLY where the
// CUDA reference explicitly fuses (__fmaf_rn); every site the CUDA reference deliberately leaves
// unfused (__fmul_rn followed by a separate __fadd_rn) is written here as two separate
// expressions, mirroring the CUDA source's literal shape rather than a mathematically-equivalent
// rewrite — see metal/batched_verify_kernels.go's note on why literal source form, not just
// arithmetic equivalence, is what fast-math contraction keys off.
const deltaNetKernels = `
inline float dn_silu(float x) { return x / (1.0f + exp(-x)); }

// softplus with torch's threshold=20 linear branch — without it exp(a) overflows to +inf for
// large a and the decay becomes NaN; the CPU reference (softplusf) has the same branch.
inline float dn_softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// delta_conv: depthwise causal conv over [q|k|v] with SiLU, plus the ring-window slide.
// One thread per channel c of convDim. win holds the last K-1 mixed vectors, oldest first, and
// is updated in place after the read. NO per-thread array (K is a runtime argument; a local
// float[K] would land in device-allocated per-thread scratch on a family already tight on VRAM,
// the same local-memory trap CUDA's TestKernelLocalMemoryCensus exists to catch) — the shift
// re-reads the window instead of caching it, ascending j so slot j reads from slot j+1 before it
// is overwritten, safe in place without a copy.
kernel void delta_conv(device const float* mixed[[buffer(0)]], device const float* convW[[buffer(1)]],
    device float* win[[buffer(2)]], device float* conv[[buffer(3)]],
    constant uint& convDim[[buffer(4)]], constant uint& K[[buffer(5)]],
    uint c[[thread_position_in_grid]]) {
    if (c >= convDim) return;
    float xc = mixed[c];
    float s = convW[c*K + (K-1)] * xc; // tap j=K-1 is the current token
    for (uint j=0; j<K-1; j++) s = fma(convW[c*K+j], win[j*convDim+c], s);
    conv[c] = dn_silu(s);
    for (uint j=0; j+1<K-1; j++) win[j*convDim+c] = win[(j+1)*convDim+c];
    win[(K-2)*convDim+c] = xc;
}

// delta_gates: the two per-value-head scalars the recurrence consumes, packed interleaved as
// (beta, gt) so delta_rule reads one aligned pair per head. negExpA is ALREADY -exp(A_log),
// precomputed at load — re-exponentiating here would be silently wrong on both loaders.
kernel void delta_gates(device const float* bt[[buffer(0)]], device const float* at[[buffer(1)]],
    device const float* dtBias[[buffer(2)]], device const float* negExpA[[buffer(3)]],
    device float* headP[[buffer(4)]], constant uint& nv[[buffer(5)]],
    uint h[[thread_position_in_grid]]) {
    if (h >= nv) return;
    float sp = dn_softplus(at[h] + dtBias[h]);
    headP[h*2+0] = 1.0f / (1.0f + exp(-bt[h]));
    headP[h*2+1] = exp(negExpA[h] * sp);
}

// delta_norm: l2-normalize the per-head q and k slices of the conv output and apply the query
// scale. One threadgroup per key head (tgReduceAttn width, matching hk=128 on every released
// model), reducing q's and k's sum-of-squares in separate static threadgroup arrays — same two
// independent reductions the CUDA kernel packs into one dynamic shared block, just laid out as
// two static ones (no numeric difference: each reduction tree sums the same per-lane partials in
// the same order).
kernel void delta_norm(device const float* conv[[buffer(0)]], device float* qn[[buffer(1)]],
    device float* kn[[buffer(2)]], constant uint& nk[[buffer(3)]], constant uint& hk[[buffer(4)]],
    constant uint& keyDim[[buffer(5)]], constant float& qScale[[buffer(6)]],
    uint h[[threadgroup_position_in_grid]],
    uint t[[thread_index_in_threadgroup]], uint nt[[threads_per_threadgroup]]) {
    if (h >= nk) return;
    threadgroup float redQ[128];
    threadgroup float redK[128];
    uint base = h*hk;
    float qs=0.0f, ks=0.0f;
    for (uint i=t; i<hk; i+=nt) {
        float qv=conv[base+i], kv=conv[keyDim+base+i];
        qs = fma(qv, qv, qs);
        ks = fma(kv, kv, ks);
    }
    redQ[t]=qs; redK[t]=ks;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=nt>>1; o>0; o>>=1) {
        if (t<o) { redQ[t]+=redQ[t+o]; redK[t]+=redK[t+o]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // sqrt(1/(ss+eps)), NOT rsqrt(ss) — see the file header. eps is FLA's 1e-6.
    float qi = sqrt(1.0f/(redQ[0]+1e-6f)) * qScale;
    float ki = sqrt(1.0f/(redK[0]+1e-6f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=t; i<hk; i+=nt) {
        qn[base+i] = conv[base+i] * qi;
        kn[base+i] = conv[keyDim+base+i] * ki;
    }
}

// delta_rule: the recurrence. One thread per (headV, vd); each owns a contiguous state row
// state[(headV*hv+vd)*hk : +hk], read/written stride-1. No threadgroup reduction — a flat 1D
// dispatch, matching the CUDA launch shape exactly (LaunchConfig1D(nv*hv, 128), no shared mem).
//
//   S[kd] *= gt                      // decay
//   kv     = sum_kd S[kd]*k[kd]
//   delta  = (v[vd] - kv)*beta
//   S[kd] += k[kd]*delta ; o += S[kd]*q[kd]
//
// vBase lets the caller point at the v slice of the whole post-conv [q|k|v] buffer (conv's
// output) instead of copying it out; it offsets ONLY the v read, not the state row.
kernel void delta_rule(device const float* qn[[buffer(0)]], device const float* kn[[buffer(1)]],
    device const float* v[[buffer(2)]], device const float* headP[[buffer(3)]],
    device float* state[[buffer(4)]], device float* yout[[buffer(5)]],
    constant uint& nv[[buffer(6)]], constant uint& hk[[buffer(7)]], constant uint& hv[[buffer(8)]],
    constant uint& rep[[buffer(9)]], constant uint& vBase[[buffer(10)]],
    uint t[[thread_position_in_grid]]) {
    if (t >= nv*hv) return;
    uint headV = t / hv;
    uint vd = t % hv;
    uint headK = headV / rep; // GVA: rep value heads share one key head
    float beta = headP[headV*2+0];
    float gt = headP[headV*2+1];
    device float* S = state + (uint)(headV*hv+vd)*hk;
    device const float* k = kn + headK*hk;
    device const float* q = qn + headK*hk;

    // CANDIDATE: manual 8-wide unroll (grounded in the measured trend: scalar ~409.8k -> 2-wide
    // ~220k -> 4-wide ~128k ns/dispatch; testing whether the win keeps compounding or plateaus/
    // regresses on register pressure). Same strictly sequential kd=0,1,...,7,... accumulation
    // order as the scalar loop (no reassociation) -- just written in groups of 8, bit-identical; a
    // scalar tail handles a non-multiple-of-8 hk defensively even though every released family's hk
    // (128) is a clean multiple.
    float kvdot = 0.0f;
    uint kd = 0;
    for (; kd+7 < hk; kd += 8) {
        float k0=k[kd],k1=k[kd+1],k2=k[kd+2],k3=k[kd+3],k4=k[kd+4],k5=k[kd+5],k6=k[kd+6],k7=k[kd+7];
        float s0=S[kd]*gt,s1=S[kd+1]*gt,s2=S[kd+2]*gt,s3=S[kd+3]*gt;
        float s4=S[kd+4]*gt,s5=S[kd+5]*gt,s6=S[kd+6]*gt,s7=S[kd+7]*gt;
        S[kd]=s0; S[kd+1]=s1; S[kd+2]=s2; S[kd+3]=s3; S[kd+4]=s4; S[kd+5]=s5; S[kd+6]=s6; S[kd+7]=s7;
        kvdot = fma(s0, k0, kvdot);
        kvdot = fma(s1, k1, kvdot);
        kvdot = fma(s2, k2, kvdot);
        kvdot = fma(s3, k3, kvdot);
        kvdot = fma(s4, k4, kvdot);
        kvdot = fma(s5, k5, kvdot);
        kvdot = fma(s6, k6, kvdot);
        kvdot = fma(s7, k7, kvdot);
    }
    for (; kd < hk; kd++) {
        float s = S[kd] * gt;
        S[kd] = s;
        kvdot = fma(s, k[kd], kvdot);
    }
    float delta = (v[vBase+headV*hv+vd] - kvdot) * beta;
    float o = 0.0f;
    kd = 0;
    for (; kd+7 < hk; kd += 8) {
        float k0=k[kd],k1=k[kd+1],k2=k[kd+2],k3=k[kd+3],k4=k[kd+4],k5=k[kd+5],k6=k[kd+6],k7=k[kd+7];
        float q0=q[kd],q1=q[kd+1],q2=q[kd+2],q3=q[kd+3],q4=q[kd+4],q5=q[kd+5],q6=q[kd+6],q7=q[kd+7];
        float s0 = fma(k0, delta, S[kd]);
        float s1 = fma(k1, delta, S[kd+1]);
        float s2 = fma(k2, delta, S[kd+2]);
        float s3 = fma(k3, delta, S[kd+3]);
        float s4 = fma(k4, delta, S[kd+4]);
        float s5 = fma(k5, delta, S[kd+5]);
        float s6 = fma(k6, delta, S[kd+6]);
        float s7 = fma(k7, delta, S[kd+7]);
        S[kd]=s0; S[kd+1]=s1; S[kd+2]=s2; S[kd+3]=s3; S[kd+4]=s4; S[kd+5]=s5; S[kd+6]=s6; S[kd+7]=s7;
        o = fma(s0, q0, o);
        o = fma(s1, q1, o);
        o = fma(s2, q2, o);
        o = fma(s3, q3, o);
        o = fma(s4, q4, o);
        o = fma(s5, q5, o);
        o = fma(s6, q6, o);
        o = fma(s7, q7, o);
    }
    for (; kd < hk; kd++) {
        float s = fma(k[kd], delta, S[kd]);
        S[kd] = s;
        o = fma(s, q[kd], o);
    }
    yout[headV*hv+vd] = o;
}

// delta_gnorm: DeltaNet's gated RMSNorm — normalize the recurrence output, THEN gate (NOT the
// Mamba shape; see the file header). One threadgroup per value head. normW is [hv], shared
// across heads and indexed by vd. rsqrt is fine here (unlike delta_norm): this normalizes the
// RECURRENCE OUTPUT, which the state's own decay keeps away from the zero-vector edge case that
// makes delta_norm's epsilon load-bearing.
kernel void delta_gnorm(device const float* core[[buffer(0)]], device const float* z[[buffer(1)]],
    device const float* normW[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& nv[[buffer(4)]], constant uint& hv[[buffer(5)]], constant float& eps[[buffer(6)]],
    uint h[[threadgroup_position_in_grid]],
    uint t[[thread_index_in_threadgroup]], uint nt[[threads_per_threadgroup]]) {
    if (h >= nv) return;
    threadgroup float red[128];
    uint base = h*hv;
    float ss=0.0f;
    for (uint i=t; i<hv; i+=nt) { float c=core[base+i]; ss = fma(c, c, ss); }
    red[t]=ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o=nt>>1; o>0; o>>=1) { if (t<o) red[t]+=red[t+o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(red[0]/float(hv) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=t; i<hv; i+=nt) {
        float g = core[base+i] * inv * normW[i];
        out[base+i] = g * dn_silu(z[base+i]);
    }
}

// The SOFTMAX layers of this family are not ordinary GQA either: with attn_output_gate, q_proj
// emits [query | gate] PER HEAD at double width and the context is scaled by sigmoid(gate)
// before o_proj. Interleaved per head, NOT two concatenated blocks — reading it as two blocks
// yields plausible logits from the wrong tensor (measured cosine 0.90 on the WebGPU side). The
// split is on the ACTIVATION, not the weight, because the weight is quantized and slicing rows
// out of an int4 bundle with its per-group scales is real surgery.
kernel void delta_qsplit(device const float* qg[[buffer(0)]], device float* q[[buffer(1)]],
    device float* gate[[buffer(2)]], constant uint& n[[buffer(3)]], constant uint& hd[[buffer(4)]],
    uint t[[thread_position_in_grid]]) {
    if (t >= n) return;
    uint h = t / hd, d = t % hd;
    uint base = h*2*hd + d;
    q[t] = qg[base];
    gate[t] = qg[base+hd];
}

// delta_attn_gate: ctx *= sigmoid(gate), in place, after attention and before o_proj.
kernel void delta_attn_gate(device float* ctx[[buffer(0)]], device const float* gate[[buffer(1)]],
    constant uint& n[[buffer(2)]], uint t[[thread_position_in_grid]]) {
    if (t >= n) return;
    ctx[t] = ctx[t] / (1.0f + exp(-gate[t]));
}
`
