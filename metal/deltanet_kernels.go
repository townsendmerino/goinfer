//go:build darwin

package metal

// deltaNetKernels is the Gated-DeltaNet decode mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8), ported verbatim from
// cuda/deltanet.cu (read its header comment first): the same five-stage split, kernel for kernel, so the CPU capture hook
// (decoder.deltaCapHook) gates each stage the same way on both backends. Metal had no recurrent-state kernel before this
// file, so every kernel here is new code, gated from its own input via the hook's mixed slot.
//
// THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, same as CUDA. decoder/deltanet.go holds S as [hk, hv] and walks it
// column-wise with stride hv, in two passes. Here S is [hv, hk], so thread (headV, vd) owns a contiguous row
// S[headV][vd][0:hk] and reads it stride-1.
//
// delta_gnorm IS NOT the Mamba gated norm. Mamba normalizes the gated product; DeltaNet normalizes the recurrence output and
// gates AFTERWARDS. Substituting one for the other gives a plausible tensor of the right shape and the wrong values
// (measured on the WebGPU port: docs/code-notes/metal.md#deltaNetKernels).
//
// THE L2-NORM EPSILON (delta_norm) IS A ZERO-GUARD, NOT A PRECISION KNOB: sqrt(1/(ss+1e-6)), never rsqrt(ss). silu(0) is
// exactly 0, so an all-zero head is reachable, and rsqrt(0) is +inf, poisoning every downstream state entry with NaN where
// the reference yields a finite scale on a zero vector. This does not show in the chained-drift gate (1e-6 is below f32
// resolution at normal magnitudes); TestDeltaNorm_zeroHead exists specifically to catch it.
//
// FMA DISCIPLINE. Metal has no per-operation explicit-rounding intrinsics the way CUDA's __fmaf_rn/__fmul_rn/__fadd_rn does;
// the whole kernel library compiles under one library-wide fast-math setting (metal/model.go's preciseMathCompile). fma()
// below is used ONLY where the CUDA reference explicitly fuses (__fmaf_rn); every site the CUDA reference deliberately
// leaves unfused (__fmul_rn followed by a separate __fadd_rn) is written here as two separate expressions, mirroring the
// CUDA source's literal shape rather than a mathematically-equivalent rewrite: see
// docs/completed/task-metal-batched-verify-kernel.md on why literal source form, not just arithmetic equivalence, is what
// fast-math contraction keys off.
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

    // Manual 8-wide unroll, shipped in 4090dc45: scalar ~409.8k -> 2-wide ~220k -> 4-wide ~128k ->
    // 8-wide ~108.9k ns/dispatch (~3.76x the scalar loop; the commit records the measurement).
    // Same strictly sequential kd=0,1,...,7,... accumulation
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

// ---- D-B01: the mixer over M prompt rows in one dispatch each (metal/prefill_deltanet.go) ----
// Each kernel below is the decode kernel above with its body copied VERBATIM and only the row
// addressing added: a token loop where the stage is a recurrence (conv's window, rule's state),
// a row index in the grid where it is per-token. So given the same inputs a row's outputs, and the
// window and state after the last row, are the decode kernel's run M times
// (TestDeltaNetSeqKernels_matchDecodeBitwise). The decode kernels are left untouched: refactoring
// them into shared helpers could move decode's own bits under fast-math.

// delta_conv_seq: delta_conv over rows t = 0..M-1, reading mixed as half (the prefill GEMM's
// output, row stride convDim) and writing conv rows of convDim floats.
kernel void delta_conv_seq(device const half* mixed[[buffer(0)]], device const float* convW[[buffer(1)]],
    device float* win[[buffer(2)]], device float* conv[[buffer(3)]],
    constant uint& convDim[[buffer(4)]], constant uint& K[[buffer(5)]], constant uint& M[[buffer(6)]],
    uint c[[thread_position_in_grid]]) {
    if (c >= convDim) return;
    for (uint r=0; r<M; r++) {
        float xc = float(mixed[r*convDim + c]);
        float s = convW[c*K + (K-1)] * xc; // tap j=K-1 is the current token
        for (uint j=0; j<K-1; j++) s = fma(convW[c*K+j], win[j*convDim+c], s);
        conv[r*convDim + c] = dn_silu(s);
        for (uint j=0; j+1<K-1; j++) win[j*convDim+c] = win[(j+1)*convDim+c];
        win[(K-2)*convDim+c] = xc;
    }
}

// delta_gates_rows: delta_gates for M rows; bt, at are [M][nv], headP [M][nv][2].
kernel void delta_gates_rows(device const float* bt[[buffer(0)]], device const float* at[[buffer(1)]],
    device const float* dtBias[[buffer(2)]], device const float* negExpA[[buffer(3)]],
    device float* headP[[buffer(4)]], constant uint& nv[[buffer(5)]], constant uint& M[[buffer(6)]],
    uint i[[thread_position_in_grid]]) {
    if (i >= M*nv) return;
    uint h = i % nv;
    float sp = dn_softplus(at[i] + dtBias[h]);
    headP[i*2+0] = 1.0f / (1.0f + exp(-bt[i]));
    headP[i*2+1] = exp(negExpA[h] * sp);
}

// delta_norm_rows: delta_norm, one threadgroup per (row, key head); conv rows are convDim wide,
// qn/kn rows keyDim.
kernel void delta_norm_rows(device const float* convAll[[buffer(0)]], device float* qnAll[[buffer(1)]],
    device float* knAll[[buffer(2)]], constant uint& nk[[buffer(3)]], constant uint& hk[[buffer(4)]],
    constant uint& keyDim[[buffer(5)]], constant float& qScale[[buffer(6)]],
    constant uint& convDim[[buffer(7)]], constant uint& M[[buffer(8)]],
    uint gid[[threadgroup_position_in_grid]],
    uint t[[thread_index_in_threadgroup]], uint nt[[threads_per_threadgroup]]) {
    if (gid >= M*nk) return;
    uint row = gid / nk, h = gid % nk;
    device const float* conv = convAll + row*convDim;
    device float* qn = qnAll + row*keyDim;
    device float* kn = knAll + row*keyDim;
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
    float qi = sqrt(1.0f/(redQ[0]+1e-6f)) * qScale;
    float ki = sqrt(1.0f/(redK[0]+1e-6f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=t; i<hk; i+=nt) {
        qn[base+i] = conv[base+i] * qi;
        kn[base+i] = conv[keyDim+base+i] * ki;
    }
}

// delta_rule_seq: delta_rule over rows r = 0..M-1. A thread owns one state row for the whole
// loop, so rows need no synchronisation between threads: each row reads only inputs earlier
// dispatches wrote, and its own state row. qn/kn rows are keyDim wide, the v slice sits at vBase
// of a convDim-wide conv row, headP rows are 2*nv, y rows nv*hv.
kernel void delta_rule_seq(device const float* qnAll[[buffer(0)]], device const float* knAll[[buffer(1)]],
    device const float* vAll[[buffer(2)]], device const float* headPAll[[buffer(3)]],
    device float* state[[buffer(4)]], device float* youtAll[[buffer(5)]],
    constant uint& nv[[buffer(6)]], constant uint& hk[[buffer(7)]], constant uint& hv[[buffer(8)]],
    constant uint& rep[[buffer(9)]], constant uint& vBase[[buffer(10)]],
    constant uint& keyDim[[buffer(11)]], constant uint& convDim[[buffer(12)]], constant uint& M[[buffer(13)]],
    uint t[[thread_position_in_grid]]) {
    if (t >= nv*hv) return;
    uint headV = t / hv;
    uint vd = t % hv;
    uint headK = headV / rep;
    device float* S = state + (uint)(headV*hv+vd)*hk;
    for (uint r=0; r<M; r++) {
    device const float* headP = headPAll + r*2*nv;
    device const float* v = vAll + r*convDim;
    device float* yout = youtAll + r*nv*hv;
    float beta = headP[headV*2+0];
    float gt = headP[headV*2+1];
    device const float* k = knAll + r*keyDim + headK*hk;
    device const float* q = qnAll + r*keyDim + headK*hk;

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
}

// delta_gnorm_rows: delta_gnorm, one threadgroup per (row, value head); core and out rows are nv*hv
// floats, z rows are half (the z projection's prefill GEMM output, row stride nv*hv).
kernel void delta_gnorm_rows(device const float* coreAll[[buffer(0)]], device const half* zAll[[buffer(1)]],
    device const float* normW[[buffer(2)]], device float* outAll[[buffer(3)]],
    constant uint& nv[[buffer(4)]], constant uint& hv[[buffer(5)]], constant float& eps[[buffer(6)]],
    constant uint& M[[buffer(7)]],
    uint gid[[threadgroup_position_in_grid]],
    uint t[[thread_index_in_threadgroup]], uint nt[[threads_per_threadgroup]]) {
    if (gid >= M*nv) return;
    uint row = gid / nv, h = gid % nv;
    device const float* core = coreAll + row*nv*hv;
    device const half* z = zAll + row*nv*hv;
    device float* out = outAll + row*nv*hv;
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
        out[base+i] = g * dn_silu(float(z[base+i]));
    }
}

// delta_proj_w8_rows: in_proj_b / in_proj_a for M rows on the prefill lane: half activations
// (rows of K) against int8 weights with one scale per output row, f32 accumulation. One simdgroup
// per (row, output); out rows are N wide. Not decode's W8A8 arithmetic (that quantizes the
// activation to int8 first): the batched lane keeps activations in f16 throughout.
kernel void delta_proj_w8_rows(device const half* x[[buffer(0)]], device const char* w[[buffer(1)]],
    device const float* wsc[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& M[[buffer(6)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    if (gid >= M*N) return;
    uint row = gid / N, n = gid % N;
    device const half* xr = x + row*K;
    device const char* wr = w + n*K;
    float acc = 0.0f;
    for (uint k=lid; k<K; k+=32) acc = fma(float(xr[k]), float(wr[k]), acc);
    acc = simd_sum(acc);
    if (lid == 0) out[row*N+n] = acc * wsc[n];
}
`
