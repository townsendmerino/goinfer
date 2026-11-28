package wgsl

// DeltaRuleShaderWGSL — Resident Gated-DeltaNet decode step (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8). Design: docs/deltanet-residency-plan.md;
// history: docs/code-notes/gpu.md#DeltaRuleShaderWGSL.
//
// It slots onto the resident DecodeRunner the same way the Mamba-2 engine does — the conv window
// and the gated norm are that engine's (mambaConv / mambaGNorm, same shapes) and the persistent
// per-layer state is a storage buffer updated in place per token. Only the delta rule below is new.
//
// THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, and that is the point of the port rather
// than an implementation detail. decoder/deltanet.go holds S as [hk, hv] and walks it COLUMN-wise:
//
//	for vd:  for kd: kv += S[kd*hv+vd]*k[kd]          // stride hv — a cache line per access
//	         for kd: S[kd*hv+vd] += ...; o += ...     // second pass over the same memory
//
// Here S is [hv, hk], so thread (headV, vd) OWNS the contiguous row S[headV][vd][0:hk] and reads it
// with stride 1. That gives the same thread-owns-its-row race-freedom mambaSSM relies on (no scan,
// no atomics, no position loop) and fixes the CPU's strided access and double pass at the same time.
//
// Per thread, for its own output element vd:
//
//	S[kd] *= gt                        // decay (gt = exp(negExpA·softplus(a+dt_bias)))
//	kv     = Σ_kd S[kd]·k[kd]
//	delta  = (v[vd] − kv)·beta         // beta = sigmoid(b)
//	S[kd] += k[kd]·delta ; o += S[kd]·q[kd]
//	out[headV*hv+vd] = o
//
// vBase lets the caller bind the WHOLE post-conv [q|k|v] buffer and point at its v slice, the
// alignment-free trick mambaSSMOp uses for its in_proj slices — a byte offset of 2*keyDim*4 is not
// guaranteed to satisfy minStorageBufferOffsetAlignment at every geometry.
//
// q and k arrive ALREADY l2-normalized (deltaNorm below): the norms are per KEY head, so computing
// them inside this kernel would repeat each one hv times — the same work as the recurrence itself.
const DeltaRuleShaderWGSL = `
struct P { nv: u32, nk: u32, hk: u32, hv: u32, rep: u32, vBase: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qn:    array<f32>;  // [nk*hk] l2-normalized, scaled
@group(0) @binding(1) var<storage, read>       kn:    array<f32>;  // [nk*hk] l2-normalized
@group(0) @binding(2) var<storage, read>       v:     array<f32>;  // [nv*hv] at vBase
@group(0) @binding(3) var<storage, read>       headP: array<f32>;  // [nv*2]: beta, gt
@group(0) @binding(4) var<storage, read_write> state: array<f32>;  // [nv*hv*hk], in place, [hv,hk]
@group(0) @binding(5) var<storage, read_write> yout:  array<f32>;  // [nv*hv]
@group(0) @binding(6) var<uniform>             p:     P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.nv * p.hv) { return; }
    let headV = t / p.hv;
    let vd    = t % p.hv;
    let headK = headV / p.rep;          // GVA: rep value heads share one key head

    let beta = headP[headV * 2u + 0u];
    let gt   = headP[headV * 2u + 1u];

    let sBase = (headV * p.hv + vd) * p.hk;   // this thread's contiguous row
    let kBase = headK * p.hk;

    // decay + kv, in one pass over the row
    var kv = 0.0;
    for (var kd = 0u; kd < p.hk; kd = kd + 1u) {
        let s = state[sBase + kd] * gt;
        state[sBase + kd] = s;
        kv = kv + s * kn[kBase + kd];
    }
    let delta = (v[p.vBase + headV * p.hv + vd] - kv) * beta;

    var o = 0.0;
    for (var kd = 0u; kd < p.hk; kd = kd + 1u) {
        let s = state[sBase + kd] + kn[kBase + kd] * delta;
        state[sBase + kd] = s;
        o = o + s * qn[kBase + kd];
    }
    yout[headV * p.hv + vd] = o;
}
`

// DeltaNormShaderWGSL — deltaNorm l2-normalizes the per-head q and k slices of the conv output, and applies the query scale. One thread per KEY
// head: the work is nk·hk, far too small to be worth splitting further, and doing it here instead of inside deltaRule keeps
// the recurrence from recomputing each norm hv times.
//
// The conv output is [q(keyDim) ‖ k(keyDim) ‖ v(valueDim)] contiguous, the same packing the CPU reference slices, so the
// layouts agree by construction rather than by comment.
const DeltaNormShaderWGSL = `
struct P { nk: u32, hk: u32, keyDim: u32, _a: u32, qScale: f32, _b: f32, _c: f32, _d: f32 };
@group(0) @binding(0) var<storage, read>       conv: array<f32>;  // [q|k|v]
@group(0) @binding(1) var<storage, read_write> qn:   array<f32>;  // [nk*hk]
@group(0) @binding(2) var<storage, read_write> kn:   array<f32>;  // [nk*hk]
@group(0) @binding(3) var<uniform>             p:    P;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.nk) { return; }
    let base = h * p.hk;

    var qs = 0.0;
    var ks = 0.0;
    for (var i = 0u; i < p.hk; i = i + 1u) {
        let qv = conv[base + i];
        let kvv = conv[p.keyDim + base + i];
        qs = qs + qv * qv;
        ks = ks + kvv * kvv;
    }
    // sqrt(1/(ss+eps)), NOT inverseSqrt(ss). The epsilon is FLA's 1e-6, and it is NOT a precision
    // knob: on a head whose conv output is all zero — reachable, since silu(x) is exactly 0 at
    // x=0 — inverseSqrt(0) is +inf and every downstream state entry becomes NaN, where the CPU
    // reference yields a finite 1e3 scale and a zero vector. TestDeltaNorm_cpuParity gates that
    // case explicitly; at ordinary magnitudes the two forms are indistinguishable, which is why
    // the recurrence gate alone does not catch it.
    let qi = sqrt(1.0 / (qs + 1e-6)) * p.qScale;
    let ki = sqrt(1.0 / (ks + 1e-6));
    for (var i = 0u; i < p.hk; i = i + 1u) {
        qn[base + i] = conv[base + i] * qi;
        kn[base + i] = conv[p.keyDim + base + i] * ki;
    }
}
`

// DeltaGatesShaderWGSL — deltaGates turns the two small per-value-head projections into the pair the delta rule consumes: beta = betaScale·sigmoid(b)
// (betaScale is 1, or 2 for a write gate widened to [0, 2); multiplying by either is exact) and the decay gt = exp(negExpA·softplus(a + dt_bias)). Trivial work, but it has to happen ON DEVICE: the alternative is a
// round-trip per layer per token.
//
// softplus carries torch's threshold=20 linear branch. Without it exp(a) overflows to +inf for large a and the decay becomes
// NaN; the CPU reference does the same thing, so matching it is not optional.
const DeltaGatesShaderWGSL = `
struct P { nv: u32, betaScale: f32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       bt:      array<f32>;  // [nv] write-gate logits
@group(0) @binding(1) var<storage, read>       at:      array<f32>;  // [nv] decay-gate input
@group(0) @binding(2) var<storage, read>       dtBias:  array<f32>;  // [nv]
@group(0) @binding(3) var<storage, read>       negExpA: array<f32>;  // [nv] = -exp(A_log)
@group(0) @binding(4) var<storage, read_write> headP:   array<f32>;  // [nv*2]: beta, gt
@group(0) @binding(5) var<uniform>             p:       P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.nv) { return; }
    let x = at[h] + dtBias[h];
    var sp = x;
    if (x <= 20.0) { sp = log(1.0 + exp(x)); }
    headP[h * 2u + 0u] = (1.0 / (1.0 + exp(-bt[h]))) * p.betaScale;
    headP[h * 2u + 1u] = exp(negExpA[h] * sp);
}
`

// DeltaGNormShaderWGSL — deltaGNorm is DeltaNet's gated RMSNorm — and it is NOT mambaGNorm, which is the reusable-looking
// trap here. Both spell "gated RMSNorm over a head-sized group", but the ORDER differs:
//
//	mamba2.go step 5:   g = y·silu(z);  out = w·g/rms(g)
//	deltanet.go step 4: out = core/rms(core) · w · silu(z)
//
// Mamba normalizes the gated product; DeltaNet normalizes the recurrence output and gates
// afterwards. Feeding one to the other would produce a plausible tensor of the right shape and the
// wrong values, so this is a separate kernel rather than a flag on a shipped, gated one.
//
// One thread per VALUE head. The [hv] weight is shared across heads and indexed by vd, so unlike a
// mamba-style [dInner] weight it needs no per-head tiling at load.
const DeltaGNormShaderWGSL = `
struct P { nv: u32, hv: u32, _a: u32, _b: u32, eps: f32, _c: f32, _d: f32, _e: f32 };
@group(0) @binding(0) var<storage, read>       core:  array<f32>;  // [nv*hv] recurrence output
@group(0) @binding(1) var<storage, read>       z:     array<f32>;  // [nv*hv] output gate (pre-silu)
@group(0) @binding(2) var<storage, read>       normW: array<f32>;  // [hv], shared across heads
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;  // [nv*hv]
@group(0) @binding(4) var<uniform>             p:     P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.nv) { return; }
    let base = h * p.hv;
    var ss = 0.0;
    for (var i = 0u; i < p.hv; i = i + 1u) {
        let v = core[base + i];
        ss = ss + v * v;
    }
    let inv = 1.0 / sqrt(ss / f32(p.hv) + p.eps);
    for (var i = 0u; i < p.hv; i = i + 1u) {
        let zv = z[base + i];
        out[base + i] = core[base + i] * inv * normW[i] * (zv / (1.0 + exp(-zv)));
    }
}
`

// DeltaQSplitShaderWGSL — The SOFTMAX layers of this family are not ordinary GQA either: with attn_output_gate, q_proj emits [query ‖ gate] PER HEAD
// at double width, and the attention context is scaled by sigmoid(gate) before o_proj. Two small kernels rather than a
// load-time weight split, because the weight is quantized: slicing rows out of an int4 WeightMat with its per-group scales is
// real surgery, while splitting the [nH*2*hd] activation is a copy.
//
// deltaQSplit: qg[head*2*hd .. ] → q[head*hd ..] and gate[head*hd ..]. Interleaved PER HEAD, not two concatenated blocks;
// reading it as two blocks gives plausible logits from the wrong tensor (TestQwen35ResidentParity). History:
// docs/code-notes/gpu.md#DeltaQSplitShaderWGSL.
const DeltaQSplitShaderWGSL = `
struct P { n: u32, hd: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       qg:   array<f32>;  // [nH*2*hd]
@group(0) @binding(1) var<storage, read_write> q:    array<f32>;  // [nH*hd]
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;  // [nH*hd]
@group(0) @binding(3) var<uniform>             p:    P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.n) { return; }
    let h = t / p.hd;
    let d = t % p.hd;
    let base = h * 2u * p.hd + d;
    q[t]    = qg[base];
    gate[t] = qg[base + p.hd];
}
`

// DeltaAttnGateShaderWGSL — deltaAttnGate: ctx *= sigmoid(gate), in place, after attention and before o_proj.
const DeltaAttnGateShaderWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> ctx:  array<f32>;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<uniform>             p:    P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.n) { return; }
    ctx[t] = ctx[t] / (1.0 + exp(-gate[t]));
}
`

// MambaSSMShaderWGSL — Resident Mamba-2 selective-SSM decode step (Granite-4.0-H / Nemotron-H hybrids).
// Decode is a BOUNDED per-token recurrence — not the prefill scan — so it slots onto
// the resident DecodeRunner like a KV cache: persistent {conv ring, ssm} state updated
// in place per token inside the single command buffer. See docs/ssm-residency-scope.md.
//
// mambaSSM is the selective state update (mamba2.go step 4): one thread per (head, pi),
// each OWNING its state row ssm[head,pi,·] (N contiguous floats), so the in-place update
// is race-free with no scan and no position loop. Per thread:
//
//	dth = softplus(dtRaw[head] + dtBias[head]);  dA = exp(dth · Aexp[head])
//	for n: S[n] = S[n]·dA + (dth·x)·B[n];  acc += S[n]·C[n]
//	y[head·P+pi] = acc + D[head]·x
//
// conv holds [x(dInner) ‖ B(gSize) ‖ C(gSize)] contiguous (the mambaConv output layout).
// Aexp[head] = -exp(aLog[head]) is precomputed host-side in f64 (the static weight part of
// dA), matching mamba2Step's f64 A; only the per-token dth·exp stays f32. The recurrence is
// stable (dA∈(0,1) ⇒ state decays), so f32 drift over long sequences stays bounded —
// gated at 1/16/256/1k/2k tokens by TestMambaSSM_driftParity.
const MambaSSMShaderWGSL = `
struct P { nHeads: u32, hp: u32, dn: u32, ng: u32, repeat: u32, gSize: u32, dInner: u32, dtBase: u32 };
@group(0) @binding(0) var<storage, read>       conv:  array<f32>;  // [x(dInner) | B(gSize) | C(gSize)]
@group(0) @binding(1) var<storage, read>       dtRaw: array<f32>;  // [.. dtBase ..] (in_proj slice)
@group(0) @binding(2) var<storage, read>       headP: array<f32>;  // [nHeads*3]: Aexp, dtBias, D
@group(0) @binding(3) var<storage, read_write> ssm:   array<f32>;  // [nHeads*hp*dn], in place
@group(0) @binding(4) var<storage, read_write> yout:  array<f32>;  // [dInner]
@group(0) @binding(5) var<uniform>             p:     P;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.nHeads * p.hp) { return; }
    let head = t / p.hp;
    let pi = t % p.hp;
    let g = head / p.repeat;
    let Aexp = headP[head * 3u + 0u];
    let dtb = headP[head * 3u + 1u];
    let Dw = headP[head * 3u + 2u];
    let dth = softplus(dtRaw[p.dtBase + head] + dtb);
    let dA = exp(dth * Aexp);
    let xv = conv[head * p.hp + pi];
    let dx = dth * xv;
    let sBase = (head * p.hp + pi) * p.dn;
    let bBase = p.dInner + g * p.dn;
    let cBase = p.dInner + p.gSize + g * p.dn;
    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < p.dn; n = n + 1u) {
        let v = ssm[sBase + n] * dA + dx * conv[bBase + n];
        ssm[sBase + n] = v;
        acc = acc + v * conv[cBase + n];
    }
    yout[head * p.hp + pi] = acc + Dw * xv;
}
`

// MambaConvShaderWGSL — mambaConv is the depthwise causal conv (mamba2.go step 2): per channel c,
// conv[c] = silu(convB[c] + Σ_{j<K-1} convW[c·K+j]·win[j][c] + convW[c·K+K-1]·xBC[c]),
// then shift the K-1 ring window (drop oldest, append xBC). One thread per channel owns
// column c, so the in-place window shift is race-free. The window starts zeroed — that
// exactly reproduces the reference's early-token zero-padding (verified by trace). win is
// stored [K-1, convDim] oldest-first; K (DConv) ≤ 8.
const MambaConvShaderWGSL = `
struct P { convDim: u32, K: u32, xbcBase: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       xBC:   array<f32>;  // [.. xbcBase ..] (in_proj slice)
@group(0) @binding(1) var<storage, read>       convW: array<f32>;  // [convDim*K]
@group(0) @binding(2) var<storage, read>       convB: array<f32>;  // [convDim]
@group(0) @binding(3) var<storage, read_write> win:   array<f32>;  // [(K-1)*convDim], in place
@group(0) @binding(4) var<storage, read_write> conv:  array<f32>;  // [convDim]
@group(0) @binding(5) var<uniform>             p:     P;

fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.convDim) { return; }
    let K = p.K;
    let xc = xBC[p.xbcBase + c];
    var w: array<f32, 8>;
    var s = convB[c] + convW[c * K + (K - 1u)] * xc;
    for (var j: u32 = 0u; j < K - 1u; j = j + 1u) {
        let wv = win[j * p.convDim + c];
        w[j] = wv;
        s = s + convW[c * K + j] * wv;
    }
    conv[c] = silu(s);
    for (var j: u32 = 0u; j + 1u < K - 1u; j = j + 1u) { win[j * p.convDim + c] = w[j + 1u]; }
    win[(K - 2u) * p.convDim + c] = xc;
}
`

// MambaGNormShaderWGSL — mambaGatedNorm is the gated grouped RMSNorm (mamba2.go step 5): gated = y·silu(z), then
// each of nGroups contiguous groups is RMS-normalized independently × weight. One workgroup
// per group (groupSize = dInner/nGroups). Granite normalizes over full dInner (nGroups=1);
// Nemotron-H per group (nGroups=NGroups).
const MambaGNormShaderWGSL = `
struct P { dInner: u32, nGroups: u32, groupSize: u32, eps: f32, zBase: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       yin:    array<f32>;  // [dInner] (SSM output)
@group(0) @binding(1) var<storage, read>       z:      array<f32>;  // [.. zBase ..] (in_proj slice, gate)
@group(0) @binding(2) var<storage, read>       weight: array<f32>;  // [dInner]
@group(0) @binding(3) var<storage, read_write> gated:  array<f32>;  // [dInner] out
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;

fn siluG(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.x;
    if (g >= p.nGroups) { return; }
    let t = lid.x;
    let base = g * p.groupSize;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.groupSize; i = i + 64u) {
        let v = yin[base + i] * siluG(z[p.zBase + base + i]);
        gated[base + i] = v;
        ss = ss + v * v;
    }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.groupSize) + p.eps);
    for (var i: u32 = t; i < p.groupSize; i = i + 64u) {
        gated[base + i] = weight[base + i] * gated[base + i] * inv;
    }
}
`

// MlaAttnWGSL — MLA residency — DeepSeek / Kimi Multi-head Latent Attention on the GPU
// resident decode path. The compressed-KV latent ([kv_lora_rank | qk_rope_head_dim]
// per position) is the only cached payload; per-head K/V are never materialized.
// This file is the absorb-path core primitive (mirrors decoder.mlaAttentionAbsorb):
// the query is pre-absorbed through W_UK once per head (→ rank-space), each cached
// latent is scored by a rank+rope-dim dot, and V is collapsed to a rank-space weighted
// sum (lifted by W_UV later, on the host side of the runner). The surrounding GEMVs
// (q-LoRA, kv-down, W_UK absorb, W_UV lift, o-proj) reuse the W8A8 path; this kernel
// is the genuinely-new attention shape — a single-query attend with a KV "head" shared
// across all query heads and a score width (latDim) wider than the value width (rank).
//
// Layout — qAbs[h] = qNopeAbs_h[rank] ‖ qRope_h[qkRope] (the absorbed query, width
// latDim = rank+qkRope); lat[j] = cn_j[rank] ‖ krj_j[qkRope] (the stored latent —
// normalized + rope-keyed at store time). Score_j = scale·(qAbs[h]·lat[j]) over the
// full latDim; value = cn_j = lat[j][:rank]; output wsum[h] = Σ_j softmax_j·cn_j.
//
// One workgroup per query head, online (FlashAttention-style) softmax over keys so no
// per-key score row is materialized. The score dot and the value accumulate are both
// STRIDED over the 128-lane workgroup because rank (≤1024) and latDim exceed the lane
// count — each lane owns dims {t, t+128, …}; m/l/corr/pe are scalar (identical across
// lanes, all see the same reduced score) so no extra reduction beyond the score dot.
const MlaAttnWGSL = `
const WG: u32 = 128u;
struct P { nH: u32, latDim: u32, rank: u32, nKeys: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qAbs: array<f32>;  // [nH*latDim] qNopeAbs ‖ qRope per head
@group(0) @binding(1) var<storage, read>       lat:  array<f32>;  // [nKeys*latDim] cn ‖ krj per key
@group(0) @binding(2) var<storage, read_write> wsum: array<f32>;  // [nH*rank] rank-space V collapse
@group(0) @binding(3) var<uniform>             p:    P;
var<workgroup> red: array<f32, WG>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.nH) { return; }
    let t = lid.x;
    let latDim = p.latDim;
    let rank = p.rank;
    let qbase = h * latDim;
    var acc: array<f32, 8>;  // per-lane value accumulators (rank ≤ 8*WG = 1024)
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc[i] = 0.0; }
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    for (var j: u32 = 0u; j < p.nKeys; j = j + 1u) {
        let kbase = j * latDim;
        // score = qAbs[h] · lat[j] over the full latDim (strided across lanes).
        var partial: f32 = 0.0;
        for (var d: u32 = t; d < latDim; d = d + WG) {
            partial = partial + qAbs[qbase + d] * lat[kbase + d];
        }
        red[t] = partial;
        workgroupBarrier();
        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        // value = cn_j = lat[j][:rank]; accumulate the rank dims this lane owns.
        var i: u32 = 0u;
        for (var d: u32 = t; d < rank; d = d + WG) {
            acc[i] = acc[i] * corr + pe * lat[kbase + d];
            i = i + 1u;
        }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();  // all lanes consumed red[0] before next key overwrites red[t]
    }
    var i: u32 = 0u;
    for (var d: u32 = t; d < rank; d = d + WG) {
        wsum[h * rank + d] = acc[i] / l;
        i = i + 1u;
    }
}
`

// MlaLatentStoreWGSL — mlaLatentStore appends one token's compressed KV to the latent cache,
// mirroring decoder's cache.AppendLatent feed: the kv_a_proj output kvDown =
// [latent(rank) | rope-key(qkRope)] is split, the rank latent is RMSNorm'd by
// kvALayernorm (cn, the value + score key body) and the rope-key is decoupled-RoPE'd at
// the token position (krj). Both are normalized/roped at STORE time (the position is
// fixed per entry, so this is equivalent to the CPU's attend-time recompute and saves
// re-roping every cached key each step). interleave=1 is the V3 GPT-J pairwise layout
// (de-interleave to [evens|odds] before NeoX rotate); =0 is plain NeoX. ropeScale folds
// the YaRN attention factor (1.0 when none). One workgroup; base = pos*latDim.
const MlaLatentStoreWGSL = RopeCosSinWGSL + `
struct P { rank: u32, qkRope: u32, pos: u32, eps: f32, base: u32, ropeScale: f32, interleave: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       kvDown:   array<f32>;  // [rank+qkRope] current token
@group(0) @binding(1) var<storage, read>       normW:    array<f32>;  // [rank] kvALayernorm
@group(0) @binding(2) var<storage, read>       invFreq:  array<f32>;  // [qkRope/2]
@group(0) @binding(3) var<storage, read_write> latCache: array<f32>;  // [maxLen*(rank+qkRope)]
@group(0) @binding(4) var<uniform>             p:        P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // RMSNorm over the rank latent → cn.
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.rank; i = i + 64u) { let v = kvDown[i]; ss = ss + v*v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.rank) + p.eps);
    workgroupBarrier();
    for (var i: u32 = t; i < p.rank; i = i + 64u) {
        latCache[p.base + i] = kvDown[i] * inv * normW[i];
    }
    // Decoupled RoPE the qkRope key → krj, written after the rank block.
    let half = p.qkRope / 2u;
    for (var i: u32 = t; i < half; i = i + 64u) {
        let theta = f32(p.pos) * invFreq[i];
        let cs = ropeCosSin(theta); let c = cs.x * p.ropeScale;
        let s = cs.y * p.ropeScale;
        var a: f32;
        var b: f32;
        if (p.interleave == 1u) {
            a = kvDown[p.rank + 2u*i];
            b = kvDown[p.rank + 2u*i + 1u];
        } else {
            a = kvDown[p.rank + i];
            b = kvDown[p.rank + half + i];
        }
        latCache[p.base + p.rank + i]        = a * c - b * s;
        latCache[p.base + p.rank + half + i] = b * c + a * s;
    }
}
`

// MlaHeadMatvecWGSL — mlaHeadMatvec is the per-head block-diagonal matvec behind the absorb (W_UK) and lift
// (W_UV) steps. dst[h*N+n] = Σ_k a[h*aStride+k]·w[(h*N+n)*K+k] — each head
// has its OWN weight block and reads its OWN activation slice, so it's not a single dense
// GEMV. Kept f32 (parity-first, and W_UK/W_UV are small for the 16B targets; the big-V3
// case doesn't fit the 8 GB resident budget regardless). Used twice:
//   - absorb: a=q (aStride=qk_head_dim, reads the qk_nope prefix), w=W_UKᵀ, N=rank, K=qk_nope
//   - lift:   a=wsum (aStride=rank),                                w=W_UV,  N=v_head_dim, K=rank
//
// One workgroup per output element over a 2D grid (H·N can exceed the 65535/dim cap).
// outStride lets the absorb write strided into the combined qAbs buffer ([rank|qkRope]
// per head): outStride=latDim places qNopeAbs at qAbs[h*latDim..+rank]. The lift writes
// contiguously (outStride=N).
const MlaHeadMatvecWGSL = `
struct P { nH: u32, N: u32, K: u32, aStride: u32, outStride: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       a:   array<f32>;  // [nH*aStride] per-head activation
@group(0) @binding(1) var<storage, read>       w:   array<f32>;  // [nH*N*K] per-head weight, row-major
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;  // [nH*outStride]
@group(0) @binding(3) var<uniform>             p:   P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let elem = wid.x + wid.y * 32768u;
    if (elem >= p.nH * p.N) { return; }
    let h = elem / p.N;
    let n = elem % p.N;
    let t = lid.x;
    let abase = h * p.aStride;
    let wbase = (h * p.N + n) * p.K;
    var acc: f32 = 0.0;
    for (var k: u32 = t; k < p.K; k = k + 64u) {
        acc = acc + a[abase + k] * w[wbase + k];
    }
    red[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { dst[h * p.outStride + n] = red[0]; }
}
`

// MlaQRopeWGSL — mlaQRope gathers each head's qk_rope slice from the projected query q [nH*qkHead]
// (laid out [qk_nope | qk_rope] per head), decoupled-RoPEs it at the current position,
// and writes it into the combined qAbs buffer at [h*latDim + rank ..] — directly after
// the W_UK-absorbed qNopeAbs. Same de-interleave+NeoX as the key store (mlaLatentStore),
// so the q·k rope dot matches. One thread per (head, rope-pair).
const MlaQRopeWGSL = RopeCosSinWGSL + `
struct P { nH: u32, qkHead: u32, qkNope: u32, qkRope: u32, rank: u32, latDim: u32, pos: u32, interleave: u32, ropeScale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       q:       array<f32>;  // [nH*qkHead]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [qkRope/2]
@group(0) @binding(2) var<storage, read_write> qAbs:    array<f32>;  // [nH*latDim], rope at [h*latDim+rank ..]
@group(0) @binding(3) var<uniform>             p:       P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.qkRope / 2u;
    let idx = gid.x;
    if (idx >= p.nH * half) { return; }
    let h = idx / half;
    let i = idx % half;
    let qbase = h * p.qkHead + p.qkNope;  // rope dims of head h
    let theta = f32(p.pos) * invFreq[i];
    let cs = ropeCosSin(theta); let c = cs.x * p.ropeScale;
    let s = cs.y * p.ropeScale;
    var a: f32;
    var b: f32;
    if (p.interleave == 1u) {
        a = q[qbase + 2u*i];
        b = q[qbase + 2u*i + 1u];
    } else {
        a = q[qbase + i];
        b = q[qbase + half + i];
    }
    let obase = h * p.latDim + p.rank;
    qAbs[obase + i]        = a * c - b * s;
    qAbs[obase + half + i] = b * c + a * s;
}
`
