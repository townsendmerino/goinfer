package wgsl

const MoeRouteWGSL = `
const MAXE: u32 = 512u;
struct P { nE: u32, k: u32, sigmoid: u32, norm: u32, scale: f32, hasBias: u32, nGroup: u32, topkGroup: u32 };
@group(0) @binding(0) var<storage, read>       logits: array<f32>;  // [nE] router logits
@group(0) @binding(1) var<storage, read>       bias:   array<f32>;  // [nE] selection bias (read only if hasBias)
@group(0) @binding(2) var<storage, read_write> outIdx: array<u32>;  // [k] chosen expert indices
@group(0) @binding(3) var<storage, read_write> outWgt: array<f32>;  // [k] chosen expert weights
@group(0) @binding(4) var<uniform>             p:      P;
@compute @workgroup_size(1)
fn main() {
    let nE = min(p.nE, MAXE);
    var score: array<f32, 512>;  // un-biased score (softmax prob or sigmoid) — MAXE
    var sel:   array<f32, 512>;  // selection score (score + bias) — MAXE
    if (p.sigmoid == 1u) {
        for (var i: u32 = 0u; i < nE; i = i + 1u) { score[i] = 1.0 / (1.0 + exp(-logits[i])); }
    } else {
        var mx: f32 = logits[0];
        for (var i: u32 = 1u; i < nE; i = i + 1u) { mx = max(mx, logits[i]); }
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < nE; i = i + 1u) { let e = exp(logits[i] - mx); score[i] = e; sum = sum + e; }
        for (var i: u32 = 0u; i < nE; i = i + 1u) { score[i] = score[i] / sum; }
    }
    for (var i: u32 = 0u; i < nE; i = i + 1u) {
        sel[i] = score[i];
        if (p.hasBias == 1u) { sel[i] = sel[i] + bias[i]; }
    }
    // Group-limited selection (DeepSeek nGroup>1): partition the nE selection scores into
    // nGroup contiguous groups, score each group by its top-2 sum, keep the topkGroup best
    // groups, mask every expert outside them to -inf. Mirrors decoder.groupLimit.
    if (p.nGroup > 1u) {
        let gsz = nE / p.nGroup;
        var gscore: array<f32, 32>;
        for (var g: u32 = 0u; g < p.nGroup; g = g + 1u) {
            var t1: f32 = -3.4e38;
            var t2: f32 = -3.4e38;
            for (var i: u32 = g*gsz; i < (g+1u)*gsz; i = i + 1u) {
                let v = sel[i];
                if (v > t1) { t2 = t1; t1 = v; } else if (v > t2) { t2 = v; }
            }
            gscore[g] = t1 + t2;
        }
        var keep: array<bool, 32>;
        for (var g: u32 = 0u; g < p.nGroup; g = g + 1u) { keep[g] = false; }
        for (var j: u32 = 0u; j < p.topkGroup; j = j + 1u) {
            var bg: u32 = 0u;
            var bv: f32 = -3.4e38;
            for (var g: u32 = 0u; g < p.nGroup; g = g + 1u) {
                if (!keep[g] && gscore[g] > bv) { bv = gscore[g]; bg = g; }
            }
            keep[bg] = true;
        }
        for (var g: u32 = 0u; g < p.nGroup; g = g + 1u) {
            if (!keep[g]) {
                for (var i: u32 = g*gsz; i < (g+1u)*gsz; i = i + 1u) { sel[i] = -3.4e38; }
            }
        }
    }
    // top-k by selection score; weight is the un-biased score. Mask by selection index
    // (set sel to -inf after picking) so a chosen expert is not re-picked.
    var wsum: f32 = 0.0;
    for (var j: u32 = 0u; j < p.k; j = j + 1u) {
        var best: u32 = 0u;
        var bestv: f32 = -3.4e38;
        for (var i: u32 = 0u; i < nE; i = i + 1u) {
            if (sel[i] > bestv) { bestv = sel[i]; best = i; }
        }
        outIdx[j] = best;
        outWgt[j] = score[best];
        wsum = wsum + score[best];
        sel[best] = -3.4e38;
    }
    if (p.norm == 1u && wsum > 0.0) {
        for (var j: u32 = 0u; j < p.k; j = j + 1u) { outWgt[j] = outWgt[j] / wsum; }
    }
    if (p.scale != 0.0 && p.scale != 1.0) {
        for (var j: u32 = 0u; j < p.k; j = j + 1u) { outWgt[j] = outWgt[j] * p.scale; }
    }
}
`

// RouteGptOssWGSL — route_gptoss (FeatAttnSink's MoE half): gpt-oss's router
// disagrees with MoeRouteWGSL about what the bias means, so it is its own kernel rather than a
// parameter on that one — mirroring cuda/gptoss_act.cu's route_gptoss and metal/moe.go's twin
// exactly (same contract, same math):
//
//	MoeRouteWGSL (DeepSeek/GLM/Mixtral/Qwen2-MoE): sel = score + bias; top-k on sel;
//	                                               weight = score[best] (the UNBIASED score)
//	                                               — "the bias steers SELECTION only".
//	route_gptoss:                                 logits += bias; top-k on the BIASED logits;
//	                                               weight = softmax over the SELECTED top-k
//	                                               logits (bias included).
//
// Reference — decoder/forward_gptoss.go's gptOssMoE — addBias(logits, RouterBias) before top-k,
// then softmax over the top-k selected logits. Single workgroup, same MAXE cap as MoeRouteWGSL.
const RouteGptOssWGSL = `
const MAXE: u32 = 512u;
struct P { nE: u32, k: u32, hasBias: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       logits: array<f32>;  // [nE] router logits
@group(0) @binding(1) var<storage, read>       bias:   array<f32>;  // [nE] router bias (read only if hasBias)
@group(0) @binding(2) var<storage, read_write> outIdx: array<u32>;  // [k] chosen expert indices
@group(0) @binding(3) var<storage, read_write> outWgt: array<f32>;  // [k] chosen expert weights
@group(0) @binding(4) var<uniform>             p:      P;
@compute @workgroup_size(1)
fn main() {
    let nE = min(p.nE, MAXE);
    var sc:     array<f32, 512>;  // logits + bias — the BIASED value, used for BOTH selection and weight
    var taken:  array<bool, 512>;
    for (var i: u32 = 0u; i < nE; i = i + 1u) {
        sc[i] = logits[i];
        if (p.hasBias == 1u) { sc[i] = sc[i] + bias[i]; }
        taken[i] = false;
    }
    var chosen: array<f32, 512>;  // the k selected BIASED logits, in selection order
    for (var j: u32 = 0u; j < p.k; j = j + 1u) {
        var best: u32 = 0u;
        var bv: f32 = -3.4e38;
        for (var i: u32 = 0u; i < nE; i = i + 1u) {
            if (!taken[i] && sc[i] > bv) { bv = sc[i]; best = i; }
        }
        taken[best] = true;
        outIdx[j] = best;
        chosen[j] = bv;
    }
    // softmax over the k chosen (biased) logits — NOT the full nE, and NOT the unbiased score.
    var mx: f32 = chosen[0];
    for (var j: u32 = 1u; j < p.k; j = j + 1u) { mx = max(mx, chosen[j]); }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < p.k; j = j + 1u) { chosen[j] = exp(chosen[j] - mx); sum = sum + chosen[j]; }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < p.k; j = j + 1u) { outWgt[j] = chosen[j] * inv; }
}
`

// GptossGluQuantWGSL (FeatAttnSink's clamped interleaved-SwiGLU expert): gpt-oss's expert
// activation is a DIFFERENT function from swigluQuantWGSL's plain silu(gate)·up, not a
// parameterization of it — mirrors cuda/gptoss_act.cu's glu_quant_gptoss / metal/moe.go's
// swiglu_quant_gptoss exactly:
//
//	gate = clamp(Gate·h + gateBias, max = limit)        // UPPER bound only
//	up   = clamp(Up·h   + upBias,   [-limit, limit])    // BOTH bounds
//	glu  = gate · sigmoid(alpha · gate)
//	d    = (up + 1) · glu
//
// The bias table is expert-major, gate then up: [idx[slot]*2*I .. +I) is gate's bias,
// [+I .. +2I) is up's. hasBias==0 (impossible for a real gpt-oss layer, but matching every
// other "always bound, gated by flag" convention here) skips the bias add entirely.
const GptossGluQuantWGSL = `
struct P { n: u32, np: u32, slot: u32, hasBias: u32, alpha: f32, limit: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;  // [n]
@group(0) @binding(1) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(2) var<storage, read>       biasGU: array<f32>;  // [nE*2*n] expert-major gate‖up bias
@group(0) @binding(3) var<storage, read>       idx:    array<u32>;  // [k] chosen expert indices
@group(0) @binding(4) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(5) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(6) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn gptoss_glu(gx0: f32, ux0: f32, alpha: f32, limit: f32) -> f32 {
    let gx = min(gx0, limit);
    let ux = clamp(ux0, -limit, limit);
    let glu = gx / (1.0 + exp(-alpha * gx));
    return (ux + 1.0) * glu;
}
fn mid(i: u32) -> f32 {
    var gx = gate[i];
    var ux = up[i];
    if (p.hasBias == 1u) {
        let base = idx[p.slot] * 2u * p.n;
        gx = gx + biasGU[base + i];
        ux = ux + biasGU[base + p.n + i];
    }
    return gptoss_glu(gx, ux, p.alpha, p.limit);
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, abs(mid(i))); }
    sh[t] = mx;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = max(sh[t], sh[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    var s: f32 = sh[0] / 127.0;
    if (s == 0.0) { s = 1.0; }
    if (t == 0u) { scales[0] = s; }
    let invs = 1.0 / s;
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.n) {
                q = i32(round(mid(k) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// MoeExpertGptOssDownGEMVWGSL (FeatAttnSink's down-projection combine): MoeExpertGEMVWGSL's
// mode-1 epilogue plus a per-(expert,row) bias — decoder/forward_gptoss.go's gptOssExpertDown
// computes `dst = Down·h + downBias` BEFORE the router weight scales the result
// (gptOssMoE: `out += wgt·dst`), so the combine is `wgt[slot]*(r + bias[idx[slot]*N+row])`, NOT
// a plain `+= bias[row]` — mirrors metal/moe.go's gemv_w4a8_moe_wacc_bias exactly, including its
// warning that the bias table is indexed by the routed EXPERT id (idx[slot]), never by a
// weight-cache slot id (this backend uploads every expert unconditionally — gpu/moe.go's
// ResidentStackedW8A8 has no paging/caching layer — so that particular confusion is structurally
// moot here, but the indexing convention is kept identical for anyone porting a future cache).
const MoeExpertGptOssDownGEMVWGSL = `
struct Dims { kp: u32, n: u32, slot: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] quantized activation
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/16] stacked experts
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [nE*N] stacked
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<storage, read>       idx:     array<u32>;        // [k] chosen expert indices
@group(0) @binding(6) var<storage, read>       wgt:     array<f32>;        // [k] chosen expert weights
@group(0) @binding(7) var<storage, read>       dbias:   array<f32>;        // [nE*N] expert-major down bias
@group(0) @binding(8) var<uniform>             dims:    Dims;
fn unpack_i8x4g(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotwg(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4g(a); let bv = unpack_i8x4g(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}
var<workgroup> partg: array<i32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let kv = dims.kp / 16u;
    let row = e * dims.n + n;
    let bBase = row * kv;
    var acc: i32 = 0;
    for (var v: u32 = t; v < kv; v = v + 64u) {
        let a4 = aq[v]; let b4 = bq[bBase + v];
        acc = acc + dotwg(a4.x, b4.x) + dotwg(a4.y, b4.y) + dotwg(a4.z, b4.z) + dotwg(a4.w, b4.w);
    }
    partg[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partg[t] = partg[t] + partg[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = f32(partg[0]) * aScales[0] * bScales[row];
        dst[n] = dst[n] + wgt[dims.slot] * (r + dbias[row]);
    }
}
`

// MoeExpertGEMVWGSL — Indexed sparse-expert GEMV. Identical int8 GEMV math to gemvW8A8, but the
// weight ROW base is computed from a DYNAMIC expert index read out of the routing buffer
// (idx[slot], produced on-GPU by moeRoute) into a STACKED [nE,N,kp] weight buffer — so the
// resident plan records a FIXED k dispatches (one per selected expert slot) while the
// actual expert is chosen at run time, no host round-trip. mode 1 folds the router weight
// into the epilogue (dst[n] += wgt[slot]·r) for the down-projection combine; mode 0
// overwrites (gate/up into their own scratch).
const MoeExpertGEMVWGSL = `
struct Dims { kp: u32, n: u32, slot: u32, mode: u32 };
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] quantized activation
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/16] stacked experts
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [nE*N] stacked
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<storage, read>       idx:     array<u32>;        // [k] chosen expert indices
@group(0) @binding(6) var<storage, read>       wgt:     array<f32>;        // [k] chosen expert weights
@group(0) @binding(7) var<uniform>             dims:    Dims;
fn unpack_i8x4e(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotwe(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4e(a); let bv = unpack_i8x4e(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}
var<workgroup> parte: array<i32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let kv = dims.kp / 16u;
    let row = e * dims.n + n;          // stacked: expert e, output row n
    let bBase = row * kv;
    var acc: i32 = 0;
    for (var v: u32 = t; v < kv; v = v + 64u) {
        let a4 = aq[v]; let b4 = bq[bBase + v];
        acc = acc + dotwe(a4.x, b4.x) + dotwe(a4.y, b4.y) + dotwe(a4.z, b4.z) + dotwe(a4.w, b4.w);
    }
    parte[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { parte[t] = parte[t] + parte[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = f32(parte[0]) * aScales[0] * bScales[row];
        if (dims.mode == 1u) { dst[n] = dst[n] + wgt[dims.slot] * r; } else { dst[n] = r; }
    }
}
`

// SharedGateWGSL — Gated shared-expert combine. Qwen2-MoE adds an always-on shared expert
// scaled by a per-token sigmoid gate g = sigmoid(SharedGate·h): xd[n] += g·sharedDown[n].
// gl is the 1-element gate logit (the SharedGate GEMV result); each thread reads it and
// sigmoids inline. GLM/DeepSeek add the shared expert UNGATED (a plain residual add via
// gemvAdd), so they never reach this kernel.
const SharedGateWGSL = `
struct D { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;  // [n] running residual
@group(0) @binding(1) var<storage, read>       src: array<f32>;  // [n] shared-expert down output
@group(0) @binding(2) var<storage, read>       gl:  array<f32>;  // [1] shared-gate logit
@group(0) @binding(3) var<uniform>             d:   D;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    let g = 1.0 / (1.0 + exp(-gl[0]));
    dst[i] = dst[i] + g * src[i];
}
`

// MoeExpertGEMVW4WGSL is the int4 (W4A8) twin of MoeExpertGEMVWGSL: ONE indexed
// stacked expert's group-wise int4 weight × int8 activation. The stacking + index
// addressing (e = idx[slot], row = e*N + n) match the int8 kernel; the int4
// nibble/group-scale unpack matches GemvW4A8ShaderWGSL (group = 32 = one vec4<u32>,
// nibble value = nibble−8, f16 group scales 2-per-u32). Activations are int8 padded
// to a multiple of 32 (the same assumption the dense W4A8 GEMV makes).
const MoeExpertGEMVW4WGSL = `
struct Dims { kp: u32, n: u32, slot: u32, mode: u32 };
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;        // [nE*N, kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<storage, read>       idx:     array<u32>;        // [k] chosen expert indices
@group(0) @binding(6) var<storage, read>       wgt:     array<f32>;        // [k] chosen expert weights
@group(0) @binding(7) var<uniform>             dims:    Dims;
fn unpack_i8x4w(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
// dot8w: 8 int4 nibbles of ww (value nibble−8) · 8 int8 (alo=elems 0–3, ahi=4–7).
fn dot8w(ww: u32, alo: u32, ahi: u32) -> i32 {
    let al = unpack_i8x4w(alo);
    let ah = unpack_i8x4w(ahi);
    var s: i32 = 0;
    s = s + (i32((ww >>  0u) & 0xFu) - 8) * al.x;
    s = s + (i32((ww >>  4u) & 0xFu) - 8) * al.y;
    s = s + (i32((ww >>  8u) & 0xFu) - 8) * al.z;
    s = s + (i32((ww >> 12u) & 0xFu) - 8) * al.w;
    s = s + (i32((ww >> 16u) & 0xFu) - 8) * ah.x;
    s = s + (i32((ww >> 20u) & 0xFu) - 8) * ah.y;
    s = s + (i32((ww >> 24u) & 0xFu) - 8) * ah.z;
    s = s + (i32((ww >> 28u) & 0xFu) - 8) * ah.w;
    return s;
}
var<workgroup> partw: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let ng = dims.kp / 32u;             // groups per row
    let row = e * dims.n + n;           // stacked: expert e, output row n
    let wBase = row * ng;
    let sBase = row * ng;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < ng; v = v + 64u) {
        let w4 = bq[wBase + v];
        let a0 = aq[v * 2u];
        let a1 = aq[v * 2u + 1u];
        var idot: i32 = 0;
        idot = idot + dot8w(w4.x, a0.x, a0.y);
        idot = idot + dot8w(w4.y, a0.z, a0.w);
        idot = idot + dot8w(w4.z, a1.x, a1.y);
        idot = idot + dot8w(w4.w, a1.z, a1.w);
        let s = sBase + v;
        let pair = unpack2x16float(bScales[s >> 1u]);
        acc = acc + f32(idot) * select(pair.x, pair.y, (s & 1u) == 1u);
    }
    partw[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partw[t] = partw[t] + partw[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partw[0] * aScales[0];
        if (dims.mode == 1u) { dst[n] = dst[n] + wgt[dims.slot] * r; } else { dst[n] = r; }
    }
}
`

// MoeExpertGptOssDownGEMVW4WGSL is the int4 (W4A8) twin of MoeExpertGptOssDownGEMVWGSL. It is
// MoeExpertGEMVW4WGSL's stacked-int4 body with gpt-oss's down combine: dst += wgt·(r + dbias[row]),
// with the bias indexed by the routed EXPERT's row. Without it an int4 stack would run
// through the int8 kernel. That kernel strides rows at 2x the true width, unpacks nibble
// pairs as int8, and reads packed f16 scale pairs as f32, so every routed expert's down matmul
// would collapse to its bias.
const MoeExpertGptOssDownGEMVW4WGSL = `
struct Dims { kp: u32, n: u32, slot: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;        // [nE*N, kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<storage, read>       idx:     array<u32>;        // [k] chosen expert indices
@group(0) @binding(6) var<storage, read>       wgt:     array<f32>;        // [k] chosen expert weights
@group(0) @binding(7) var<storage, read>       dbias:   array<f32>;        // [nE*N] expert-major down bias
@group(0) @binding(8) var<uniform>             dims:    Dims;
fn unpack_i8x4w(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
// dot8w: 8 int4 nibbles of ww (value nibble−8) · 8 int8 (alo=elems 0–3, ahi=4–7).
fn dot8w(ww: u32, alo: u32, ahi: u32) -> i32 {
    let al = unpack_i8x4w(alo);
    let ah = unpack_i8x4w(ahi);
    var s: i32 = 0;
    s = s + (i32((ww >>  0u) & 0xFu) - 8) * al.x;
    s = s + (i32((ww >>  4u) & 0xFu) - 8) * al.y;
    s = s + (i32((ww >>  8u) & 0xFu) - 8) * al.z;
    s = s + (i32((ww >> 12u) & 0xFu) - 8) * al.w;
    s = s + (i32((ww >> 16u) & 0xFu) - 8) * ah.x;
    s = s + (i32((ww >> 20u) & 0xFu) - 8) * ah.y;
    s = s + (i32((ww >> 24u) & 0xFu) - 8) * ah.z;
    s = s + (i32((ww >> 28u) & 0xFu) - 8) * ah.w;
    return s;
}
var<workgroup> partw: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let ng = dims.kp / 32u;             // groups per row
    let row = e * dims.n + n;           // stacked: expert e, output row n
    let wBase = row * ng;
    let sBase = row * ng;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < ng; v = v + 64u) {
        let w4 = bq[wBase + v];
        let a0 = aq[v * 2u];
        let a1 = aq[v * 2u + 1u];
        var idot: i32 = 0;
        idot = idot + dot8w(w4.x, a0.x, a0.y);
        idot = idot + dot8w(w4.y, a0.z, a0.w);
        idot = idot + dot8w(w4.z, a1.x, a1.y);
        idot = idot + dot8w(w4.w, a1.z, a1.w);
        let s = sBase + v;
        let pair = unpack2x16float(bScales[s >> 1u]);
        acc = acc + f32(idot) * select(pair.x, pair.y, (s & 1u) == 1u);
    }
    partw[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partw[t] = partw[t] + partw[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partw[0] * aScales[0];
        dst[n] = dst[n] + wgt[dims.slot] * (r + dbias[row]);
    }
}
`

// G4RouteScaleWGSL — wgt[j] *= pes[idx[j]] over the k chosen experts, the per-expert scale on the renormalised top-k
// weights.
const G4RouteScaleWGSL = `
struct P { k: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       idx: array<u32>; // [k]
@group(0) @binding(1) var<storage, read>       pes: array<f32>; // [nE]
@group(0) @binding(2) var<storage, read_write> wgt: array<f32>; // [k]
@group(0) @binding(3) var<uniform>             p:   P;
@compute @workgroup_size(1)
fn main() {
    for (var j: u32 = 0u; j < p.k; j = j + 1u) { wgt[j] = wgt[j] * pes[idx[j]]; }
}
`

// G4TailCopyWGSL — dst[i] = src[off + i] for i < n, the up half of an expert's fused gate‖up output.
const G4TailCopyWGSL = `
struct P { n: u32, off: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { dst[i] = src[p.off + i]; }
}
`
