package wgsl

// RmsnormQuantWGSL fuses RMSNorm → activation-quantize: it reads the hidden vector
// once, computes xn = src·inv·weight(+1) in registers, finds its max-abs by tree
// reduce, and emits the packed int8 activation + scale directly — replacing the
// rms(→xn global)→quant(xn→q,s) pair with ONE dispatch and no xn round-trip.
// One workgroup (64 lanes); xn is recomputed in the quantize pass rather than
// staged, so there is no width-bounded workgroup array (works for any hidden).
const RmsnormQuantWGSL = `
struct P { h: u32, eps: f32, addone: u32, np: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // 1) sum of squares → rmsnorm inverse scale
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps);
    workgroupBarrier();
    // 2) max-abs of xn = src·inv·weight(+1), parallel tree reduce
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        mx = max(mx, abs(src[i] * inv * w));
    }
    sh[t] = mx;
    workgroupBarrier();
    stride = 32u;
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
    // 3) quantize + pack (recompute xn; one u32 word = 4 int8)
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.h) {
                var w = weight[k];
                if (p.addone == 1u) { w = w + 1.0; }
                q = i32(round(src[k] * inv * w * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// SwigluQuantWGSL fuses SwiGLU → activation-quantize: silu(gate)·up is computed, max-abs'd, and packed to int8 in ONE dispatch,
// so the inter-wide SwiGLU output never materializes in global memory or crosses a barrier. The product is recomputed in
// the pack pass (cheap arithmetic) rather than staged, so no inter-wide workgroup array is needed.
const SwigluQuantWGSL = `
struct P { n: u32, np: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;  // [n]
@group(0) @binding(1) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn silu_up(g: f32, u: f32) -> f32 { return (g / (1.0 + exp(-g))) * u; }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // 1) max-abs of mid = silu(gate)·up, parallel tree reduce
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, abs(silu_up(gate[i], up[i]))); }
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
    // 2) quantize + pack
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.n) {
                q = i32(round(silu_up(gate[k], up[k]) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// GegluQuantWGSL is swigluQuant's GELU-tanh-gated twin, Gemma's FeatGatedGELU. Same
// fused shape (product recomputed in the pack pass, no inter-wide global-memory round-trip); only the activation differs.
// gelu_tanh's argument is CLAMPED to ±15 before tanh: unclamped, it overflows f32 before saturating. See
// gegluShaderWGSL's own comment (layer.go). History: docs/code-notes/gpu.md#gegluQuant.
const GegluQuantWGSL = `
struct P { n: u32, np: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;  // [n]
@group(0) @binding(1) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn gelu_tanh(x: f32) -> f32 {
    let a = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(clamp(a, -15.0, 15.0)));
}
fn geglu_up(g: f32, u: f32) -> f32 { return gelu_tanh(g) * u; }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // 1) max-abs of mid = gelu_tanh(gate)·up, parallel tree reduce
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, abs(geglu_up(gate[i], up[i]))); }
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
    // 2) quantize + pack
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.n) {
                q = i32(round(geglu_up(gate[k], up[k]) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// LayerNormQuantWGSL fuses a bias-free LayerNorm (decoder.FeatLayerNorm: Cohere, Command-R7B) into the int8 activation
// quantize: xn = (x - mean)·inv·weight with inv = 1/sqrt(var + eps), max-abs'd and packed in one dispatch. Its bindings
// and uniform are RmsnormQuantWGSL's (p.addone is unused), so the runner binds it in that kernel's place
// (RunModel.LayerNorm). One 64-lane workgroup; the mean and the variance are tree sums, and xn is recomputed in the pack
// pass rather than staged.
const LayerNormQuantWGSL = `
struct P { h: u32, eps: f32, addone: u32, np: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn treeSum(t: u32) -> f32 {
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var acc: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { acc = acc + src[i]; }
    sh[t] = acc;
    workgroupBarrier();
    let mean = treeSum(t) / f32(p.h);
    acc = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let d = src[i] - mean; acc = acc + d * d; }
    sh[t] = acc;
    workgroupBarrier();
    let inv = 1.0 / sqrt(treeSum(t) / f32(p.h) + p.eps);
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { mx = max(mx, abs((src[i] - mean) * inv * weight[i])); }
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
            if (k < p.h) {
                q = i32(round((src[k] - mean) * inv * weight[k] * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// Relu2QuantWGSL serves the Nemotron-H squared-ReLU FFN (port from the granite SSM work). Nemotron-H's MLP block is
// NON-gated: down(relu²(up(x))) — no gate projection, unlike every SwiGLU family. relu2Quant
// is the fused activation+quant for it: relu²(up) → max-abs → int8, one dispatch (mirrors
// swigluQuant but unary). relu²(x)=max(0,x)² ≥ 0, so the int8 is non-negative ([0,127]); the
// down GEMV reads it like any int8 activation. The up/down GEMVs reuse the existing W8A8 path —
// this kernel is the only genuinely-new resident piece for Nemotron-H.
const Relu2QuantWGSL = `
struct P { n: u32, np: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(1) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(3) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn relu2(x: f32) -> f32 { let r = max(0.0, x); return r * r; }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, relu2(up[i])); }
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
                q = i32(round(relu2(up[k]) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// QkNormWGSL — Per-head QK-norm (Lever C — extend resident DecodeRunner eligibility). Qwen3 / GLM /
// Mellum RMS-normalize each query and key HEAD over headDim (with its own learned
// weight[headDim]) BEFORE RoPE — the one feature that otherwise made these archs
// residency-ineligible. This kernel does that in place: one workgroup per head, 64 lanes
// tree-reduce the sum-of-squares over headDim, then rescale. Same f32 RMS math as the
// CPU rmsNorm(qi, lw.QNorm, nH, hd, …) (decoder/attention.go), so the resident forward
// stays cosine ~1.0 vs CPU (TestResidentQKNorm_parity). addOne carries the Gemma-style
// (1+w) offset (false for Qwen3).
const QkNormWGSL = `
struct P { heads: u32, hd: u32, eps: f32, addone: u32 };
@group(0) @binding(0) var<storage, read_write> vec:    array<f32>;  // [heads*hd], normalized in place
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [hd]
@group(0) @binding(2) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x + wid.y * 65535u;
    if (h >= p.heads) { return; }
    let t = lid.x;
    let base = h * p.hd;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.hd; i = i + 64u) { let v = vec[base + i]; ss = ss + v * v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.hd) + p.eps);
    for (var i: u32 = t; i < p.hd; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        vec[base + i] = vec[base + i] * inv * w;
    }
}
`

// VNormWGSL — Gemma 4 K=V (attention_k_eq_v): V is scale-less v_norm(raw k).
// Reads raw K (src), computes RMSNorm per head with no learned weight, and writes to V (dst).
// One workgroup per head (heads = nKV).
const VNormWGSL = `
struct P { heads: u32, hd: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;  // [heads*hd], raw k
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;  // [heads*hd], v
@group(0) @binding(2) var<uniform>             p:   P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.heads) { return; }
    let t = lid.x;
    let base = h * p.hd;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.hd; i = i + 64u) {
        let v = src[base + i];
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
    let inv = 1.0 / sqrt(sh[0] / f32(p.hd) + p.eps);
    for (var i: u32 = t; i < p.hd; i = i + 64u) {
        dst[base + i] = src[base + i] * inv;
    }
}
`

// ScaleVecWGSL — Vector scaling kernel: vec[i] *= scale for i in [0, n).
const ScaleVecWGSL = `
struct P { n: u32, scale: f32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read_write> vec: array<f32>;
@group(0) @binding(1) var<uniform>             p:   P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        vec[i] = vec[i] * p.scale;
    }
}
`

const GumbelCommonWGSL = `
fn mulhi32(a: u32, b: u32) -> u32 {
  let al = a & 0xFFFFu; let ah = a >> 16u;
  let bl = b & 0xFFFFu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
  return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}

// Philox4x32-10. Must match decoder.philox4x32 (Random123 known-answer vectors).
fn philox(c0_: u32, c1_: u32, c2_: u32, c3_: u32, k0_: u32, k1_: u32) -> vec4<u32> {
  var c0 = c0_; var c1 = c1_; var c2 = c2_; var c3 = c3_;
  var k0 = k0_; var k1 = k1_;
  for (var r = 0u; r < 10u; r = r + 1u) {
    if (r > 0u) { k0 = k0 + 0x9E3779B9u; k1 = k1 + 0xBB67AE85u; }
    let hi0 = mulhi32(0xD2511F53u, c0); let lo0 = 0xD2511F53u * c0;
    let hi1 = mulhi32(0xCD9E8D57u, c2); let lo1 = 0xCD9E8D57u * c2;
    let n0 = hi1 ^ c1 ^ k0; let n2 = hi0 ^ c3 ^ k1;
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
  }
  return vec4<u32>(c0, c1, c2, c3);
}

// log(1 - w) for w in (0, 0.5): a polynomial where log's absolute error would swamp the result.
fn log1p_neg(w: f32) -> f32 {
  if (w < 0.25) {
    var p = 1.0 / 12.0;
    p = p * w + 1.0 / 11.0;
    p = p * w + 1.0 / 10.0;
    p = p * w + 1.0 / 9.0;
    p = p * w + 1.0 / 8.0;
    p = p * w + 1.0 / 7.0;
    p = p * w + 1.0 / 6.0;
    p = p * w + 1.0 / 5.0;
    p = p * w + 1.0 / 4.0;
    p = p * w + 1.0 / 3.0;
    p = p * w + 0.5;
    p = p * w + 1.0;
    return -w * p;
  }
  return log(1.0 - w);
}

fn gumbel(h: u32) -> f32 {
  var e: f32;
  if (h < 0x80000000u) {
    let w = (f32(h) + 0.5) * 2.3283064365386963e-10;
    e = -log1p_neg(w);
  } else {
    let v = (f32(~h) + 0.5) * 2.3283064365386963e-10; // = 1 - w, exactly
    e = -log(v);
  }
  return -log(e);
}

fn better(ak: f32, ai: i32, bk: f32, bi: i32) -> bool {
  if (ai < 0) { return false; }
  if (bi < 0) { return true; }
  return ak > bk || (ak == bk && ai < bi);
}

var<workgroup> sk: array<f32, 256>;
var<workgroup> si: array<i32, 256>;
`

const GumbelStage1WGSL = GumbelCommonWGSL + `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> prm: array<u32>;
@group(0) @binding(2) var<storage, read_write> bkey: array<f32>;
@group(0) @binding(3) var<storage, read_write> bidx: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let b = wid.x * 256u + t; // group of 4 consecutive tokens
  let v = prm[0];
  let invT = bitcast<f32>(prm[1]);
  var best = -3.4028234e38;
  var bi: i32 = -1;
  if (b * 4u < v) {
    let r = philox(b, prm[4], prm[5], 0u, prm[2], prm[3]);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
      let i = b * 4u + lane;
      if (i < v) {
        let k = logits[i] * invT + gumbel(r[lane]);
        if (k > best) { best = k; bi = i32(i); }
      }
    }
  }
  sk[t] = best; si[t] = bi;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (t < o && better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
    workgroupBarrier();
  }
  if (t == 0u) { bkey[wid.x] = sk[0]; bidx[wid.x] = si[0]; }
}
`

const GumbelStage2WGSL = GumbelCommonWGSL + `
@group(0) @binding(0) var<storage, read> prm: array<u32>;
@group(0) @binding(1) var<storage, read> bkey: array<f32>;
@group(0) @binding(2) var<storage, read> bidx: array<i32>;
@group(0) @binding(3) var<storage, read_write> outv: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let n = prm[6];
  var best = -3.4028234e38;
  var bi: i32 = -1;
  for (var i = t; i < n; i = i + 256u) {
    if (better(bkey[i], bidx[i], best, bi)) { best = bkey[i]; bi = bidx[i]; }
  }
  sk[t] = best; si[t] = bi;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (t < o && better(sk[t + o], si[t + o], sk[t], si[t])) { sk[t] = sk[t + o]; si[t] = si[t + o]; }
    workgroupBarrier();
  }
  if (t == 0u) { outv[0] = si[0]; } // -1 when nothing was comparable: the caller falls back to the host argmax
}
`

const LoraDeltaDownWGSL = `
struct PDown { k: u32, r: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       aq: array<vec4<u32>>;   // [kp/16] int8 act (kp >= k)
@group(0) @binding(1) var<storage, read>       ascale: array<f32>;     // [1]
@group(0) @binding(2) var<storage, read>       amat: array<f32>;       // [r, k] row-major
@group(0) @binding(3) var<storage, read_write> tout: array<f32>;       // [r]
@group(0) @binding(4) var<uniform>             p: PDown;

fn get_aq_i8(k: u32) -> i32 {
    let v = aq[k >> 4u];
    var word: u32;
    let wi = (k >> 2u) & 3u;
    if (wi == 0u) { word = v.x; } else if (wi == 1u) { word = v.y; } else if (wi == 2u) { word = v.z; } else { word = v.w; }
    let shifted = word << (24u - (k & 3u) * 8u);
    return i32(shifted) >> 24u;
}

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let sc = ascale[0];
    for (var r: u32 = 0u; r < p.r; r = r + 1u) {
        let base = r * p.k;
        var part: f32 = 0.0;
        for (var k: u32 = t; k < p.k; k = k + 64u) {
            part = part + amat[base + k] * (f32(get_aq_i8(k)) * sc);
        }
        red[t] = part;
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) { tout[r] = red[0]; }
        workgroupBarrier();
    }
}
`

const LoraDeltaUpWGSL = `
struct PUp { r: u32, outn: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       bmat: array<f32>;  // [outn, r] row-major
@group(0) @binding(1) var<storage, read>       tin:  array<f32>;  // [r]
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;  // [outn], accumulated into
@group(0) @binding(3) var<uniform>             p:    PUp;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.outn) { return; }
    let base = row * p.r;
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < p.r; r = r + 1u) {
        acc = acc + bmat[base + r] * tin[r];
    }
    dst[row] = dst[row] + p.scale * acc;
}
`
