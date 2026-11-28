package wgsl

// RmsnormBatchedShaderWGSL is rmsnormShaderWGSL (layer.go) widened to M rows in one
// dispatch: grid (1, M) instead of (1, 1), each workgroup owns row wid.y and reduces
// only that row (workgroupBarrier never crosses rows), so the per-row math and
// reduction order are identical to calling the M=1 kernel M times — bit-identical
// by construction, gated by TestRMSNormBatched_parity. The pass uses it for a
// sublayer's float norm (Gemma's sandwich norms), where decode runs rmsnormShaderWGSL.
const RmsnormBatchedShaderWGSL = `
struct P { h: u32, eps: f32, addone: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [M, h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;  // [M, h]
@group(0) @binding(3) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let base = wid.y * p.h;
    var s: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[base + i]; s = s + v*v; }
    sh[t] = s;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps);
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        dst[base + i] = src[base + i] * inv * w;
    }
}
`

// RopeBatchedShaderWGSL is RopeShaderWGSL (attention.go) widened to M rows in one dispatch, grid (ceil(heads*half/64), M).
// Positions are contiguous within one PrefillLastW8A8 call, so row r's position is p.start+row. Same per-(row, head, d)
// math as the M=1 kernel, so bit-identical (TestRoPEBatched_parity).
const RopeBatchedShaderWGSL = RopeCosSinWGSL + `
struct P { heads: u32, headDim: u32, half: u32, start: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> vec:     array<f32>;  // [M, heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<uniform>             p:       P;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row = gid.y;
    if (idx >= p.heads * p.half) { return; }
    let h = idx / p.half;
    let d = idx % p.half;
    let pos = p.start + row;
    let theta = f32(pos) * invFreq[d];
    let cs = ropeCosSin(theta); let c = cs.x * p.scale;
    let s = cs.y * p.scale;
    let base = row * p.heads * p.headDim;
    let off = base + h * p.headDim;
    let x1 = vec[off + d];
    let x2 = vec[off + p.half + d];
    vec[off + d]           = x1 * c - x2 * s;
    vec[off + p.half + d]  = x2 * c + x1 * s;
}
`

// AttnBatchedShaderWGSL is AttnShaderWGSL (attention.go) widened to M query rows in one dispatch, grid (nH, M); row r's
// causal bound is basePos+r+1 (positions are contiguous within one PrefillLastW8A8 call). No sink support:
// runModelToModelW declines FeatAttnSink. Not claimed bit-identical to M calls of AttnShaderWGSL: gated on
// cosine/maxAbs like every attention kernel pair here.
const AttnBatchedShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, basePos: u32, group: u32, scale: f32, m: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:     array<f32>;  // [M, nH*hd]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       keys:  array<f32>;  // [nKeysCache*nKV*hd]
@group(0) @binding(2) var<storage, read>       vals:  array<f32>;  // [nKeysCache*nKV*hd]
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;  // [M, nH*hd]
@group(0) @binding(4) var<uniform>             p:     P;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    let row = wid.y;
    if (qh >= p.nH || row >= p.m) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = row * p.nH * hd + qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }
    var acc: f32 = 0.0;
    var mx: f32 = -1e30;
    var l: f32 = 0.0;
    let nKeys = p.basePos + row + 1u;
    for (var s: u32 = 0u; s < nKeys; s = s + 1u) {
        let kbase = s * kvDim + kvbase;
        var prod: f32 = 0.0;
        if (lane) { prod = qd * keys[kbase + d]; }
        red[d] = prod;
        workgroupBarrier();
        var stride: u32 = 64u;
        loop {
            if (stride == 0u) { break; }
            if (d < stride) { red[d] = red[d] + red[d + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(mx, x);
        let corr = exp(mx - mnew);
        let pe = exp(x - mnew);
        if (lane) { acc = acc * corr + pe * vals[kbase + d]; }
        l = l * corr + pe;
        mx = mnew;
        workgroupBarrier();
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// AttnKeysBatchedShaderWGSL is AttnKeysShaderWGSL (attention.go) widened to M query rows the way
// AttnBatchedShaderWGSL widens the plain kernel: grid (nH, M), each row's own causal tile loop from key 0 to
// basePos+row inclusive. Selected whenever attnKeysEligible(hd, kvDim, false, false), matching attnKernel's
// preference for the tiled/key-split decomposition; most real dense architectures take this path.
const AttnKeysBatchedShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, basePos: u32, group: u32, scale: f32, m: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q4:    array<vec4<f32>>;  // [M, nH*hd/4]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       k4:    array<vec4<f32>>;  // [nKeysCache*kvDim/4]
@group(0) @binding(2) var<storage, read>       vals:  array<f32>;        // [nKeysCache*kvDim]
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;        // [M, nH*hd]
@group(0) @binding(4) var<uniform>             p:     P;

var<workgroup> sc:  array<f32, 2048>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    let row = wid.y;
    if (qh >= p.nH || row >= p.m) { return; }
    let t = lid.x;
    let hd = p.hd;
    let hd4 = hd / 4u;
    let kvDim = p.nKV * hd;
    let kvDim4 = kvDim / 4u;
    let kvh = qh / p.group;
    let qb4 = (row * p.nH * hd + qh * hd) / 4u;
    let kvb4 = (kvh * hd) / 4u;
    let kvbase = kvh * hd;
    let TILE: u32 = 2048u;

    var mVal: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: f32 = 0.0;

    let nKeys = p.basePos + row + 1u;
    var tileStart: u32 = 0u;
    loop {
        if (tileStart >= nKeys) { break; }
        var tileEnd: u32 = tileStart + TILE;
        if (tileEnd > nKeys) { tileEnd = nKeys; }

        var lm: f32 = -1e30;
        for (var s: u32 = tileStart + t; s < tileEnd; s = s + 128u) {
            let kb = s * kvDim4 + kvb4;
            var dot: f32 = 0.0;
            for (var i: u32 = 0u; i < hd4; i = i + 1u) {
                let qq = q4[qb4 + i];
                let kk = k4[kb + i];
                dot = dot + qq.x * kk.x;
                dot = dot + qq.y * kk.y;
                dot = dot + qq.z * kk.z;
                dot = dot + qq.w * kk.w;
            }
            let x = dot * p.scale;
            sc[s - tileStart] = x;
            lm = max(lm, x);
        }
        red[t] = lm;
        workgroupBarrier();
        var stride: u32 = 64u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = max(red[t], red[t + stride]); }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let mnew = max(mVal, red[0]);
        let corr = exp(mVal - mnew);
        workgroupBarrier();

        var ls: f32 = 0.0;
        for (var s: u32 = tileStart + t; s < tileEnd; s = s + 128u) {
            let e = exp(sc[s - tileStart] - mnew);
            sc[s - tileStart] = e;
            ls = ls + e;
        }
        red[t] = ls;
        workgroupBarrier();
        stride = 64u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        l = l * corr + red[0];

        if (t < hd) {
            var a: f32 = acc * corr;
            for (var s: u32 = tileStart; s < tileEnd; s = s + 1u) {
                a = a + sc[s - tileStart] * vals[s * kvDim + kvbase + t];
            }
            acc = a;
        }
        mVal = mnew;
        workgroupBarrier();
        tileStart = tileEnd;
    }
    if (t < hd) { ctx[row * p.nH * hd + qh * hd + t] = acc / l; }
}
`

// AttnBatchedF16ShaderWGSL is AttnBatchedShaderWGSL with f16-packed keys and values:
// keys and vals are array<u32>, two f16 per word, unpacked with unpack2x16float.
const AttnBatchedF16ShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, basePos: u32, group: u32, scale: f32, m: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:     array<f32>;  // [M, nH*hd]
@group(0) @binding(1) var<storage, read>       keys:  array<u32>;  // [nKeysCache*nKV*hd/2] f16-packed
@group(0) @binding(2) var<storage, read>       vals:  array<u32>;  // [nKeysCache*nKV*hd/2] f16-packed
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;  // [M, nH*hd]
@group(0) @binding(4) var<uniform>             p:     P;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    let row = wid.y;
    if (qh >= p.nH || row >= p.m) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = row * p.nH * hd + qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }
    var acc: f32 = 0.0;
    var mx: f32 = -1e30;
    var l: f32 = 0.0;
    let nKeys = p.basePos + row + 1u;
    for (var s: u32 = 0u; s < nKeys; s = s + 1u) {
        let ki = s * kvDim + kvbase + d;
        var prod: f32 = 0.0;
        if (lane) {
            let kpair = unpack2x16float(keys[ki >> 1u]);
            prod = qd * select(kpair.x, kpair.y, (ki & 1u) == 1u);
        }
        red[d] = prod;
        workgroupBarrier();
        var stride: u32 = 64u;
        loop {
            if (stride == 0u) { break; }
            if (d < stride) { red[d] = red[d] + red[d + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(mx, x);
        let corr = exp(mx - mnew);
        let pe = exp(x - mnew);
        if (lane) {
            let vpair = unpack2x16float(vals[ki >> 1u]);
            acc = acc * corr + pe * select(vpair.x, vpair.y, (ki & 1u) == 1u);
        }
        l = l * corr + pe;
        mx = mnew;
        workgroupBarrier();
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// BiasRowsShaderWGSL adds a per-column bias to every row of a packed [M, n] buffer: decode's standalone bias add
// (residualShaderWGSL against the bias) for all rows in one dispatch, the same single addition per element.
const BiasRowsShaderWGSL = `
struct P { total: u32, n: u32, stride: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> x:    array<f32>;  // [M, n]
@group(0) @binding(1) var<storage, read>       bias: array<f32>;  // [n]
@group(0) @binding(2) var<uniform>             p:    P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.stride;
    if (gid.x >= p.stride || i >= p.total) { return; }
    x[i] = x[i] + bias[i % p.n];
}
`
