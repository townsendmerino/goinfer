package wgsl

// RopeShaderWGSL — RoPE: rotate the (d, half+d) pair of each head by pos·invFreq[d], scaled. One
// thread per (head, d) pair. vec is q or k in place; invFreq is the layer's
// precomputed inverse frequencies [half].
const RopeShaderWGSL = RopeCosSinWGSL + `
struct P { heads: u32, headDim: u32, half: u32, pos: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> vec:     array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<uniform>             p:       P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.heads * p.half) { return; }
    let h = idx / p.half;
    let d = idx % p.half;
    let theta = f32(p.pos) * invFreq[d];
    let cs = ropeCosSin(theta); let c = cs.x * p.scale;
    let s = cs.y * p.scale;
    let off = h * p.headDim;
    let x1 = vec[off + d];
    let x2 = vec[off + p.half + d];
    vec[off + d]        = x1 * c - x2 * s;
    vec[off + p.half + d] = x2 * c + x1 * s;
}
`

// AttnShaderWGSL — Single-query attention (decode): one workgroup per query head, an online (FlashAttention-style) softmax over keys
// [start, nKeys), so it is numerically stable and needs no scratch for the full score row. GQA maps kvh = qh/group.
// workgroup_size = WG (one lane per dim, hd ≤ WG): each key's score is a workgroup tree-reduce and the value accumulate is
// one lane per dim. acc is per-dim (per lane); m and l are replicated identically across lanes (every lane sees the same
// score), so no extra reduction is needed. History: docs/code-notes/gpu.md#AttnShaderWGSL.
const AttnShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
// HS is a genuinely PER-LAYER uniform, separate from P: P's uniform buffer is shared across every
// layer with the same {hd,nKV,half,kEqV} geometry tuple (geomFor's dedup cache), but hasSink is a
// per-LAYER property that can legitimately differ between two layers sharing a geometry — so it
// cannot be baked into P without risking one layer's flag leaking onto another's cached buffer.
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       q:     array<f32>;  // [nH*hd]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       keys:  array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(2) var<storage, read>       vals:  array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;  // [nH*hd]
@group(0) @binding(4) var<storage, read>       sinks: array<f32>;  // [nH]; real but unused when hasSink==0
@group(0) @binding(5) var<uniform>             p:     P;
@group(0) @binding(6) var<uniform>             hs:    HS;
var<workgroup> red: array<f32, 128>;  // per-key dot-product reduction
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }
    var acc: f32 = 0.0;
    // gpt-oss's learned per-head attention sink (FeatAttnSink):
    // an extra logit with NO key and NO value, joining the softmax MAX and DENOMINATOR only —
    // seeding the online-softmax state with an imaginary zero-value key whose score is the sink
    // is exactly equivalent (m=sink, l=1 ⇒ the first real key's mnew=max(sink,x) folds it into
    // the max, and its corr/pe terms fold it into the denominator; acc gets no contribution since
    // the imaginary key's value is zero). hasSink==0 (every other family) is m=-1e30, l=0 — no-op.
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u);
    var l: f32 = select(0.0, 1.0, hs.v == 1u);
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
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
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        if (lane) { acc = acc * corr + pe * vals[kbase + d]; }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();  // all lanes done reading red[0] before next key overwrites red[d]
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// AttnKeysShaderWGSL is decode attention that splits the workgroup over KEYS, not over the head dimension. Same math as
// AttnShaderWGSL, different decomposition, and the decomposition is the point.
//
// AttnShaderWGSL puts one lane on each of the hd dimensions, so the q·k dot for every key is a cross-lane reduction with nine
// workgroupBarrier()s per key: a latency wall, not a bandwidth one. Here each lane owns a disjoint set of keys and computes
// its own dots with no cross-lane traffic; only the softmax max and denominator reduce, once per tile rather than once per
// key. cuda/attn_block.cu uses the same shape.
//
// TILED, because WGSL has no dynamic workgroup storage: a var<workgroup> is fixed at compile time, so scores are processed in
// attnKeysTile-key tiles with the online-softmax state (m, l, acc) carried across them. Any context length works and storage
// stays inside WebGPU's guaranteed 16 KB.
//
// The vec4 K/q loads are load-bearing, not a micro-optimization: splitting over keys makes the K read stride kvDim across the
// warp, and without vec4 loads that pattern wastes most of each L1TEX sector (cuda/attn_block.cu reads float4 for the same
// reason). That is why attnKeysEligible requires hd%4==0 and kvDim%4==0.
//
// NOT bit-identical to AttnShaderWGSL: the denominator sums in a different order and the tiled rescale reassociates.
// Attention was never bit-exact (f32 against the CPU oracle's f64), so the standing gate is TestAttention_parity's
// cosine/maxAbs against that reference, plus argmax through TestWebGPU_forwardParity. Figures and history:
// docs/code-notes/gpu.md#AttnKeysShaderWGSL.
const AttnKeysShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };  // per-layer (see AttnShaderWGSL's own comment)
@group(0) @binding(0) var<storage, read>       q4:    array<vec4<f32>>;  // [nH*hd/4]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       k4:    array<vec4<f32>>;  // [nKeys*kvDim/4]
@group(0) @binding(2) var<storage, read>       vals:  array<f32>;        // [nKeys*kvDim]
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;        // [nH*hd]
@group(0) @binding(4) var<storage, read>       sinks: array<f32>;        // [nH]; real but unused when hasSink==0
@group(0) @binding(5) var<uniform>             p:     P;
@group(0) @binding(6) var<uniform>             hs:    HS;

var<workgroup> sc:  array<f32, 2048>;  // one tile of scores; TILE below must match
var<workgroup> red: array<f32, 128>;   // the two per-tile reductions

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }   // uniform per workgroup ⇒ the barriers below stay uniform
    let t = lid.x;
    let hd = p.hd;
    let hd4 = hd / 4u;
    let kvDim = p.nKV * hd;
    let kvDim4 = kvDim / 4u;
    let kvh = qh / p.group;
    let qb4 = (qh * hd) / 4u;
    let kvb4 = (kvh * hd) / 4u;
    let kvbase = kvh * hd;
    let TILE: u32 = 2048u;

    // gpt-oss's attention sink (FeatAttnSink): seeding m/l with the sink/1 before the tile loop
    // starts is equivalent to an imaginary zero-value key at the sink's score — see
    // AttnShaderWGSL's own comment for the full argument. Correct here specifically because m/l
    // persist ACROSS tiles exactly like the per-key kernels do; the first tile's
    // mnew=max(m,red[0]) folds the sink into the max before any real key's exp is computed.
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u);   // running softmax max
    var l: f32 = select(0.0, 1.0, hs.v == 1u);           // running denominator
    var acc: f32 = 0.0;   // lane t's running weighted V-sum for dim t (t < hd)

    var tileStart: u32 = p.start;
    loop {
        if (tileStart >= p.nKeys) { break; }
        var tileEnd: u32 = tileStart + TILE;
        if (tileEnd > p.nKeys) { tileEnd = p.nKeys; }

        // pass 1 — scores. Lanes split over KEYS: each lane's dot is entirely its own.
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
        let mnew = max(m, red[0]);
        let corr = exp(m - mnew);
        workgroupBarrier();   // every lane has read red[0] before pass 2 overwrites red[t]

        // pass 2 — exponentiate in place + partial denominator, still split over KEYS.
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

        // pass 3 — V-sum. Lanes switch to owning DIMS, which is what makes the V read
        // coalesced across the warp (adjacent lanes read adjacent floats of one row).
        if (t < hd) {
            var a: f32 = acc * corr;
            for (var s: u32 = tileStart; s < tileEnd; s = s + 1u) {
                a = a + sc[s - tileStart] * vals[s * kvDim + kvbase + t];
            }
            acc = a;
        }
        m = mnew;
        workgroupBarrier();   // sc and red are reused by the next tile
        tileStart = tileEnd;
    }
    if (t < hd) { ctx[qh * hd + t] = acc / l; }
}
`

// RopeStoreShaderWGSL — ropeStore: like rope, but reads the q/k-projection output from a separate src
// buffer and writes the rotated result into dst (the KV cache) at element offset
// p.base = pos*kvDim. Used for K so the decode token never needs a
// CopyBufferToBuffer to append the cache — the rotate IS the append. One thread
// per (head, d) pair. p.base is rewritten per token via the runner's posUni.
const RopeStoreShaderWGSL = RopeCosSinWGSL + `
struct P { heads: u32, headDim: u32, half: u32, pos: u32, scale: f32, base: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src:     array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<storage, read_write> dst:     array<f32>;  // KV cache [maxLen*kvDim]
@group(0) @binding(3) var<uniform>             p:       P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let tail = p.headDim - 2u * p.half; // pass-through width (0 for full rotary)
    if (idx >= p.heads * (p.half + tail)) { return; } // heads*(headDim-half)
    if (idx < p.heads * p.half) {
        let h = idx / p.half;
        let d = idx % p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        let off = h * p.headDim;
        let x1 = src[off + d];
        let x2 = src[off + p.half + d];
        dst[p.base + off + d]          = x1 * c - x2 * s;
        dst[p.base + off + p.half + d] = x2 * c + x1 * s;
    } else {
        // partial rotary: store the un-rotated tail [2*half, headDim) — the CPU ref stores the
        // full k, and attention reads zeros for these dims otherwise (C4).
        let t = idx - p.heads * p.half;
        let h = t / tail;
        let e = t % tail;
        let off = h * p.headDim + 2u * p.half + e;
        dst[p.base + off] = src[off];
    }
}
`

// KvStoreShaderWGSL — kvStore: copy src [n] into dst (the V cache) at element offset p.base=pos*kvDim.
// A compute-pass-safe replacement for the V CopyBufferToBuffer append, so the
// decode token stays a single compute pass. p.base is rewritten per token.
const KvStoreShaderWGSL = `
struct P { n: u32, base: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    dst[p.base + i] = src[i];
}
`

// QkvFinalizeShaderWGSL — qkvFinalize fuses the three post-projection KV ops of the decode attention block —
// rope(q) in place, rope(k)+store into KCache, store v into VCache — into ONE dispatch,
// cutting two dispatches (and their barriers) per layer off the serially-dependent decode
// chain. Math is bit-identical to the separate rope / ropeStore / kvStore f32 kernels
// (same rotate_half, same base = pos*kvDim). f32-KV only; the f16/int8-KV caches keep
// their own packed-store kernels. The three index ranges differ (q: nH·half, k: nKV·half,
// v: kvDim), so each thread does whichever apply to its global index — independent writes,
// no barrier needed between them.
const QkvFinalizeShaderWGSL = RopeCosSinWGSL + `
struct P { nH: u32, nKV: u32, hd: u32, half: u32, pos: u32, base: u32, scale: f32, kvDim: u32 };
@group(0) @binding(0) var<storage, read_write> q:       array<f32>;  // [nH*hd] rotated in place
@group(0) @binding(1) var<storage, read>       k:       array<f32>;  // [nKV*hd]
@group(0) @binding(2) var<storage, read>       v:       array<f32>;  // [nKV*hd]
@group(0) @binding(3) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(4) var<storage, read_write> kCache:  array<f32>;  // rotated K at base
@group(0) @binding(5) var<storage, read_write> vCache:  array<f32>;  // V at base
@group(0) @binding(6) var<uniform>             p:       P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // q rope (nH heads), in place.
    if (i < p.nH * p.half) {
        let h = i / p.half; let d = i % p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale; let s = cs.y * p.scale;
        let off = h * p.hd;
        let x1 = q[off + d]; let x2 = q[off + p.half + d];
        q[off + d]          = x1 * c - x2 * s;
        q[off + p.half + d] = x2 * c + x1 * s;
    }
    // k rope (nKV heads) + store into KCache at base.
    if (i < p.nKV * p.half) {
        let h = i / p.half; let d = i % p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale; let s = cs.y * p.scale;
        let off = h * p.hd;
        let x1 = k[off + d]; let x2 = k[off + p.half + d];
        kCache[p.base + off + d]          = x1 * c - x2 * s;
        kCache[p.base + off + p.half + d] = x2 * c + x1 * s;
    }
    // k pass-through tail [2*half, hd): partial rotary (2*half < hd — GLM, some Phi) leaves these
    // key dims un-rotated, and the CPU reference stores the FULL k. Store them here or attention
    // reads zeros for the tail at every cached position — plausible-looking, silently wrong logits
    // (C4). ktail=0 for full rotary ⇒ no-op. These threads are within the kvDim range the v-store
    // already dispatches (nKV*(hd-2*half) < nKV*hd = kvDim), so no extra dispatch is needed.
    let ktail = p.hd - 2u * p.half;
    if (i < p.nKV * ktail) {
        let h = i / ktail; let t = i % ktail;
        let off = h * p.hd + 2u * p.half + t;
        kCache[p.base + off] = k[off];
    }
    // v store (kvDim elements) into VCache at base.
    if (i < p.kvDim) {
        vCache[p.base + i] = v[i];
    }
}
`

// AttnF16ShaderWGSL — attnF16: AttnShaderWGSL with the K/V cache as packed f16. Identical online-softmax
// math; the only change is unpacking each cached K/V element to f32 before use.
const AttnF16ShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };  // per-layer (see AttnShaderWGSL's own comment)
@group(0) @binding(0) var<storage, read>       q:     array<f32>;  // [nH*hd]  (RoPE'd, f32)
@group(0) @binding(1) var<storage, read>       keys:  array<u32>;  // [nKeys*nKV*hd] f16-packed
@group(0) @binding(2) var<storage, read>       vals:  array<u32>;  // [nKeys*nKV*hd] f16-packed
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;  // [nH*hd]
@group(0) @binding(4) var<storage, read>       sinks: array<f32>;  // [nH]; real but unused when hasSink==0
@group(0) @binding(5) var<uniform>             p:     P;
@group(0) @binding(6) var<uniform>             hs:    HS;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }
    var acc: f32 = 0.0;
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u); // gpt-oss attention sink; see AttnShaderWGSL
    var l: f32 = select(0.0, 1.0, hs.v == 1u);
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
        let ki = s * kvDim + kvbase + d;  // logical element index of this lane's K/V
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
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        if (lane) {
            let vpair = unpack2x16float(vals[ki >> 1u]);
            acc = acc * corr + pe * select(vpair.x, vpair.y, (ki & 1u) == 1u);
        }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// RopeStoreF16ShaderWGSL — ropeStoreF16: rotate K and store as f16. One thread per cache WORD (kvDim/2
// threads — same count as the f32 ropeStore's nKV*half), owning the two logical
// elements 2*idx, 2*idx+1. Each is rotated independently (its (d,half+d) RoPE
// partner is read from src) and the pair is pack2x16float'd into one word.
const RopeStoreF16ShaderWGSL = RopeCosSinWGSL + `
struct P { heads: u32, headDim: u32, half: u32, pos: u32, scale: f32, base: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src:     array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<storage, read_write> dst:     array<u32>;  // KV cache f16-packed
@group(0) @binding(3) var<uniform>             p:       P;
fn rotated(e: u32) -> f32 {
    let h = e / p.headDim;
    let dd = e - h * p.headDim;
    let off = h * p.headDim;
    if (dd < p.half) {
        let theta = f32(p.pos) * invFreq[dd];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        return src[off + dd] * c - src[off + p.half + dd] * s;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        return src[off + p.half + d] * c + src[off + d] * s;
    }
    return src[off + dd]; // pass-through tail [2*half, headDim): partial rotary, un-rotated (C4)
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;                              // word within the token
    if (idx >= p.heads * (p.headDim / 2u)) { return; } // kvDim/2 words (covers rotated + tail, C4)
    let e0 = 2u * idx;
    dst[(p.base >> 1u) + idx] = pack2x16float(vec2<f32>(rotated(e0), rotated(e0 + 1u)));
}
`

// KvStoreF16ShaderWGSL — kvStoreF16: store V as f16. One thread per word, packing the two adjacent
// V elements 2*idx, 2*idx+1. base (pos*kvDim) is even so words align.
const KvStoreF16ShaderWGSL = `
struct P { n: u32, base: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;  // V cache f16-packed
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (2u * idx >= p.n) { return; }
    let e0 = 2u * idx;
    dst[(p.base >> 1u) + idx] = pack2x16float(vec2<f32>(src[e0], src[e0 + 1u]));
}
`

// AttnI8ShaderWGSL — attnI8: AttnShaderWGSL reading int8 K/V: unpack each element (sign-extend the byte) and scale by the per-(position,head)
// f32 scale. q stays f32 and the dot is f32; AttnI8DP4AShaderWGSL is the dot4I8Packed variant.
const AttnI8ShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };  // per-layer (see AttnShaderWGSL's own comment)
@group(0) @binding(0) var<storage, read>       q:      array<f32>;  // [nH*hd] (RoPE'd, f32)
@group(0) @binding(1) var<storage, read>       keys:   array<u32>;  // [nKeys*nKV*hd] int8-packed (4/word)
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;  // [nKeys*nKV*hd] int8-packed
@group(0) @binding(3) var<storage, read>       kScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(4) var<storage, read>       vScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(5) var<storage, read_write> ctx:    array<f32>;  // [nH*hd]
@group(0) @binding(6) var<storage, read>       sinks:  array<f32>;  // [nH]; real but unused when hasSink==0
@group(0) @binding(7) var<uniform>             p:      P;
@group(0) @binding(8) var<uniform>             hs:     HS;
var<workgroup> red: array<f32, 128>;
fn unpacki8(w: u32, e: u32) -> f32 {
    let b = (e & 3u) * 8u;
    return f32(i32(w << (24u - b)) >> 24u);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }
    var acc: f32 = 0.0;
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u); // gpt-oss attention sink; see AttnShaderWGSL
    var l: f32 = select(0.0, 1.0, hs.v == 1u);
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
        let ki = s * kvDim + kvbase + d;
        let sci = s * p.nKV + kvh;
        var prod: f32 = 0.0;
        if (lane) { prod = qd * unpacki8(keys[ki >> 2u], ki) * kScale[sci]; }
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
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        if (lane) { acc = acc * corr + pe * unpacki8(vals[ki >> 2u], ki) * vScale[sci]; }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// AttnI8DP4AShaderWGSL — hardware-accelerated dot4I8Packed variant of AttnI8ShaderWGSL.
// The query vector is dynamically quantized to int8 once per head, packing 4 int8s per
// word into workgroup shared memory. Each key dot product is computed using dot4I8Packed
// over (hd/4) words in parallel, cutting key read bandwidth by 4x and reducing reduction
// barriers from 7 to 5 steps per position.
const AttnI8DP4AShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       q:      array<f32>;  // [nH*hd] (RoPE'd, f32)
@group(0) @binding(1) var<storage, read>       keys:   array<u32>;  // [nKeys*nKV*hd] int8-packed (4/word)
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;  // [nKeys*nKV*hd] int8-packed
@group(0) @binding(3) var<storage, read>       kScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(4) var<storage, read>       vScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(5) var<storage, read_write> ctx:    array<f32>;  // [nH*hd]
@group(0) @binding(6) var<storage, read>       sinks:  array<f32>;  // [nH]; real but unused when hasSink==0
@group(0) @binding(7) var<uniform>             p:      P;
@group(0) @binding(8) var<uniform>             hs:     HS;
var<workgroup> red: array<f32, 128>;
var<workgroup> sh_qw: array<u32, 32>;
var<workgroup> sh_qs: f32;

fn unpacki8(w: u32, e: u32) -> f32 {
    let b = (e & 3u) * 8u;
    return f32(i32(w << (24u - b)) >> 24u);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let d = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    let lane = d < hd;
    var qd: f32 = 0.0;
    if (lane) { qd = q[qbase + d]; }

    red[d] = abs(qd);
    workgroupBarrier();
    var qstride: u32 = 64u;
    loop {
        if (qstride == 0u) { break; }
        if (d < qstride) { red[d] = max(red[d], red[d + qstride]); }
        workgroupBarrier();
        qstride = qstride / 2u;
    }
    if (d == 0u) {
        var amax = red[0];
        if (amax == 0.0) { amax = 1.0; }
        sh_qs = amax / 127.0;
    }
    workgroupBarrier();
    let qs = sh_qs;
    var qi: i32 = 0;
    if (lane) {
        qi = clamp(i32(round(qd / qs)), -127, 127);
    }
    red[d] = f32(qi);
    workgroupBarrier();
    let numWords = hd >> 2u;
    if (d < numWords) {
        let b0 = u32(i32(red[d * 4u + 0u]) & 0xff);
        let b1 = u32(i32(red[d * 4u + 1u]) & 0xff);
        let b2 = u32(i32(red[d * 4u + 2u]) & 0xff);
        let b3 = u32(i32(red[d * 4u + 3u]) & 0xff);
        sh_qw[d] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    }
    workgroupBarrier();

    let q_scale_comb = qs * p.scale;
    var acc: f32 = 0.0;
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u);
    var l: f32 = select(0.0, 1.0, hs.v == 1u);
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
        let kw_base = (s * kvDim + kvbase) >> 2u;
        let sci = s * p.nKV + kvh;
        var idot: i32 = 0;
        if (d < numWords) {
            idot = dot4I8Packed(sh_qw[d], keys[kw_base + d]);
        }
        red[d] = f32(idot);
        workgroupBarrier();
        var stride: u32 = 16u;
        loop {
            if (stride == 0u) { break; }
            if (d < stride) { red[d] = red[d] + red[d + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * (q_scale_comb * kScale[sci]);
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        let ki = s * kvDim + kvbase + d;
        if (lane) { acc = acc * corr + pe * unpacki8(vals[ki >> 2u], ki) * vScale[sci]; }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();
    }
    if (lane) { ctx[qbase + d] = acc / l; }
}
`

// RopeStoreI8ShaderWGSL — ropeStoreI8: rotate K, per-head absmax → int8. One thread per KV head; two
// passes over headDim (recompute the cheap rope rather than spill a local array).
const RopeStoreI8ShaderWGSL = RopeCosSinWGSL + `
struct P { heads: u32, headDim: u32, half: u32, pos: u32, scale: f32, base: u32, nKV: u32, spos: u32 };
@group(0) @binding(0) var<storage, read>       src:     array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<storage, read_write> dst:     array<u32>;  // K cache int8-packed
@group(0) @binding(3) var<storage, read_write> scales:  array<f32>;  // [maxLen*nKV]
@group(0) @binding(4) var<uniform>             p:       P;
fn rot(off: u32, dd: u32) -> f32 {
    if (dd < p.half) {
        let theta = f32(p.pos) * invFreq[dd];
        let cs = ropeCosSin(theta);
        return src[off + dd] * cs.x * p.scale - src[off + p.half + dd] * cs.y * p.scale;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta);
        return src[off + p.half + d] * cs.x * p.scale + src[off + d] * cs.y * p.scale;
    }
    return src[off + dd]; // pass-through tail [2*half, headDim): partial rotary, un-rotated (C4)
}
fn qb(x: f32, inv: f32) -> u32 { return u32(i32(clamp(round(x * inv), -127.0, 127.0))) & 0xffu; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.heads) { return; }
    let off = h * p.headDim;
    var amax: f32 = 0.0;
    for (var d: u32 = 0u; d < p.headDim; d = d + 1u) { amax = max(amax, abs(rot(off, d))); }
    var sc: f32 = amax / 127.0;
    if (sc == 0.0) { sc = 1.0; }
    scales[p.spos * p.nKV + h] = sc; // the TRUE position; p.pos is the rope angle's (audit C-09)
    let inv = 1.0 / sc;
    let wbase = (p.base + off) / 4u;
    for (var d: u32 = 0u; d < p.headDim; d = d + 4u) {
        dst[wbase + d / 4u] = qb(rot(off, d), inv) | (qb(rot(off, d+1u), inv) << 8u) | (qb(rot(off, d+2u), inv) << 16u) | (qb(rot(off, d+3u), inv) << 24u);
    }
}
`

// KvStoreI8ShaderWGSL — kvStoreI8: store V as int8, per-head absmax → scale. One thread per KV head.
const KvStoreI8ShaderWGSL = `
struct P { heads: u32, headDim: u32, base: u32, pos: u32, nKV: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read_write> dst:    array<u32>;  // V cache int8-packed
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;  // [maxLen*nKV]
@group(0) @binding(3) var<uniform>             p:      P;
fn qb(x: f32, inv: f32) -> u32 { return u32(i32(clamp(round(x * inv), -127.0, 127.0))) & 0xffu; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.heads) { return; }
    let off = h * p.headDim;
    var amax: f32 = 0.0;
    for (var d: u32 = 0u; d < p.headDim; d = d + 1u) { amax = max(amax, abs(src[off + d])); }
    var sc: f32 = amax / 127.0;
    if (sc == 0.0) { sc = 1.0; }
    scales[p.pos * p.nKV + h] = sc;
    let inv = 1.0 / sc;
    let wbase = (p.base + off) / 4u;
    for (var d: u32 = 0u; d < p.headDim; d = d + 4u) {
        dst[wbase + d / 4u] = qb(src[off+d], inv) | (qb(src[off+d+1u], inv) << 8u) | (qb(src[off+d+2u], inv) << 16u) | (qb(src[off+d+3u], inv) << 24u);
    }
}
`

// AttnWideTemplateWGSL — The WIDE single-query attention kernel: one template, STRIDED lanes.
//
// The narrow kernels put one lane on each head dim, which makes the workgroup width a hard ceiling on head_dim; 256, the
// guaranteed maxComputeInvocationsPerWorkgroup, is as far as that idea goes. The wide variant gives each lane a STRIDE of
// dims (d0, d0+WG, d0+2·WG, …), so any head_dim up to attnWGWide·attnMaxPerLane works with a fixed workgroup. The narrow
// kernels still serve head_dim ≤ attnWG, because striding costs a per-lane array and two extra loops.
//
// One template rather than three near-identical copies: the algorithm (online softmax, tree-reduce, rescale) is the part that
// is easy to get subtly wrong, and three copies are three places for a fix to land in two. The three variants differ ONLY in
// how a K/V element is fetched. History: docs/code-notes/gpu.md#AttnWideTemplateWGSL.
const AttnWideTemplateWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
struct HS { v: u32, _a: u32, _b: u32, _c: u32 };  // per-layer (see AttnShaderWGSL's own comment)
@group(0) @binding(0) var<storage, read> q: array<f32>;  // [nH*hd] (RoPE'd, f32)
__BINDINGS__
__HELPERS__
var<workgroup> red: array<f32, __WG__>;
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let d0 = lid.x;
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    // This lane owns dims d0, d0+WG, … — nper of them. A lane past hd owns none and simply
    // contributes 0 to every reduction, which keeps the tree-reduce below unconditional.
    let nper = (hd + __WG__u - 1u) / __WG__u;
    var qd:  array<f32, __MAXPER__>;
    var acc: array<f32, __MAXPER__>;
    for (var j = 0u; j < nper; j = j + 1u) {
        qd[j] = 0.0;
        acc[j] = 0.0;
        let d = d0 + j * __WG__u;
        if (d < hd) { qd[j] = q[qbase + d]; }
    }
    var m: f32 = select(-1e30, sinks[qh], hs.v == 1u); // gpt-oss attention sink; see AttnShaderWGSL
    var l: f32 = select(0.0, 1.0, hs.v == 1u);
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
        let kbase = s * kvDim + kvbase;
        let sci = s * p.nKV + kvh;
        var prod: f32 = 0.0;
        for (var j = 0u; j < nper; j = j + 1u) {
            let d = d0 + j * __WG__u;
            if (d < hd) {
                let ki = kbase + d;
                prod = prod + qd[j] * (__KREAD__);
            }
        }
        red[d0] = prod;
        workgroupBarrier();
        var stride: u32 = __HALFWG__u;
        loop {
            if (stride == 0u) { break; }
            if (d0 < stride) { red[d0] = red[d0] + red[d0 + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        for (var j = 0u; j < nper; j = j + 1u) {
            let d = d0 + j * __WG__u;
            if (d < hd) {
                let ki = kbase + d;
                acc[j] = acc[j] * corr + pe * (__VREAD__);
            }
        }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();  // all lanes done reading red[0] before the next key overwrites red[d0]
    }
    for (var j = 0u; j < nper; j = j + 1u) {
        let d = d0 + j * __WG__u;
        if (d < hd) { ctx[qbase + d] = acc[j] / l; }
    }
}
`
