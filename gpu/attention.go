//go:build gpu

package gpu

import (
	"math"

	"github.com/cogentcore/webgpu/wgpu"
)

// Stage 3 attention primitives — RoPE and single-query attention on the GPU, so a
// decode token's full forward records into one command buffer (no CPU interleave).
// Both match the CPU (decoder.applyRoPE / attendQuery) to f32 tolerance (the CPU
// uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).

// RoPE: rotate the (d, half+d) pair of each head by pos·invFreq[d], scaled. One
// thread per (head, d) pair. vec is q or k in place; invFreq is the layer's
// precomputed inverse frequencies [half].
const ropeShaderWGSL = `
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
    let c = cos(theta) * p.scale;
    let s = sin(theta) * p.scale;
    let off = h * p.headDim;
    let x1 = vec[off + d];
    let x2 = vec[off + p.half + d];
    vec[off + d]        = x1 * c - x2 * s;
    vec[off + p.half + d] = x2 * c + x1 * s;
}
`

// attnMaxHeadDim is the largest head_dim the single-query attention kernels support: they run
// at @workgroup_size(128) with a 128-entry `red` reduction array (one lane per dim), so a
// head_dim above this would leave the tail dims un-dotted. newDecodeRunner declines any resident
// arch/layer above it (audit M-12). It MUST stay equal to the @workgroup_size(128) below and the
// `red: array<f32, 128>` widths in both single-query kernels.
const attnMaxHeadDim = 128

// Single-query attention (decode): one workgroup per query head, an online
// (FlashAttention-style) softmax over keys [start, nKeys) so it is numerically
// stable and needs no scratch for the full score row. GQA maps kvh = qh/group.
// Parallel over the head dimension: workgroup_size = WG (one lane per dim, hd ≤
// WG), so each key's score is a workgroup tree-reduce and the value accumulate is
// one lane per dim — replacing the original single-thread-per-head kernel (the §5
// finding's largest remaining glue kernel, ~5.8 ms). acc/m/l for the online
// softmax: acc is per-dim (per lane); m and l are replicated identically across
// lanes (every lane sees the same score x), so no extra reduction is needed.
const attnShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // [nH*hd]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       keys: array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(2) var<storage, read>       vals: array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;  // [nH*hd]
@group(0) @binding(4) var<uniform>             p:    P;
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
    var m: f32 = -1e30;
    var l: f32 = 0.0;
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

// ropeStore: like rope, but reads the q/k-projection output from a separate src
// buffer and writes the rotated result into dst (the KV cache) at element offset
// p.base = pos*kvDim. Used for K so the decode token never needs a
// CopyBufferToBuffer to append the cache — the rotate IS the append. One thread
// per (head, d) pair. p.base is rewritten per token via the runner's posUni.
const ropeStoreShaderWGSL = `
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
        let c = cos(theta) * p.scale;
        let s = sin(theta) * p.scale;
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

// kvStore: copy src [n] into dst (the V cache) at element offset p.base=pos*kvDim.
// A compute-pass-safe replacement for the V CopyBufferToBuffer append, so the
// decode token stays a single compute pass. p.base is rewritten per token.
const kvStoreShaderWGSL = `
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

// qkvFinalize fuses the three post-projection KV ops of the decode attention block —
// rope(q) in place, rope(k)+store into KCache, store v into VCache — into ONE dispatch,
// cutting two dispatches (and their barriers) per layer off the serially-dependent decode
// chain. Math is bit-identical to the separate rope / ropeStore / kvStore f32 kernels
// (same rotate_half, same base = pos*kvDim). f32-KV only; the f16/int8-KV caches keep
// their own packed-store kernels. The three index ranges differ (q: nH·half, k: nKV·half,
// v: kvDim), so each thread does whichever apply to its global index — independent writes,
// no barrier needed between them.
const qkvFinalizeShaderWGSL = `
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
        let c = cos(theta) * p.scale; let s = sin(theta) * p.scale;
        let off = h * p.hd;
        let x1 = q[off + d]; let x2 = q[off + p.half + d];
        q[off + d]          = x1 * c - x2 * s;
        q[off + p.half + d] = x2 * c + x1 * s;
    }
    // k rope (nKV heads) + store into KCache at base.
    if (i < p.nKV * p.half) {
        let h = i / p.half; let d = i % p.half;
        let theta = f32(p.pos) * invFreq[d];
        let c = cos(theta) * p.scale; let s = sin(theta) * p.scale;
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

// --- f16-KV variants (opt-in precision knob; see task-gpu-f16-kv.md) ---
// The KV cache is array<u32>, two f16 per word, packed/read with the core WGSL
// builtins pack2x16float / unpack2x16float — NO shader-f16 device feature, so the
// CI software adapter still compiles them. The query and the ctx output stay f32;
// only the resident cache is f16. Logical element e of a cache lives at word e>>1,
// half e&1; base (pos*kvDim) is even (kvDim even), so a token's words align and
// each store thread owns one whole word (writes both halves) — no atomics, no race.

// attnF16: attnShaderWGSL with the K/V cache as packed f16. Identical online-softmax
// math; the only change is unpacking each cached K/V element to f32 before use.
const attnF16ShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // [nH*hd]  (RoPE'd, f32)
@group(0) @binding(1) var<storage, read>       keys: array<u32>;  // [nKeys*nKV*hd] f16-packed
@group(0) @binding(2) var<storage, read>       vals: array<u32>;  // [nKeys*nKV*hd] f16-packed
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;  // [nH*hd]
@group(0) @binding(4) var<uniform>             p:    P;
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
    var m: f32 = -1e30;
    var l: f32 = 0.0;
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

// ropeStoreF16: rotate K and store as f16. One thread per cache WORD (kvDim/2
// threads — same count as the f32 ropeStore's nKV*half), owning the two logical
// elements 2*idx, 2*idx+1. Each is rotated independently (its (d,half+d) RoPE
// partner is read from src) and the pair is pack2x16float'd into one word.
const ropeStoreF16ShaderWGSL = `
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
        let c = cos(theta) * p.scale;
        let s = sin(theta) * p.scale;
        return src[off + dd] * c - src[off + p.half + dd] * s;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        let c = cos(theta) * p.scale;
        let s = sin(theta) * p.scale;
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

// kvStoreF16: store V as f16. One thread per word, packing the two adjacent
// V elements 2*idx, 2*idx+1. base (pos*kvDim) is even so words align.
const kvStoreF16ShaderWGSL = `
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

// --- int8-KV variants (opt-in; see task-gpu-kv-i8.md) ---
// Cache is array<u32>, 4 int8/word; a side buffer holds one f32 scale per
// (position, KV-head) at [pos*nKV + head]. WRITE kernels run one thread per KV
// head: rotate (K) / read (V) the head's headDim elements, reduce absmax → scale
// (maxabs/127, matching the CPU QuantizeRowInt8), quantize, pack 4/word. The
// query + ctx stay f32; only the resident cache is int8. base (pos*kvDim) and
// headDim are multiples of 4, so a head's elements pack into whole words.

// attnI8: attnShaderWGSL reading int8 K/V — unpack each element (sign-extend the
// byte) and scale by the per-(position,head) f32 scale. q stays f32; the dot is
// f32 (no integer dot / DP4A — a free future upgrade).
const attnI8ShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:      array<f32>;  // [nH*hd] (RoPE'd, f32)
@group(0) @binding(1) var<storage, read>       keys:   array<u32>;  // [nKeys*nKV*hd] int8-packed (4/word)
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;  // [nKeys*nKV*hd] int8-packed
@group(0) @binding(3) var<storage, read>       kScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(4) var<storage, read>       vScale: array<f32>;  // [nKeys*nKV]
@group(0) @binding(5) var<storage, read_write> ctx:    array<f32>;  // [nH*hd]
@group(0) @binding(6) var<uniform>             p:      P;
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
    var m: f32 = -1e30;
    var l: f32 = 0.0;
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

// ropeStoreI8: rotate K, per-head absmax → int8. One thread per KV head; two
// passes over headDim (recompute the cheap rope rather than spill a local array).
const ropeStoreI8ShaderWGSL = `
struct P { heads: u32, headDim: u32, half: u32, pos: u32, scale: f32, base: u32, nKV: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src:     array<f32>;  // [heads*headDim]
@group(0) @binding(1) var<storage, read>       invFreq: array<f32>;  // [half]
@group(0) @binding(2) var<storage, read_write> dst:     array<u32>;  // K cache int8-packed
@group(0) @binding(3) var<storage, read_write> scales:  array<f32>;  // [maxLen*nKV]
@group(0) @binding(4) var<uniform>             p:       P;
fn rot(off: u32, dd: u32) -> f32 {
    if (dd < p.half) {
        let theta = f32(p.pos) * invFreq[dd];
        return src[off + dd] * cos(theta) * p.scale - src[off + p.half + dd] * sin(theta) * p.scale;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        return src[off + p.half + d] * cos(theta) * p.scale + src[off + d] * sin(theta) * p.scale;
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
    scales[p.pos * p.nKV + h] = sc;
    let inv = 1.0 / sc;
    let wbase = (p.base + off) / 4u;
    for (var d: u32 = 0u; d < p.headDim; d = d + 4u) {
        dst[wbase + d / 4u] = qb(rot(off, d), inv) | (qb(rot(off, d+1u), inv) << 8u) | (qb(rot(off, d+2u), inv) << 16u) | (qb(rot(off, d+3u), inv) << 24u);
    }
}
`

// kvStoreI8: store V as int8, per-head absmax → scale. One thread per KV head.
const kvStoreI8ShaderWGSL = `
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

func (c *Context) ensureAttn() error {
	// Guard on the LAST pipeline built, not the first: these are created in order, so a non-nil last
	// field means every earlier one succeeded too. Guarding on ropePipeline (the first) let a
	// mid-build failure leave the guard satisfied with later pipelines nil, and the next call
	// dispatched a nil pipeline (audit R-30). On a retry after a partial build the earlier fields are
	// rebuilt (the old ones stay tracked for release at Close — bounded, not leaked).
	if c.kvStoreI8Pipeline != nil {
		return nil
	}
	// Shared tracked constructor (gpu.go): registers shader+pipeline for release (audit C-26).
	mk := c.mkPipeline
	var err error
	if c.ropeShader, c.ropePipeline, c.ropeLayout, err = mk("rope", ropeShaderWGSL); err != nil {
		return err
	}
	if c.attnShader, c.attnPipeline, c.attnLayout, err = mk("attn", attnShaderWGSL); err != nil {
		return err
	}
	if c.qkvFinShader, c.qkvFinPipeline, c.qkvFinLayout, err = mk("qkvFinalize", qkvFinalizeShaderWGSL); err != nil {
		return err
	}
	if c.ropeStoreShader, c.ropeStorePipeline, c.ropeStoreLayout, err = mk("rope-store", ropeStoreShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreShader, c.kvStorePipeline, c.kvStoreLayout, err = mk("kv-store", kvStoreShaderWGSL); err != nil {
		return err
	}
	if c.attnF16Shader, c.attnF16Pipeline, c.attnF16Layout, err = mk("attn-f16", attnF16ShaderWGSL); err != nil {
		return err
	}
	if c.ropeStoreF16Shader, c.ropeStoreF16Pipeline, c.ropeStoreF16Layout, err = mk("rope-store-f16", ropeStoreF16ShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreF16Shader, c.kvStoreF16Pipeline, c.kvStoreF16Layout, err = mk("kv-store-f16", kvStoreF16ShaderWGSL); err != nil {
		return err
	}
	if c.attnI8Shader, c.attnI8Pipeline, c.attnI8Layout, err = mk("attn-i8", attnI8ShaderWGSL); err != nil {
		return err
	}
	if c.ropeStoreI8Shader, c.ropeStoreI8Pipeline, c.ropeStoreI8Layout, err = mk("rope-store-i8", ropeStoreI8ShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreI8Shader, c.kvStoreI8Pipeline, c.kvStoreI8Layout, err = mk("kv-store-i8", kvStoreI8ShaderWGSL); err != nil {
		return err
	}
	return nil
}

// RoPE applies rotary embeddings to a host q/k vector [heads*headDim] at position
// pos, returning the rotated vector. invFreq is the layer's [half] inverse freqs.
// (Host in/out — a standalone op for parity; the graph uses the device variant.)
func (c *Context) RoPE(vec []float32, heads, headDim, pos int, invFreq []float32, scale float32) ([]float32, error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	half := len(invFreq)
	vBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rope-vec", Contents: wgpu.ToBytes(vec), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer vBuf.Release()
	fBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rope-invfreq", Contents: wgpu.ToBytes(invFreq), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer fBuf.Release()
	pBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rope-p", Contents: wgpu.ToBytes([]uint32{uint32(heads), uint32(headDim), uint32(half), uint32(pos), f32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.ropeLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: vBuf, Size: vBuf.GetSize()},
		{Binding: 1, Buffer: fBuf, Size: fBuf.GetSize()},
		{Binding: 2, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	if err := c.submitUnary(c.ropePipeline, bg, heads*half); err != nil {
		return nil, err
	}
	return c.Readback(&DeviceBuffer{buf: vBuf, n: len(vec)})
}

// Attention runs single-query attention on host inputs (q RoPE'd [nH*hd], keys/
// vals [nKeys*nKV*hd]) and returns ctx [nH*hd]. Standalone op for parity.
func (c *Context) Attention(q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) ([]float32, error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	group := nH / nKV
	qBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "attn-q", Contents: wgpu.ToBytes(q), Usage: wgpu.BufferUsageStorage})
	defer qBuf.Release()
	kBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "attn-k", Contents: wgpu.ToBytes(keys), Usage: wgpu.BufferUsageStorage})
	defer kBuf.Release()
	vBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "attn-v", Contents: wgpu.ToBytes(vals), Usage: wgpu.BufferUsageStorage})
	defer vBuf.Release()
	cBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "attn-ctx", Size: uint64(nH * hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer cBuf.Release()
	pBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "attn-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(nKeys), uint32(start), uint32(group), f32bits(scale), 0}), Usage: wgpu.BufferUsageUniform})
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.attnLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()},
		{Binding: 1, Buffer: kBuf, Size: kBuf.GetSize()},
		{Binding: 2, Buffer: vBuf, Size: vBuf.GetSize()},
		{Binding: 3, Buffer: cBuf, Size: cBuf.GetSize()},
		{Binding: 4, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.attnPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), 1, 1) // one workgroup per query head
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.Readback(&DeviceBuffer{buf: cBuf, n: nH * hd})
}

func f32bits(f float32) uint32 { return math.Float32bits(f) }
