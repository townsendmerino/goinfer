//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

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

// attnWG is the single-query attention kernels' NARROW workgroup width, and attnWGWide the wide
// one. The kernels put one lane on each head dim, so the workgroup width IS the largest head_dim
// they can dot: above it the tail dims go un-dotted and the o-projection consumes half-zero
// context — plausible-looking WRONG output, no error (audit M-12).
//
// 128 covered every family until the Gated-DeltaNet hybrids, whose RELEASED checkpoints all use
// head_dim 256 (Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Next-80B — verified from their configs, not
// assumed). Their tiny fixtures use 32, so the limit was invisible until a real-width fixture met
// it, and Gemma 4's global layers at head_dim 512 would have met it again.
//
// The wide kernel (attnWideTemplateWGSL, below) gives each lane a STRIDE of dims rather than one
// dim, so head_dim stopped being a reason to decline at all. The narrow kernels are untouched and
// still serve head_dim <= attnWG: striding costs a per-lane array and two extra loops, and every
// ordinary model would pay that for nothing.
//
// attnWG MUST stay equal to the @workgroup_size(128) and `red: array<f32, 128>` in the three
// narrow single-query kernels below.
const (
	attnWG         = 128 // shipped narrow kernels: one lane per dim
	attnWGWide     = 256 // wide kernel workgroup — WebGPU's guaranteed max invocations
	attnMaxPerLane = 8   // dims each wide lane strides over
	// attnMaxHeadDim is therefore 2048, which no model is near. That is the point: head_dim
	// stopped being a reason to decline, rather than the wall moving up one notch to 256.
	attnMaxHeadDim = attnWGWide * attnMaxPerLane
)

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

// attnKeys — decode attention that splits the workgroup over KEYS, not over the head
// dimension. Same math as attnShaderWGSL, different decomposition, and the decomposition
// is the whole point.
//
// attnShaderWGSL puts one lane on each of the hd dimensions, so the q·k dot for EVERY key
// is a cross-lane reduction: red[d]=prod, barrier, 7 barrier'd tree levels, trailing
// barrier = 9 workgroupBarrier() PER KEY. At nKeys=513 that is 4,617 barriers per layer,
// and measured 13.69 ms/token at pos 512 on the 1.5B — 28 layers reading 29.4 MB of KV,
// which at 448 GB/s is a 0.066 ms job. ~0.5% of streaming roofline, against the GEMV path's
// ~83%: a latency wall, not a bandwidth one. (TestDecode_dispatchProfile, G35.)
//
// Here each lane owns a disjoint set of KEYS and computes its own dots with no cross-lane
// traffic; only the softmax max and denominator reduce, ONCE per tile rather than once per
// key. Barriers per layer fall ~250×. This is the shape cuda/attn_block.cu already uses —
// WebGPU's kernel was a generation behind it, not blocked by WGSL.
//
// TILED, because WGSL has no dynamic workgroup storage. CUDA sizes sc[nWin] per launch via
// extern __shared__; a WGSL var<workgroup> is fixed at compile time, so scores are processed
// in TILE-key tiles with the online-softmax state (m, l, acc) carried across them. Storage is
// 2048*4 + 128*4 = 8.5 KB, inside WebGPU's guaranteed 16 KB — no limit raise, no portability
// cost, and any context length works.
//
// vec4 K/q loads are load-bearing, not a micro-optimization. Splitting over keys makes the K
// read stride kvDim across the warp; ncu measured that pattern using only ~22% of each 32-byte
// L1TEX sector on the CUDA twin, which is why attn_block.cu reads float4. Same fix here, and
// it is why the eligibility guard requires hd%4==0 and kvDim%4==0.
//
// NOT bit-identical to attnShaderWGSL: the denominator sums in a different order and the tiled
// rescale reassociates. Attention was never bit-exact anyway — it runs f32 against the CPU
// oracle's f64 (see the note at the top of this file), so the standing gate is TestAttention_parity's
// cosine/maxAbs against that f64 reference, plus argmax through TestWebGPU_forwardParity.
const attnKeysShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q4:   array<vec4<f32>>;  // [nH*hd/4]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       k4:   array<vec4<f32>>;  // [nKeys*kvDim/4]
@group(0) @binding(2) var<storage, read>       vals: array<f32>;        // [nKeys*kvDim]
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;        // [nH*hd]
@group(0) @binding(4) var<uniform>             p:    P;

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

    var m: f32 = -1e30;   // running softmax max
    var l: f32 = 0.0;     // running denominator
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

// attnKeysTile is the key-tile width compiled into attnKeysShaderWGSL's sc[] array. Kept
// beside the shader because the two MUST agree: a larger TILE in the WGSL without a larger
// sc[] writes out of bounds.
const attnKeysTile = 2048

// attnKeysEligible reports whether the key-split attention kernel can serve this geometry.
// f32 KV only (the f16/int8 caches have their own packed kernels), hd within the narrow
// 128-lane kernel, and hd/kvDim both multiples of 4 so the vec4 K/q loads are in bounds and
// aligned — without those loads the key-split read pattern wastes ~78% of each L1TEX sector,
// which is the whole reason cuda/attn_block.cu reads float4.
// attnKeysDisabled force-disables the key-split kernel (GOINFER_ATTN_KEYS=0), so the old
// dim-split kernel can be A/B'd in the same binary. Read once: the plan is recorded per
// runner, and a mid-run flip would leave a half-converted plan.
var attnKeysDisabled = os.Getenv("GOINFER_ATTN_KEYS") == "0"

func attnKeysEligible(hd, kvDim int, kvF16, kvI8 bool) bool {
	if kvF16 || kvI8 || hd > attnWG || hd <= 0 {
		return false
	}
	return hd%4 == 0 && kvDim%4 == 0
}

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
	// Built here (not lazily at first use) so a shader-compile failure surfaces at ensureAttn
	// alongside every other attention pipeline, rather than mid-decode. Cheap: one compile.
	if c.attnKeysShader, c.attnKeysPipeline, c.attnKeysLayout, err = mk("attn-keys", attnKeysShaderWGSL); err != nil {
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
	return c.Readback(newDeviceBuffer(vBuf, len(vec)))
}

// Attention runs single-query attention on host inputs (q RoPE'd [nH*hd], keys/
// vals [nKeys*nKV*hd]) and returns ctx [nH*hd]. Standalone op for parity.
func (c *Context) Attention(q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) ([]float32, error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	return c.attentionOn(c.attnPipeline, c.attnLayout, q, keys, vals, nH, nKV, hd, nKeys, start, scale)
}

// attentionOn is Attention's body, parameterised by which attention pipeline to run. Split out
// so the key-split kernel is exercised through EXACTLY the same host path as the dim-split one —
// a parity test that supplied its own dispatch would be testing its own calling convention rather
// than the kernel the runner uses.
func (c *Context) attentionOn(pl *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout, q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) ([]float32, error) {
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
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ly, Entries: []wgpu.BindGroupEntry{
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
	pass.SetPipeline(pl)
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
	return c.Readback(newDeviceBuffer(cBuf, nH*hd))
}

func f32bits(f float32) uint32 { return math.Float32bits(f) }

// The WIDE single-query attention kernel: one template, STRIDED lanes.
//
// The shipped narrow kernels put one lane on each head dim, which makes the workgroup width a
// hard ceiling on head_dim — 128, and 256 is as far as that idea can go because 256 is WebGPU's
// guaranteed maxComputeInvocationsPerWorkgroup. That is a higher wall, not the absence of one,
// and the wall is real: this family's released checkpoints are head_dim 256 and Gemma 4's global
// layers are 512.
//
// So the wide variant gives each lane a STRIDE of dims — d0, d0+WG, d0+2·WG, … — and any
// head_dim up to attnWGWide·attnMaxPerLane works with a fixed workgroup. The narrow kernels are
// left untouched and still serve head_dim ≤ attnWG, because striding costs a per-lane array and
// two extra loops that every ordinary model would pay for nothing.
//
// One template rather than three near-identical copies: the algorithm (online softmax, the
// tree-reduce, the rescale) is the part that is easy to get subtly wrong and hard to notice, and
// three copies of it is three places for a fix to land in two. The three variants differ ONLY in
// how a K/V element is fetched, which is the part that is obvious on sight.
const attnWideTemplateWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
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
    var m: f32 = -1e30;
    var l: f32 = 0.0;
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

// attnWideVariant is the per-precision half of the template above: the K/V bindings and the two
// fetch expressions, in terms of `ki` (the logical element index) and `sci` (the per-(pos,head)
// scale index the int8 cache needs).
type attnWideVariant struct {
	name         string
	bindings     string
	helpers      string
	kRead, vRead string
}

var attnWideVariants = []attnWideVariant{
	{
		name: "attn-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys: array<f32>;
@group(0) @binding(2) var<storage, read>       vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;
@group(0) @binding(4) var<uniform>             p:    P;`,
		kRead: "keys[ki]",
		vRead: "vals[ki]",
	},
	{
		name: "attn-f16-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys: array<u32>;
@group(0) @binding(2) var<storage, read>       vals: array<u32>;
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;
@group(0) @binding(4) var<uniform>             p:    P;`,
		helpers: `fn f16at(w: u32, e: u32) -> f32 {
    let pair = unpack2x16float(w);
    return select(pair.x, pair.y, (e & 1u) == 1u);
}`,
		kRead: "f16at(keys[ki >> 1u], ki)",
		vRead: "f16at(vals[ki >> 1u], ki)",
	},
	{
		name: "attn-i8-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys:   array<u32>;
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;
@group(0) @binding(3) var<storage, read>       kScale: array<f32>;
@group(0) @binding(4) var<storage, read>       vScale: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctx:    array<f32>;
@group(0) @binding(6) var<uniform>             p:      P;`,
		helpers: `fn unpacki8(w: u32, e: u32) -> f32 {
    let b = (e & 3u) * 8u;
    return f32(i32(w << (24u - b)) >> 24u);
}`,
		kRead: "unpacki8(keys[ki >> 2u], ki) * kScale[sci]",
		vRead: "unpacki8(vals[ki >> 2u], ki) * vScale[sci]",
	},
}

// buildWideAttnWGSL instantiates the template for one variant. It leaves no placeholder behind:
// an unsubstituted __TOKEN__ would be a WGSL compile error rather than a silently wrong kernel,
// but the check is here anyway so the failure names the token instead of a parser column.
func buildWideAttnWGSL(v attnWideVariant) (string, error) {
	src := strings.NewReplacer(
		"__BINDINGS__", v.bindings,
		"__HELPERS__", v.helpers,
		"__KREAD__", v.kRead,
		"__VREAD__", v.vRead,
		"__MAXPER__", strconv.Itoa(attnMaxPerLane),
		"__HALFWG__", strconv.Itoa(attnWGWide/2),
		"__WG__", strconv.Itoa(attnWGWide),
	).Replace(attnWideTemplateWGSL)
	if i := strings.Index(src, "__"); i >= 0 {
		return "", fmt.Errorf("gpu: %s wide attention shader has an unsubstituted placeholder near %q",
			v.name, src[i:min(i+24, len(src))])
	}
	return src, nil
}

// ensureAttnWide compiles the 256-lane variants. Separate from ensureAttn and called only when a
// plan actually has head_dim > attnWG, so every existing family pays nothing — no extra shader
// compiles, no behaviour change, and no dependence on a device limit it does not need.
func (c *Context) ensureAttnWide() error {
	if c.attnWidePipeline != nil {
		return nil
	}
	// maxComputeInvocationsPerWorkgroup is 256 in the WebGPU default limits, but "default" is a
	// floor for conformant implementations, not a promise from this adapter. Check it: a decline
	// here falls back to the staged path, whereas compiling past it is a device error mid-build.
	if lim := c.device.GetLimits().Limits.MaxComputeInvocationsPerWorkgroup; lim < attnWGWide {
		return fmt.Errorf("gpu: device maxComputeInvocationsPerWorkgroup=%d < %d — cannot run the "+
			"wide attention kernel needed for head_dim > %d", lim, attnWGWide, attnWG)
	}
	dst := [][3]any{
		{&c.attnWideShader, &c.attnWidePipeline, &c.attnWideLayout},
		{&c.attnF16WideShader, &c.attnF16WidePipeline, &c.attnF16WideLayout},
		{&c.attnI8WideShader, &c.attnI8WidePipeline, &c.attnI8WideLayout},
	}
	for i, v := range attnWideVariants {
		src, err := buildWideAttnWGSL(v)
		if err != nil {
			return err
		}
		sh, pl, ly, err := c.mkPipeline(v.name, src)
		if err != nil {
			return err
		}
		*dst[i][0].(**wgpu.ShaderModule) = sh
		*dst[i][1].(**wgpu.ComputePipeline) = pl
		*dst[i][2].(**wgpu.BindGroupLayout) = ly
	}
	return nil
}
