//go:build gpu

package gpu

import (
	"fmt"
	"slices"
	"sync/atomic"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// rmsnormBatchedShaderWGSL is rmsnormShaderWGSL (layer.go) widened to M rows in one
// dispatch: grid (1, M) instead of (1, 1), each workgroup owns row wid.y and reduces
// only that row (workgroupBarrier never crosses rows), so the per-row math and
// reduction order are identical to calling the M=1 kernel M times — bit-identical
// by construction, gated by TestRMSNormBatched_parity. The pass uses it for a
// sublayer's float norm (Gemma's sandwich norms), where decode runs rmsnormShaderWGSL.
const rmsnormBatchedShaderWGSL = `
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

// ropeBatchedShaderWGSL is ropeShaderWGSL (attention.go) widened to M rows in one dispatch, grid (ceil(heads*half/64), M).
// Positions are contiguous within one PrefillLastW8A8 call, so row r's position is p.start+row. Same per-(row, head, d)
// math as the M=1 kernel, so bit-identical (TestRoPEBatched_parity).
const ropeBatchedShaderWGSL = `
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
    let c = cos(theta) * p.scale;
    let s = sin(theta) * p.scale;
    let base = row * p.heads * p.headDim;
    let off = base + h * p.headDim;
    let x1 = vec[off + d];
    let x2 = vec[off + p.half + d];
    vec[off + d]           = x1 * c - x2 * s;
    vec[off + p.half + d]  = x2 * c + x1 * s;
}
`

// attnBatchedShaderWGSL is attnShaderWGSL (attention.go) widened to M query rows in one dispatch, grid (nH, M); row r's
// causal bound is basePos+r+1 (positions are contiguous within one PrefillLastW8A8 call). No sink support:
// runModelToModelW declines FeatAttnSink. Not claimed bit-identical to M calls of attnShaderWGSL: gated on
// cosine/maxAbs like every attention kernel pair here.
const attnBatchedShaderWGSL = `
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

// attnKeysBatchedShaderWGSL is attnKeysShaderWGSL (attention.go) widened to M query rows the way
// attnBatchedShaderWGSL widens the plain kernel: grid (nH, M), each row's own causal tile loop from key 0 to
// basePos+row inclusive. Selected whenever attnKeysEligible(hd, kvDim, false, false), matching attnKernel's
// preference for the tiled/key-split decomposition; most real dense architectures take this path.
const attnKeysBatchedShaderWGSL = `
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

// attnBatchedF16ShaderWGSL is attnBatchedShaderWGSL with f16-packed keys and values:
// keys and vals are array<u32>, two f16 per word, unpacked with unpack2x16float.
const attnBatchedF16ShaderWGSL = `
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

// attnBatchedKernel picks the batched attention pipeline+layout for a geometry,
// mirroring attnKernel's (attention.go) own preference for the tiled key-split
// decomposition, and routing f16 KV cache to attnBatchedF16.
func (c *Context) attnBatchedKernel(hd, kvDim int, kvF16 ...bool) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	if len(kvF16) > 0 && kvF16[0] {
		return c.attnBatchedF16Pipeline, c.attnBatchedF16Layout
	}
	if !attnKeysDisabled && attnKeysEligible(hd, kvDim, false, false) {
		return c.attnKeysBatchedPipeline, c.attnKeysBatchedLayout
	}
	return c.attnBatchedPipeline, c.attnBatchedLayout
}

// ensurePrefillBatched lazily compiles the batched-row norm, rotary and attention kernels the batched pass needs beyond
// its existing ensure-list. They are kept apart from ensureLayer/ensureAttn's M=1 pipelines so
// the decode path never pays for, or risks, them. History: docs/code-notes/gpu.md#ensurePrefillBatched.
func (c *Context) ensurePrefillBatched() error {
	mk := c.mkPipeline
	var err error
	if c.rmsnormBatchedPipeline == nil {
		if c.rmsnormBatchedShader, c.rmsnormBatchedPipeline, c.rmsnormBatchedLayout, err = mk("rmsnorm-batched", rmsnormBatchedShaderWGSL); err != nil {
			return err
		}
	}
	if c.ropeBatchedPipeline == nil {
		if c.ropeBatchedShader, c.ropeBatchedPipeline, c.ropeBatchedLayout, err = mk("rope-batched", ropeBatchedShaderWGSL); err != nil {
			return err
		}
	}
	if c.attnBatchedPipeline == nil {
		if c.attnBatchedShader, c.attnBatchedPipeline, c.attnBatchedLayout, err = mk("attn-batched", attnBatchedShaderWGSL); err != nil {
			return err
		}
	}
	if c.attnKeysBatchedPipeline == nil {
		if c.attnKeysBatchedShader, c.attnKeysBatchedPipeline, c.attnKeysBatchedLayout, err = mk("attn-keys-batched", attnKeysBatchedShaderWGSL); err != nil {
			return err
		}
	}
	if c.attnBatchedF16Pipeline == nil {
		if c.attnBatchedF16Shader, c.attnBatchedF16Pipeline, c.attnBatchedF16Layout, err = mk("attn-batched-f16", attnBatchedF16ShaderWGSL); err != nil {
			return err
		}
	}
	return nil
}

// prefillProj is one projection of the batched pass: an int8 matrix or an int4 one, never both.
type prefillProj struct {
	w8 *ResidentW8A8
	w4 *ResidentW4A8
}

func asPrefillProj(w decodeWeight) (prefillProj, bool) {
	switch v := w.(type) {
	case *ResidentW8A8:
		return prefillProj{w8: v}, v != nil
	case *ResidentW4A8:
		return prefillProj{w4: v}, v != nil
	}
	return prefillProj{}, false
}

// rows, cols and kp are the projection's output width, its input width, and the input width padded as its kernel reads
// a quantized activation row.
func (p prefillProj) rows() int {
	if p.w4 != nil {
		return p.w4.rows
	}
	return p.w8.rows
}

func (p prefillProj) cols() int {
	if p.w4 != nil {
		return p.w4.cols
	}
	return p.w8.cols
}

func (p prefillProj) kp() int {
	if p.w4 != nil {
		return p.w4.kp
	}
	return p.w8.kp
}

// prefillLayerProj is one layer's projections, and what a Gemma layer adds to the plain shape. Every Gemma field is
// zero for a plain layer.
type prefillLayerProj struct {
	q, k, v, o, gate, up, down prefillProj

	hd, nKV      int          // this layer's head size and KV heads; 0 means the model's
	kEqV         bool         // no v projection: V is the value norm of the raw K
	vNorm        bool         // V is the value norm of the v projection
	kvShared     bool         // a query only; the layer's caches are its source layer's and it stores nothing
	local        bool         // a sliding-window layer
	postAttnNorm *wgpu.Buffer // float norms of a sublayer's output before its residual add; nil for none
	postMLPNorm  *wgpu.Buffer
	scalar       float32 // the layer's output scalar; 0 or 1 for none

	// A Gemma 4 E-model's per-layer-input branch after the feed-forward residual; pleGate unset for none.
	pleGate, pleProj prefillProj
	postPLENorm      *wgpu.Buffer
}

// set reports whether p names a projection.
func (p prefillProj) set() bool { return p.w8 != nil || p.w4 != nil }

// prefillProjs is a model's projections for the batched pass, beside the ModelW that carries everything else.
type prefillProjs struct {
	layers []prefillLayerProj
	lmHead prefillProj
	window int // a sliding-window layer attends its last window positions; 0 when no layer is one
	pleDim int // a Gemma 4 E-model's per-layer input width: a row is [hidden ‖ layers·pleDim]; 0 otherwise
	// mropeAxis is an m-RoPE model's axis table for a head half (decoder.Model.MRopeAxisResident): which of a row's
	// three positions each rotary pair reads. nil for any other model, and for a caller that has no such table.
	mropeAxis func(half int) []int
}

// gemma reports whether the model has anything of the Gemma shape, which a ModelW cannot carry.
func (pp prefillProjs) gemma() bool {
	if pp.window != 0 || pp.pleDim != 0 {
		return true
	}
	for i := range pp.layers {
		l := &pp.layers[i]
		if l.hd != 0 || l.kEqV || l.vNorm || l.kvShared || l.local || l.postAttnNorm != nil || l.postMLPNorm != nil ||
			(l.scalar != 0 && l.scalar != 1) || l.pleGate.set() {
			return true
		}
	}
	return false
}

// int4 reports whether any projection is int4.
func (pp prefillProjs) int4() bool {
	if pp.lmHead.w4 != nil {
		return true
	}
	for i := range pp.layers {
		l := &pp.layers[i]
		for _, p := range []prefillProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down, l.pleGate, l.pleProj} {
			if p.w4 != nil {
				return true
			}
		}
	}
	return false
}

// int8Bias reports whether a q, k or v projection that carries a bias in m is int8.
func (pp prefillProjs) int8Bias(m ModelW) bool {
	for i := range pp.layers {
		l := &pp.layers[i]
		if m.Layers[i].Attn.QBias != nil && (l.q.w8 != nil || l.k.w8 != nil || l.v.w8 != nil) {
			return true
		}
	}
	return false
}

// modelWProjs is m's int8 projections as a prefillProjs.
func modelWProjs(m ModelW) prefillProjs {
	pp := prefillProjs{layers: make([]prefillLayerProj, len(m.Layers)), lmHead: prefillProj{w8: m.LMHead}}
	for i := range m.Layers {
		l := &m.Layers[i]
		pp.layers[i] = prefillLayerProj{
			q: prefillProj{w8: l.Attn.QProj}, k: prefillProj{w8: l.Attn.KProj}, v: prefillProj{w8: l.Attn.VProj}, o: prefillProj{w8: l.Attn.OProj},
			gate: prefillProj{w8: l.Gate}, up: prefillProj{w8: l.Up}, down: prefillProj{w8: l.Down},
		}
	}
	return pp
}

// runModelToModelW is runModelToPrefill for a plain model whose projections are all int8: the ModelW PrefillLastW8A8
// takes, which has no field for a sliding window or any other part of the Gemma shape.
func runModelToModelW(rm *runModel, hd int) (ModelW, bool) {
	if rm.slidingWindow != 0 {
		return ModelW{}, false
	}
	mw, pp, ok := runModelToPrefill(rm, hd)
	if !ok || pp.int4() || pp.gemma() {
		return ModelW{}, false
	}
	return mw, true
}

// runModelToPrefill narrows a resident runModel to the dense shapes the batched pass accepts, returning ok=false if
// the model uses anything outside them: MoE (Gemma 4's parallel block included), MLA, SSM, DeltaNet, int8 KV, partial
// RoPE, a rope scale, an attention sink, an o-proj bias. Each projection may be int8 or int4; the ModelW's own
// projection fields are set for the int8 ones only. The Gemma shape (prefillLayerProj's own fields) is accepted.
//
// q/k/v bias is accepted, but callers must also check prefillProjs.int8Bias against the backend: an int8 projection's
// fused bias epilogue is proven bit-exact on Vulkan only and is declined elsewhere until proven there. See
// BuildResident's prefillLast gate in residency.go; history in docs/code-notes/gpu.md#runModelToModelW. An int4
// projection adds its bias in a separate step, as decode does, on every backend.
//
// The result is a zero-copy view over rm's buffers; callers must not Close it.
// hd is the model's head dimension, used to tell genuine partial RoPE (declined) from a half set explicitly to hd/2.
func runModelToPrefill(rm *runModel, hd int) (ModelW, prefillProjs, bool) {
	none := func() (ModelW, prefillProjs, bool) { return ModelW{}, prefillProjs{}, false }
	if rm.moe != nil || rm.g4moe || rm.mla != nil || rm.mamba != nil || rm.dnet != nil || rm.kvI8 {
		return none()
	}
	view := func(b *wgpu.Buffer) *DeviceBuffer {
		if b == nil {
			return nil
		}
		return &DeviceBuffer{buf: b}
	}
	layers := make([]LayerW, len(rm.layers))
	pp := prefillProjs{layers: make([]prefillLayerProj, len(rm.layers)), pleDim: rm.pleDim}
	anyGeom := slices.ContainsFunc(rm.layers, func(l runLayer) bool { return l.ghd != 0 })
	if !anyGeom && rm.ropeHalf != 0 && rm.ropeHalf != hd/2 {
		return none() // the pass's rotary kernel turns every pair of a head
	}
	for i := range rm.layers {
		l := &rm.layers[i]
		if l.oBias != nil || l.hasSink || (l.ropeScale != 0 && l.ropeScale != 1) ||
			l.isMoE || l.g4moe != nil || l.shGate != nil ||
			l.mlaQA != nil || l.mlaQB != nil || l.mlaQ != nil || l.mlaKVA != nil || l.mlaO != nil ||
			l.isMamba || l.nemoKind != nemoNone || l.isDeltaNet {
			return none()
		}
		if l.ghd != 0 && l.ghalf*2 != l.ghd {
			return none()
		}
		if l.kvShared {
			if l.kNorm != nil && l.qNorm == nil {
				return none()
			}
		} else if (l.qNorm == nil) != (l.kNorm == nil) {
			return none()
		}
		lp := prefillLayerProj{hd: l.ghd, nKV: l.gnKV, kEqV: l.gKEqV, vNorm: l.vNorm, kvShared: l.kvShared, local: l.isLocal,
			postAttnNorm: l.postAttnNorm, postMLPNorm: l.postMLPNorm, scalar: l.layerScalar, postPLENorm: l.postPLENorm}
		ok := true
		need := func(w decodeWeight) prefillProj {
			pj, pok := asPrefillProj(w)
			ok = ok && pok
			return pj
		}
		lp.q, lp.o, lp.gate, lp.up, lp.down = need(l.q), need(l.o), need(l.gate), need(l.up), need(l.down)
		if !l.kvShared {
			lp.k = need(l.k)
			if !l.gKEqV {
				lp.v = need(l.v)
			}
		}
		if l.pleGate != nil {
			lp.pleGate, lp.pleProj = need(l.pleGate), need(l.pleProj)
			ok = ok && l.postPLENorm != nil && rm.pleDim > 0
		}
		if !ok {
			return none() // a projection that is neither int8 nor int4: no multi-row kernel for it
		}
		if l.isLocal {
			pp.window = rm.slidingWindow
		}
		pp.layers[i] = lp
		layers[i] = LayerW{
			Attn: AttnWeights{
				Norm: view(l.attnNorm), QProj: lp.q.w8, KProj: lp.k.w8, VProj: lp.v.w8, OProj: lp.o.w8,
				InvFreq: view(l.invFreq), KCache: view(l.kCache), VCache: view(l.vCache),
				QBias: view(l.qBias), KBias: view(l.kBias), VBias: view(l.vBias),
				QNorm: view(l.qNorm), KNorm: view(l.kNorm),
				QGate: l.qGate,
			},
			MLPNorm: view(l.mlpNorm),
			Gate:    lp.gate.w8, Up: lp.up.w8, Down: lp.down.w8,
		}
	}
	lmHead, lmOK := asPrefillProj(rm.lmHead)
	if !lmOK {
		return none()
	}
	pp.lmHead = lmHead
	return ModelW{
		Layers:    layers,
		FinalNorm: view(rm.finalNorm),
		LMHead:    lmHead.w8,
		GatedGELU: rm.gatedGELU,
		KVF16:     rm.kvF16,
	}, pp, true
}

// biasRowsShaderWGSL adds a per-column bias to every row of a packed [M, n] buffer: decode's standalone bias add
// (residualShaderWGSL against the bias) for all rows in one dispatch, the same single addition per element.
const biasRowsShaderWGSL = `
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

// PrefillLastW8A8 is the batched-prefill analogue of DecodeTokenFusedBatched: M prompt-token rows through the dense W8A8
// layers with each projection as ONE tiled GEMM (the unbounded-M kernel in gemm.go, DP4A when Context.hasDP4A) and the
// cheap per-row ops looped over the rows, in DecodeTokenFusedBatched's write-all-KV-then-attend order with its
// flushPasses handling. M is not capped at gemmRowMaxM. Only the last row's logits are computed
// (decoder.Prefiller.PrefillLast's contract); every row's KV is written.
//
// Bit-equivalent to M sequential DecodeRunner.Run calls at positions start..start+M-1: the same int8 inputs (each
// activation is quantized by decode's own fused kernel in its rows form, prefill_fused.go), int32 accumulation, per-row
// ops and ordering; gated by TestPrefillLastW8A8_parity and TestTiledDP4A_parity.
func (c *Context) PrefillLastW8A8(xs [][]float32, m ModelW, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([]float32, error) {
	return c.prefillLastRows(xs, m, modelWProjs(m), nil, hidden, nH, nKV, hd, inter, positions, start, eps, scale, addOne)
}

// prefillLastRows is PrefillLastW8A8 with each projection int8 or int4 (pp; m's own projection fields are not read).
// An int4 projection runs a multi-row int4 kernel (Context.w4MultiFor) and adds its q/k/v bias in a separate step, as
// decode does. Both kernels reproduce the single-row kernel's sum, so an int4 model's pass is bit-equivalent to the
// sequential DecodeRunner calls as the int8 pass is; gated by TestPrefillInt4_matchesWalk.
//
// mrope, when not nil, is each row's three rotary positions (time, height, width) for an m-RoPE model: a pair rotates
// by the one pp.mropeAxis names for it, and positions still gives each row's cache slot.
func (c *Context) prefillLastRows(xs [][]float32, m ModelW, pp prefillProjs, mrope [][3]int, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([]float32, error) {
	M := len(xs)
	if M == 0 {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 M=0")
	}
	if len(positions) != M {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 %d rows but %d positions", M, len(positions))
	}
	if M > 65535 {
		return nil, fmt.Errorf("gpu: the batched pass takes at most 65535 rows a call, got %d", M)
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled, c.ensureTiledBias, c.ensurePrefillBatched, c.ensureQKNorm, c.ensureFuse, c.ensureFusedRows} {
		if err := f(); err != nil {
			return nil, err
		}
	}
	if m.hasQGate() {
		if err := c.ensureDeltaQSplit(); err != nil {
			return nil, err
		}
		if err := c.ensureDeltaAttnGate(); err != nil {
			return nil, err
		}
	}
	if mrope != nil {
		if len(mrope) != M || pp.mropeAxis == nil {
			return nil, fmt.Errorf("gpu: the batched pass got %d rows of rotary positions for %d rows (axis table set: %v)", len(mrope), M, pp.mropeAxis != nil)
		}
		if err := c.ensureMRoPERows(); err != nil {
			return nil, err
		}
	}
	if pp.gemma() {
		for _, f := range []func() error{c.ensureVNorm, c.ensureScaleVec, c.ensureAttnWide} {
			if err := f(); err != nil {
				return nil, err
			}
		}
	}
	if pp.int4() {
		if err := c.ensureW4Multi(); err != nil {
			return nil, err
		}
		if c.biasRowsPipeline == nil {
			var err error
			if _, c.biasRowsPipeline, c.biasRowsLayout, err = c.mkPipeline("bias-rows", biasRowsShaderWGSL); err != nil {
				return nil, err
			}
		}
	}

	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	keepBuf := func(b *wgpu.Buffer) { keep = append(keep, b.Release) }
	keepBG := func(b *wgpu.BindGroup) { keep = append(keep, b.Release) }

	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer func() { enc.Release() }()

	// buildErr accumulates the FIRST device-allocation/bind failure (audit C-27, mirrored
	// from DecodeTokenFusedBatched): short-circuit once set, return it before Submit, so
	// VRAM exhaustion is an error the caller falls back on rather than a panic or a
	// downstream nil-buffer deref.
	var buildErr error
	// R-25 (docs/tasks/task-recompute-audit.md): a layer's buffers are dead once the layer is encoded (the residual
	// stream is updated in place, K/V are copied into the cache), so each layer recycles the one before it: storF hands
	// out a free buffer of the same size before it creates one. A recycled buffer was written before, so wgpu does not
	// zero-fill it again, and the pass holds one layer's intermediates instead of every layer's. The queue orders the
	// dispatches, so a later layer's writes follow the earlier layer's reads. prefillBufFresh (tests only) restores a
	// fresh buffer per call; prefillBufPoison (tests only) overwrites every recycled buffer with NaN bytes first.
	free := map[int][]*wgpu.Buffer{}
	var layerBufs []*wgpu.Buffer
	var poison *wgpu.Buffer
	recycle := func() {
		for _, b := range layerBufs {
			n := int(b.GetSize() / 4)
			free[n] = append(free[n], b)
		}
		layerBufs = layerBufs[:0]
	}
	storF := func(n int) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		if l := free[n]; !prefillBufFresh && len(l) > 0 {
			b := l[len(l)-1]
			free[n] = l[:len(l)-1]
			if prefillBufPoison {
				if poison == nil || poison.GetSize() < uint64(n*4) {
					fill := make([]uint32, n)
					for i := range fill {
						fill[i] = 0xFFFFFFFF
					}
					p, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(fill), Usage: wgpu.BufferUsageCopySrc})
					if e != nil {
						buildErr = e
						return nil
					}
					keepBuf(p)
					poison = p
				}
				if e := enc.TryCopyBufferToBuffer(poison, 0, b, 0, uint64(n*4)); e != nil {
					buildErr = e
					return nil
				}
			}
			layerBufs = append(layerBufs, b)
			prefillBufReused.Add(1)
			return b
		}
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
		layerBufs = append(layerBufs, b)
		return b
	}
	uni := func(v []uint32) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
		return b
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		if buildErr != nil {
			return nil
		}
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			if b == nil {
				buildErr = fmt.Errorf("gpu: PrefillLastW8A8: nil buffer for binding %d (allocation failed)", i)
				return nil
			}
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBG(bg)
		return bg
	}
	// flushPasses: same Metal-uncommitted-command-buffer-cap workaround as
	// DecodeTokenFusedBatched (see its doc comment) — a real constraint here too, and
	// worse: real prefill M runs to the hundreds/thousands vs speculative verify's ≤16.
	flushPasses := func() {
		if buildErr != nil {
			return
		}
		cmd, e := enc.TryFinish(nil)
		if e != nil {
			buildErr = e
			return
		}
		c.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		enc, e = c.device.TryCreateCommandEncoder(nil)
		if e != nil {
			buildErr = e
		}
	}
	const passesPerFlush = 32
	dispCount := 0
	disp := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		if buildErr != nil || bg == nil {
			return
		}
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, gy, 1)
		if err := pass.TryEnd(); err != nil {
			buildErr = err
		}
		pass.Release()
		dispCount++
		if dispCount >= passesPerFlush {
			flushPasses()
			dispCount = 0
		}
	}
	cpy := func(src *wgpu.Buffer, so uint64, dst *wgpu.Buffer, do, sz uint64) {
		if buildErr != nil || src == nil || dst == nil {
			return
		}
		if err := enc.TryCopyBufferToBuffer(src, so, dst, do, sz); err != nil {
			buildErr = err
		}
	}
	// bindEntry / bindOff: like bind(), but each argument may be a byte-range VIEW into
	// a larger buffer rather than the whole thing — needed by dispFlat's chunking below.
	// Every offset dispFlat passes is a multiple of maxChunkElems*4 bytes, and
	// maxChunkElems is itself a multiple of 64 (the workgroup size), so every offset is
	// automatically a multiple of 256 — satisfying GPUBindGroupEntry.offset's
	// minStorageBufferOffsetAlignment requirement (the same constraint that ruled out
	// per-row offset views for the attention step above, where nH*hd*4 was not
	// guaranteed 256-aligned; a chunk boundary chosen as a multiple of the workgroup
	// size always is).
	type bindEntry struct {
		buf    *wgpu.Buffer
		off, n uint32 // n in ELEMENTS (f32); off in elements too
	}
	whole := func(b *wgpu.Buffer) bindEntry { return bindEntry{buf: b} }
	bindOff := func(layout *wgpu.BindGroupLayout, entries ...bindEntry) *wgpu.BindGroup {
		if buildErr != nil {
			return nil
		}
		es := make([]wgpu.BindGroupEntry, len(entries))
		for i, e := range entries {
			if e.buf == nil {
				buildErr = fmt.Errorf("gpu: PrefillLastW8A8: nil buffer for binding %d (allocation failed)", i)
				return nil
			}
			sz := e.buf.GetSize() - uint64(e.off)*4
			if e.n != 0 {
				sz = uint64(e.n) * 4
			}
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: e.buf, Offset: uint64(e.off) * 4, Size: sz}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBG(bg)
		return bg
	}
	// maxChunkElems: WebGPU limits a dispatch to 65535 workgroups per dimension
	// (maxComputeWorkgroupsPerDimension) and each workgroup here covers 64 elements, so one 1-D
	// dispatch tops out at 65535*64 elements; larger elementwise work is chunked. Exceeding the
	// limit is a validation error that PrefillLast declines into, not a wrong-output risk.
	const maxChunkElems = 65535 * 64
	dispFlat := func(pl *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout, total int, bufs ...*wgpu.Buffer) {
		off := 0
		for off < total {
			n := min(total-off, maxChunkElems)
			p := uni([]uint32{uint32(n), 0, 0, 0})
			entries := make([]bindEntry, 0, len(bufs)+1)
			for _, b := range bufs {
				entries = append(entries, bindEntry{buf: b, off: uint32(off), n: uint32(n)})
			}
			entries = append(entries, whole(p))
			disp(pl, bindOff(ly, entries...), uint32(n+63)/64, 1)
			off += n
		}
	}

	// The op builders below all work on PACKED [M, width] row-major buffers: one dispatch across all M rows, not M
	// separate dispatches (docs/code-notes/gpu.md#ensurePrefillBatched has the measured before and after).

	// rmsQuantB is decode's fused RMSNorm→int8 for all M rows of a packed [M, hidden] buffer, one workgroup a row,
	// into the [M, kp/4] packed-int8 / [M] scales layout a projection reads.
	rmsQuantB := func(inPacked, w *wgpu.Buffer, kp int) (aqC, asC *wgpu.Buffer) {
		aqC, asC = storF(M*kp/4), storF(M)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), uint32(kp)})
		f := c.fusedRows[fusedRowsRMS]
		disp(f.pl, bind(f.layout, inPacked, w, aqC, asC, p), uint32(M), 1)
		return aqC, asC
	}
	// qkNormB normalizes ALL M rows of a packed [M, heads*hd] buffer in place with
	// weight [hd], before RoPE (Qwen3/GLM). Each workgroup processes one head across
	// the batch.
	qkNormB := func(vecPacked, weight *wgpu.Buffer, heads, lhd int) {
		totalHeads := uint32(M * heads)
		p := uni([]uint32{totalHeads, uint32(lhd), f32bits(eps), boolU32(addOne)})
		gx := totalHeads
		gy := uint32(1)
		if gx > 65535 {
			gx = 65535
			gy = (totalHeads + 65534) / 65535
		}
		disp(c.qkNormPipeline, bind(c.qkNormLayout, vecPacked, weight, p), gx, gy)
	}
	// ropeB rotates ALL M rows of a packed [M, heads*hd] buffer in place, in one
	// dispatch. positions are always start, start+1, …, start+M-1 within one
	// PrefillLastW8A8 call (residency.go), so passing start (not a per-row array) is
	// enough — ropeBatchedShaderWGSL recovers row r's position as start+r.
	// mposBuf is each row's three rotary positions and axisFor a head half's axis table, for an m-RoPE pass.
	var mposBuf *wgpu.Buffer
	axisBufs := map[int]*wgpu.Buffer{}
	axisFor := func(half int) *wgpu.Buffer {
		if b, ok := axisBufs[half]; ok || buildErr != nil {
			return b
		}
		ax := pp.mropeAxis(half)
		if len(ax) != half {
			buildErr = fmt.Errorf("gpu: the batched pass has no m-RoPE axis table for a head half of %d", half)
			return nil
		}
		tab := make([]uint32, half)
		for d := range tab {
			tab[d] = uint32(ax[d])
			switch prefillDefectForTest {
			case 12:
				tab[d] = uint32(ax[(d+1)%half])
			case 14:
				tab[d] = []uint32{0, 2, 1}[ax[d]]
			}
		}
		b, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(tab), Usage: wgpu.BufferUsageStorage})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
		axisBufs[half] = b
		return b
	}
	if mrope != nil {
		flat := make([]uint32, 0, 3*M)
		for r := range mrope {
			src := mrope[r]
			if prefillDefectForTest == 13 {
				src = mrope[(r+1)%M]
			}
			flat = append(flat, uint32(src[0]), uint32(src[1]), uint32(src[2]))
		}
		b, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(flat), Usage: wgpu.BufferUsageStorage})
		if e != nil {
			return nil, e
		}
		keepBuf(b)
		mposBuf = b
	}
	ropeB := func(vecPacked, invFreq *wgpu.Buffer, heads, lhd int) {
		half := lhd / 2
		rpos := positions[0]
		if prefillDefectForTest == 1 {
			rpos++
		}
		p := uni([]uint32{uint32(heads), uint32(lhd), uint32(half), uint32(rpos), f32bits(1), 0, 0, 0})
		if mposBuf != nil {
			disp(c.ropeMRoPERows.pl, bind(c.ropeMRoPERows.layout, vecPacked, invFreq, p, mposBuf, axisFor(half)), uint32(heads*half+63)/64, uint32(M))
			return
		}
		disp(c.ropeBatchedPipeline, bind(c.ropeBatchedLayout, vecPacked, invFreq, p), uint32(heads*half+63)/64, uint32(M))
	}
	// residualB dispatches its elementwise add as ONE call over the full M*width flattened range instead of M separate
	// width-sized dispatches: the kernel indexes by a flat global_invocation_id with no cross-row state
	// (residualShaderWGSL, layer.go), so [M, width] IS [M*width] to it.
	residualB := func(xPacked, yPacked *wgpu.Buffer) *wgpu.Buffer { // x += y, in place; returns x
		dispFlat(c.residualPipeline, c.residualLayout, M*hidden, xPacked, yPacked)
		return xPacked
	}
	// gatedQuantB is decode's fused gated product→int8 for all M rows, one workgroup a row, over n activations a row:
	// the [M, n] product is never stored. up is a range of a buffer, so one layer's per-layer inputs bind out of the
	// model's.
	gatedQuantB := func(gatePacked *wgpu.Buffer, up bindEntry, n, kp int) (aqC, asC *wgpu.Buffer) {
		aqC, asC = storF(M*kp/4), storF(M)
		p := uni([]uint32{uint32(n), uint32(kp), 0, 0})
		f := c.fusedRows[fusedRowsSwiGLU]
		if m.GatedGELU {
			f = c.fusedRows[fusedRowsGeGLU]
		}
		disp(f.pl, bindOff(f.layout, whole(gatePacked), up, whole(aqC), whole(asC), whole(p)), uint32(M), 1)
		return aqC, asC
	}
	// rmsF32B is decode's float RMSNorm (rmsnormShaderWGSL) for all M rows of a packed [M, hidden] buffer.
	rmsF32B := func(inPacked, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(M * hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormBatchedPipeline, bind(c.rmsnormBatchedLayout, inPacked, w, out, p), 1, uint32(M))
		return out
	}
	// vNormB is decode's value norm (vNormWGSL, one workgroup a head) over every row's heads at once.
	vNormB := func(src *wgpu.Buffer, heads, lhd int) *wgpu.Buffer {
		dst := storF(M * heads * lhd)
		if M*heads > 65535 && buildErr == nil {
			buildErr = fmt.Errorf("gpu: the batched pass declines %d rows of %d value heads in one dispatch", M, heads)
		}
		p := uni([]uint32{uint32(M * heads), uint32(lhd), f32bits(eps), 0})
		disp(c.vNormPipeline, bind(c.vNormLayout, src, dst, p), uint32(M*heads), 1)
		return dst
	}
	// scaleB multiplies every row of a packed [M, hidden] buffer by a layer's output scalar, in place.
	scaleB := func(vec *wgpu.Buffer, scalar float32) {
		n := M * hidden
		p := uni([]uint32{uint32(n), f32bits(scalar), 0, 0})
		disp(c.scaleVecPipeline, bind(c.scaleVecLayout, vec, p), uint32(n+255)/256, 1)
	}
	// quantPackedM quantizes all M rows of a packed [M, K] buffer in ONE dispatch, straight into the [M, kp/4]
	// packed-int8 / [M] scales layout a projection reads (quantizeShaderWGSL, device.go, takes M rows natively via
	// QDims.m). It is decode's quantizer for an activation that no fused kernel produces: the attention context.
	quantPackedM := func(inPacked *wgpu.Buffer, K, kp int) (aqC, asC *wgpu.Buffer) {
		aqC, asC = storF(M*kp/4), storF(M)
		p := uni([]uint32{uint32(M), uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, inPacked, aqC, asC, p), uint32(M), 1)
		return aqC, asC
	}
	// projQB runs one projection over rowsM quantized rows and returns the packed [rowsM, N] output: an int8 matrix
	// through ONE unbounded-M tiled GEMM (weight streamed once, DP4A when available), an int4 one through a multi-row
	// int4 kernel.
	//
	// bias (nil for every projection except Qwen2's q/k/v) selects, for an int8 matrix, the bias-epilogue tiled kernel
	// instead of a separate add: the add has to be fused into the GEMM's expression to stay bit-exact with production's
	// gemvBias (docs/code-notes/gpu.md#runModelToModelW). An int4 matrix adds it in a separate dispatch, as decode does.
	var projQB func(aqC, asC *wgpu.Buffer, rowsM int, rm prefillProj, bias *wgpu.Buffer) *wgpu.Buffer
	// sharedQ returns the projections of one quantized input through rms. R-25 (docs/tasks/task-recompute-audit.md): the
	// input is quantized once when every projection has the same K and reads it at the same padding, which q/k/v and
	// gate/up of one precision always do. Quantization is deterministic, so each projection reads the bytes it would
	// have read from its own copy; prefillQuantPerProj (tests only) restores a quantization each.
	sharedQ := func(quantize func(kp int) (aqC, asC *wgpu.Buffer), rms []prefillProj, biases []*wgpu.Buffer) []*wgpu.Buffer {
		out := make([]*wgpu.Buffer, len(rms))
		same := !prefillQuantPerProj
		for _, rm := range rms[1:] {
			same = same && rm.cols() == rms[0].cols() && rm.kp() == rms[0].kp()
		}
		var aqC, asC *wgpu.Buffer
		if same {
			aqC, asC = quantize(rms[0].kp())
			prefillSharedQuants.Add(1)
		}
		for i, rm := range rms {
			if prefillDefectForTest == 3 && i == 0 && len(rms) == 3 && rm.w4 != nil && rms[1].w4 != nil {
				wrong := *rm.w4
				wrong.bScales = rms[1].w4.bScales
				rm = prefillProj{w4: &wrong}
			}
			if same {
				out[i] = projQB(aqC, asC, M, rm, biases[i])
			} else {
				a, as := quantize(rm.kp())
				out[i] = projQB(a, as, M, rm, biases[i])
			}
		}
		return out
	}
	projQB = func(aqC, asC *wgpu.Buffer, rowsM int, pj prefillProj, bias *wgpu.Buffer) *wgpu.Buffer {
		N := pj.rows()
		dstC := storF(rowsM * N)
		if pj.w4 != nil {
			if buildErr != nil {
				return dstC
			}
			kind := c.w4MultiFor(rowsM)
			encodeW4Multi(kind, c.w4Multi[kind], pj.w4, rowsM, aqC, asC, dstC, uni, bind, disp)
			if bias != nil {
				total := uint32(rowsM * N)
				gx := (total + 63) / 64
				gy := uint32(1)
				if gx > 65535 {
					gy = (gx + 65534) / 65535
					gx = 65535
				}
				p := uni([]uint32{total, uint32(N), gx * 64, 0})
				disp(c.biasRowsPipeline, bind(c.biasRowsLayout, dstC, bias, p), gx, gy)
			}
			return dstC
		}
		rm := pj.w8
		p := uni([]uint32{uint32(rowsM), uint32(rm.kp), uint32(N), 0})
		gx, gy := c.gemmGrid(N, rowsM)
		if bias != nil {
			disp(c.tiledBiasPipeline, bind(c.tiledBiasLayout, aqC, rm.bq, asC, rm.bScales, dstC, p, bias), gx, gy)
		} else {
			disp(c.tiledPipeline, bind(c.tiledLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), gx, gy)
		}
		return dstC
	}

	rowLen := hidden + len(pp.layers)*pp.pleDim
	xdFlat := make([]float32, 0, M*hidden)
	for r := range xs {
		if len(xs[r]) != rowLen {
			return nil, fmt.Errorf("gpu: the batched pass got a row of %d floats, want %d", len(xs[r]), rowLen)
		}
		xdFlat = append(xdFlat, xs[r][:hidden]...)
	}
	xd, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xdFlat), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	keepBuf(xd)
	// A Gemma 4 E-model's per-layer inputs, laid out [layer][row][pleDim] so one layer's rows bind as one range. A
	// layer's range starts on a 256-byte boundary, as a binding's offset must: pleStride floats a layer.
	var pleIn *wgpu.Buffer
	pleStride := (M*pp.pleDim + 63) &^ 63
	if pp.pleDim > 0 {
		flat := make([]float32, len(pp.layers)*pleStride)
		for l := range pp.layers {
			off := hidden + l*pp.pleDim
			for r := range xs {
				src := r
				if prefillDefectForTest == 4 {
					src = (r + 1) % M
				}
				copy(flat[l*pleStride+r*pp.pleDim:], xs[src][off:off+pp.pleDim])
			}
		}
		if pleIn, err = c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(flat), Usage: wgpu.BufferUsageStorage}); err != nil {
			return nil, err
		}
		keepBuf(pleIn)
	}

	// R10 prefill decomposition (gpu/prefill_prof.go): pt is a no-op (time.Time{}) whenever c.prefillProf is
	// nil, so profiling off costs one nil-check per call. Category boundaries are syncs (as in cudaResident's
	// profTic/profToc), so the category sum runs a bit over pipelined wall time; numbers in
	// docs/measurements/webgpu-prefill-decomp-2026-09-22.md. residualB and swigluB (elementwise) fold into
	// normsRope, the same "glue" role cuda/prefill.go's glueCat plays.
	// profBoundary closes one category. It FLUSHES the open encoder first: Poll only waits for submitted work,
	// so dispatches still in the encoder are invisible to it and the split would follow the 32-dispatch flush
	// cadence, not the categories. The extra flushes happen only when profiling is on.
	var pt time.Time
	profBoundary := func(cat prefillProfCat) {
		if c.prefillProf == nil {
			return
		}
		flushPasses()
		dispCount = 0
		c.prefillProfToc(cat, pt)
		pt = c.prefillProfTic()
	}
	pt = c.prefillProfTic()
	var sinks0, hs0 *wgpu.Buffer // the single-row attention kernel's unused sink bindings
	for i := range m.Layers {
		recycle() // the previous layer's buffers back to the free list (xd is not one: it is created above, outside storF)
		lw := &m.Layers[i]
		lp := &pp.layers[i]
		lhd, lnKV := hd, nKV
		if lp.hd != 0 {
			lhd, lnKV = lp.hd, lp.nKV
		}
		lkvDim := lnKV * lhd
		qRow := nH * lhd // floats in one row of q, and of the attention context
		kCache, vCache := lw.Attn.KCache.buf, lw.Attn.VCache.buf
		if lp.kvShared && prefillDefectForTest == 11 {
			kCache, vCache = m.Layers[0].Attn.KCache.buf, m.Layers[0].Attn.VCache.buf
		}
		attnNorm := lw.Attn.Norm.buf
		normQ := func(kp int) (*wgpu.Buffer, *wgpu.Buffer) { return rmsQuantB(xd, attnNorm, kp) }
		var qBias, kBias, vBias *wgpu.Buffer
		if lw.Attn.QBias != nil { // Qwen2 q/k/v bias: an int8 projection's is fused into its GEMM's epilogue, as decode's gemvBias is
			qBias, kBias, vBias = lw.Attn.QBias.buf, lw.Attn.KBias.buf, lw.Attn.VBias.buf
		}
		var q, k, v, aGate *wgpu.Buffer
		switch {
		case lp.kvShared:
			// The source layer wrote these rows' K and V earlier in the pass; this layer projects its query only.
			aq, as := normQ(lp.q.kp())
			q = projQB(aq, as, M, lp.q, qBias)
		case lp.kEqV:
			qk := sharedQ(normQ, []prefillProj{lp.q, lp.k}, []*wgpu.Buffer{qBias, kBias})
			q, k = qk[0], qk[1]
			if prefillDefectForTest == 10 {
				v = storF(M * lkvDim)
				cpy(k, 0, v, 0, uint64(M*lkvDim*4))
			} else {
				v = vNormB(k, lnKV, lhd) // of the raw K, before its own norm and rotation
			}
		default:
			qkv := sharedQ(normQ, []prefillProj{lp.q, lp.k, lp.v}, []*wgpu.Buffer{qBias, kBias, vBias})
			q, k, v = qkv[0], qkv[1], qkv[2]
			if lp.vNorm {
				v = vNormB(v, lnKV, lhd)
			}
		}
		if lw.Attn.QGate {
			qRaw := q
			qN := M * qRow
			q = storF(qN)
			aGate = storF(qN)
			p := uni([]uint32{uint32(qN), uint32(lhd), 0, 0})
			disp(c.deltaQSplitPipeline, bind(c.deltaQSplitLayout, qRaw, q, aGate, p), uint32(qN+63)/64, 1)
		}
		profBoundary(profGemm)
		if lw.Attn.QNorm != nil {
			qkNormB(q, lw.Attn.QNorm.buf, nH, lhd)
			if !lp.kvShared {
				qkNormB(k, lw.Attn.KNorm.buf, lnKV, lhd)
			}
		}
		ropeB(q, lw.Attn.InvFreq.buf, nH, lhd)
		if !lp.kvShared {
			ropeB(k, lw.Attn.InvFreq.buf, lnKV, lhd)
		}
		profBoundary(profNormsRope)
		if !lp.kvShared {
			// Bulk KV-cache write: positions are contiguous (positions[0]..positions[0]+M-1, residency.go), so k/v's packed
			// [M, kvDim] rows land at one contiguous range of the cache: a single copy per k/v instead of M.
			kvRow, kvRows := positions[0], M
			if prefillDefectForTest == 2 {
				kvRow, kvRows = positions[0]+1, M-1
			}
			if m.KVF16 {
				totalElems := kvRows * lkvDim
				totalWords := totalElems / 2
				p := uni([]uint32{uint32(totalElems), uint32(kvRow * lkvDim), 0, 0})
				disp(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, k, kCache, p), uint32(totalWords+63)/64, 1)
				disp(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, v, vCache, p), uint32(totalWords+63)/64, 1)
			} else {
				cpy(k, 0, kCache, uint64(kvRow*lkvDim*4), uint64(kvRows*lkvDim*4))
				cpy(v, 0, vCache, uint64(kvRow*lkvDim*4), uint64(kvRows*lkvDim*4))
			}
		}
		profBoundary(profKVWrite)
		// All rows' K/V are now in the cache; each row attends to its causal prefix (including earlier rows of this
		// block), the ordering DecodeTokenFusedBatched's parity gate already proves.
		ctxv := storF(M * qRow)
		if lp.local || lhd > attnWG || prefillAttnPerRow {
			// Decode's own single-row attention kernel, once a row, each row's q and context bound at its offset: the
			// batched kernels take neither a window's start nor a wide head.
			if qRow%64 != 0 && buildErr == nil {
				buildErr = fmt.Errorf("gpu: the batched pass declines a %d-float attention row: a row's range must start on a 256-byte boundary", qRow)
			}
			pl, ly := c.attnKernel(lhd, lkvDim, m.KVF16)
			if sinks0 == nil && buildErr == nil {
				var e error
				if sinks0, e = c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(make([]float32, max(nH, 4))), Usage: wgpu.BufferUsageStorage}); e != nil {
					buildErr = e
				} else {
					keepBuf(sinks0)
				}
				hs0 = uni([]uint32{0, 0, 0, 0})
			}
			for r := range M {
				pos := positions[0] + r
				first := 0 // a sliding-window layer attends its last window positions, as decoder.KVCache.WindowStart
				if lp.local && pp.window > 0 && prefillDefectForTest != 6 {
					first = max(0, pos+1-pp.window)
				}
				rp := uni([]uint32{uint32(nH), uint32(lnKV), uint32(lhd), uint32(pos + 1), uint32(first), uint32(nH / lnKV), f32bits(scale), 0})
				disp(pl, bindOff(ly, bindEntry{buf: q, off: uint32(r * qRow), n: uint32(qRow)}, whole(kCache), whole(vCache),
					bindEntry{buf: ctxv, off: uint32(r * qRow), n: uint32(qRow)}, whole(sinks0), whole(rp), whole(hs0)), uint32(nH), 1)
			}
		} else {
			// ONE dispatch, grid (nH, M), against attnBatchedKernel's chosen kernel (the same tiled-vs-plain preference as
			// attnKernel, attention.go); q/ctxv are read and written at each row's own offset inside the shader
			// (docs/completed/task-gpu-batched-prefill.md, Increment 1).
			attnPl, attnLy := c.attnBatchedKernel(lhd, lkvDim, m.KVF16)
			ap := uni([]uint32{uint32(nH), uint32(lnKV), uint32(lhd), uint32(positions[0]), uint32(nH / lnKV), f32bits(scale), uint32(M), 0})
			disp(attnPl, bind(attnLy, q, kCache, vCache, ctxv, ap), uint32(nH), uint32(M))
		}
		if aGate != nil {
			dispFlat(c.deltaAttnGatePipeline, c.deltaAttnGateLayout, M*qRow, ctxv, aGate)
		}
		profBoundary(profAttn)
		ctxQ, ctxS := quantPackedM(ctxv, lp.o.cols(), lp.o.kp())
		attnOut := projQB(ctxQ, ctxS, M, lp.o, nil)
		profBoundary(profGemm)
		if lp.postAttnNorm != nil && prefillDefectForTest != 7 {
			attnOut = rmsF32B(attnOut, lp.postAttnNorm)
		}
		xd = residualB(xd, attnOut) // in-place: xd += attnOut
		mlpNorm := lw.MLPNorm.buf
		gu := sharedQ(func(kp int) (*wgpu.Buffer, *wgpu.Buffer) { return rmsQuantB(xd, mlpNorm, kp) }, []prefillProj{lp.gate, lp.up}, []*wgpu.Buffer{nil, nil})
		gate, up := gu[0], gu[1]
		profBoundary(profGemm)
		width := lp.gate.rows() // the layer's own feed-forward width: a Gemma 4 E-model's layers do not share one
		if prefillDefectForTest == 9 && inter < width {
			width = inter
		}
		midQ, midS := gatedQuantB(gate, whole(up), width, lp.down.kp())
		profBoundary(profNormsRope)
		down := projQB(midQ, midS, M, lp.down, nil)
		profBoundary(profGemm)
		if lp.postMLPNorm != nil {
			down = rmsF32B(down, lp.postMLPNorm)
		}
		xd = residualB(xd, down) // in-place: xd += down
		if lp.pleGate.set() && prefillDefectForTest != 5 {
			// The per-layer-input branch: the residual gated down to pleDim, times this layer's slice of each row's tail
			// through the gated product, projected back, normed and added.
			hq, hs := quantPackedM(xd, hidden, lp.pleGate.kp())
			px := projQB(hq, hs, M, lp.pleGate, nil)
			rowsIn := bindEntry{buf: pleIn, off: uint32(i * pleStride), n: uint32(M * pp.pleDim)}
			pq, ps := gatedQuantB(px, rowsIn, pp.pleDim, lp.pleProj.kp())
			sub := rmsF32B(projQB(pq, ps, M, lp.pleProj, nil), lp.postPLENorm)
			xd = residualB(xd, sub)
		}
		if lp.scalar != 0 && lp.scalar != 1 && prefillDefectForTest != 8 {
			scaleB(xd, lp.scalar)
		}
		profBoundary(profNormsRope)
	}

	// Final norm + LM head on the LAST row ONLY (matches decoder/forwardn.go's
	// prefillLogits: the other M-1 rows' logits are never needed for prefill) — one
	// small copy to extract row M-1 from the packed buffer, then the same rms/
	// tiledProjB path at rowsM=1.
	xdLast := storF(hidden)
	cpy(xd, uint64((M-1)*hidden*4), xdLast, 0, uint64(hidden*4))
	// The final norm and its quantization are decode's own fused kernel on the one row.
	headKP := pp.lmHead.kp()
	fq, fs := storF(headKP/4), storF(1)
	{
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), uint32(headKP)})
		disp(c.rmsQuantPipeline, bind(c.rmsQuantLayout, xdLast, m.FinalNorm.buf, fq, fs, p), 1, 1)
	}
	logitsBuf := projQB(fq, fs, 1, pp.lmHead, nil)

	if buildErr != nil {
		return nil, buildErr
	}

	vocab := pp.lmHead.rows()
	stag, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	keepBuf(stag)
	if err := enc.TryCopyBufferToBuffer(logitsBuf, 0, stag, 0, uint64(vocab*4)); err != nil {
		return nil, err
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	status := wgpu.MapAsyncStatus(0)
	if err := stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(vocab*4), func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 map failed: %v", status)
	}
	out := make([]float32, vocab)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(vocab*4))))
	if err := stag.TryUnmap(); err != nil {
		return nil, err
	}
	return out, nil
}

// prefillDefectForTest plants one defect in the batched pass, for its gate's planted-defect arm (tests only): 1 rotates
// every row one position late, 2 writes every row's K/V one slot late, 3 gives an int4 q projection the k projection's
// scales, 4 binds each row's per-layer inputs to the next row's, 5 drops the per-layer-input branch, 6 lets a
// sliding-window layer attend its whole prefix, 7 skips the post-attention norm, 8 skips the layer scalar, 9 uses the
// model's feed-forward width on every layer, 10 skips the value norm on a K=V layer, 11 has a shared-K/V layer read the
// first layer's cache, 12 rotates the m-RoPE axis table by one pair, 13 gives each row the next row's three positions,
// 14 swaps height and width.
var prefillDefectForTest int

// prefillAttnPerRow makes the batched pass run decode's single-row attention kernel once a row in place of the batched
// attention kernel. Tests only. It needs a row of q (nH*hd floats) to be a multiple of 64 floats, so each row's
// binding offset is a multiple of 256 bytes.
var prefillAttnPerRow = false

// prefillQuantPerProj restores the batched prefill's quantization per projection (R-25, sharedQ), for the test that
// compares the two and the A/B that times them. Tests only.
var prefillQuantPerProj = false

// prefillSharedQuants counts the quantizations sharedQ shared across projections (a test reads it to see R-25 ran).
var prefillSharedQuants atomic.Int64

// prefillBufFresh and prefillBufPoison are tests-only switches on the pass's buffer recycling (R-25, storF): fresh
// buffers for every call, as before, and NaN-filled recycled buffers. prefillBufReused counts recycled buffers.
var (
	prefillBufFresh  = false
	prefillBufPoison = false
	prefillBufReused atomic.Int64
)
