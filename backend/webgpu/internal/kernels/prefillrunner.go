//go:build gpu

package kernels

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"slices"
	"sync/atomic"

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
const ropeBatchedShaderWGSL = ropeCosSinWGSL + `
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
	if !AttnKeysDisabled && attnKeysEligible(hd, kvDim, false, false) {
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

func asPrefillProj(w DecodeWeight) (prefillProj, bool) {
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
	MropeAxis func(half int) []int
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
func (pp prefillProjs) Int4() bool {
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
func (pp prefillProjs) Int8Bias(m ModelW) bool {
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
func runModelToModelW(rm *RunModel, hd int) (ModelW, bool) {
	if rm.SlidingWindow != 0 {
		return ModelW{}, false
	}
	mw, pp, ok := RunModelToPrefill(rm, hd)
	if !ok || pp.Int4() || pp.gemma() {
		return ModelW{}, false
	}
	return mw, true
}

// RunModelToPrefill narrows a resident runModel to the dense shapes the batched pass accepts, returning ok=false if
// the model uses anything outside them: MoE (Gemma 4's parallel block included), MLA, SSM, DeltaNet, int8 KV,
// whole-vector QK-norm, post-only norms, partial RoPE, a rope scale, an attention sink, an
// o-proj bias. Each projection may be int8 or int4; the ModelW's own
// projection fields are set for the int8 ones only. The Gemma shape (prefillLayerProj's own fields) is accepted.
//
// q/k/v bias is accepted, but callers must also check prefillProjs.int8Bias against the backend: an int8 projection's
// fused bias epilogue is proven bit-exact on Vulkan only and is declined elsewhere until proven there. See
// BuildResident's prefillLast gate in residency.go; history in docs/code-notes/gpu.md#runModelToModelW. An int4
// projection adds its bias in a separate step, as decode does, on every backend.
//
// The result is a zero-copy view over rm's buffers; callers must not Close it.
// hd is the model's head dimension, used to tell genuine partial RoPE (declined) from a half set explicitly to hd/2.
func RunModelToPrefill(rm *RunModel, hd int) (ModelW, prefillProjs, bool) {
	none := func() (ModelW, prefillProjs, bool) { return ModelW{}, prefillProjs{}, false }
	if rm.Moe != nil || rm.G4moe || rm.Mla != nil || rm.Mamba != nil || rm.Dnet != nil || rm.KvI8 || rm.QKNormWhole || rm.PostOnlyNorm || rm.ActG32 || rm.AttnTemp != nil || rm.Pairwise || rm.LayerNorm || rm.Parallel {
		return none()
	}
	view := func(b *wgpu.Buffer) *DeviceBuffer {
		if b == nil {
			return nil
		}
		return &DeviceBuffer{buf: b}
	}
	layers := make([]LayerW, len(rm.Layers))
	pp := prefillProjs{layers: make([]prefillLayerProj, len(rm.Layers)), pleDim: rm.PleDim}
	anyGeom := slices.ContainsFunc(rm.Layers, func(l RunLayer) bool { return l.Ghd != 0 })
	if !anyGeom && rm.RopeHalf != 0 && rm.RopeHalf != hd/2 {
		return none() // the pass's rotary kernel turns every pair of a head
	}
	for i := range rm.Layers {
		l := &rm.Layers[i]
		if l.OBias != nil || l.HasSink || (l.RopeScale != 0 && l.RopeScale != 1) ||
			l.IsMoE || l.G4moe != nil || l.ShGate != nil ||
			l.MlaQA != nil || l.MlaQB != nil || l.MlaQ != nil || l.MlaKVA != nil || l.MlaO != nil ||
			l.IsMamba || l.NemoKind != nemoNone || l.IsDeltaNet {
			return none()
		}
		if l.Ghd != 0 && l.Ghalf*2 != l.Ghd {
			return none()
		}
		if l.KvShared {
			if l.KNorm != nil && l.QNorm == nil {
				return none()
			}
		} else if (l.QNorm == nil) != (l.KNorm == nil) {
			return none()
		}
		lp := prefillLayerProj{hd: l.Ghd, nKV: l.GnKV, kEqV: l.GKEqV, vNorm: l.VNorm, kvShared: l.KvShared, local: l.IsLocal,
			postAttnNorm: l.PostAttnNorm, postMLPNorm: l.PostMLPNorm, scalar: l.LayerScalar, postPLENorm: l.PostPLENorm}
		ok := true
		need := func(w DecodeWeight) prefillProj {
			pj, pok := asPrefillProj(w)
			ok = ok && pok
			return pj
		}
		lp.q, lp.o, lp.gate, lp.up, lp.down = need(l.Q), need(l.O), need(l.Gate), need(l.Up), need(l.Down)
		if !l.KvShared {
			lp.k = need(l.K)
			if !l.GKEqV {
				lp.v = need(l.V)
			}
		}
		if l.PleGate != nil {
			lp.pleGate, lp.pleProj = need(l.PleGate), need(l.PleProj)
			ok = ok && l.PostPLENorm != nil && rm.PleDim > 0
		}
		if !ok {
			return none() // a projection that is neither int8 nor int4: no multi-row kernel for it
		}
		if l.IsLocal {
			pp.window = rm.SlidingWindow
		}
		pp.layers[i] = lp
		layers[i] = LayerW{
			Attn: AttnWeights{
				Norm: view(l.AttnNorm), QProj: lp.q.w8, KProj: lp.k.w8, VProj: lp.v.w8, OProj: lp.o.w8,
				InvFreq: view(l.InvFreq), KCache: view(l.KCache), VCache: view(l.VCache),
				QBias: view(l.QBias), KBias: view(l.KBias), VBias: view(l.VBias),
				QNorm: view(l.QNorm), KNorm: view(l.KNorm),
				QGate: l.QGate,
			},
			MLPNorm: view(l.MlpNorm),
			Gate:    lp.gate.w8, Up: lp.up.w8, Down: lp.down.w8,
		}
	}
	lmHead, lmOK := asPrefillProj(rm.LmHead)
	if !lmOK {
		return none()
	}
	pp.lmHead = lmHead
	return ModelW{
		Layers:    layers,
		FinalNorm: view(rm.FinalNorm),
		LMHead:    lmHead.w8,
		GatedGELU: rm.GatedGELU,
		KVF16:     rm.KvF16,
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
	if err := c.ensurePrefillRows(M, m, pp, mrope, hd, positions); err != nil {
		return nil, err
	}
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	pa := &prefillPass{
		c: c, m: m, pp: pp, M: M, hidden: hidden, nH: nH, nKV: nKV, hd: hd, inter: inter, positions: positions,
		eps: eps, scale: scale, addOne: addOne, enc: enc, free: map[int][]*wgpu.Buffer{}, axisBufs: map[int]*wgpu.Buffer{},
	}
	defer pa.release()
	if err := pa.upload(xs, mrope); err != nil {
		return nil, err
	}
	pa.pt = c.prefillProfTic()
	for i := range m.Layers {
		pa.layer(i)
	}
	return pa.logits()
}

// ensurePrefillRows checks a pass's shape and compiles every pipeline it will bind.
func (c *Context) ensurePrefillRows(M int, m ModelW, pp prefillProjs, mrope [][3]int, hd int, positions []int) error {
	if M == 0 {
		return fmt.Errorf("gpu: PrefillLastW8A8 M=0")
	}
	if len(positions) != M {
		return fmt.Errorf("gpu: PrefillLastW8A8 %d rows but %d positions", M, len(positions))
	}
	if M > 65535 {
		return fmt.Errorf("gpu: the batched pass takes at most 65535 rows a call, got %d", M)
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled, c.ensureTiledBias, c.ensurePrefillBatched, c.ensureQKNorm, c.ensureFuse, c.ensureFusedRows} {
		if err := f(); err != nil {
			return err
		}
	}
	if m.hasQGate() {
		if err := c.ensureDeltaQSplit(); err != nil {
			return err
		}
		if err := c.ensureDeltaAttnGate(); err != nil {
			return err
		}
	}
	if mrope != nil {
		if len(mrope) != M || pp.MropeAxis == nil {
			return fmt.Errorf("gpu: the batched pass got %d rows of rotary positions for %d rows (axis table set: %v)", len(mrope), M, pp.MropeAxis != nil)
		}
		if err := c.ensureMRoPERows(); err != nil {
			return err
		}
	}
	if pp.gemma() {
		for _, f := range []func() error{c.ensureVNorm, c.ensureScaleVec} {
			if err := f(); err != nil {
				return err
			}
		}
	}
	if hd > AttnWG || slices.ContainsFunc(pp.layers, func(l prefillLayerProj) bool { return l.hd > AttnWG }) {
		if err := c.ensureAttnWide(); err != nil { // a wide head runs decode's own attention kernel once a row
			return err
		}
	}
	if pp.Int4() {
		if err := c.ensureW4Multi(); err != nil {
			return err
		}
		if c.biasRowsPipeline == nil {
			var err error
			if _, c.biasRowsPipeline, c.biasRowsLayout, err = c.mkPipeline("bias-rows", biasRowsShaderWGSL); err != nil {
				return err
			}
		}
	}
	return nil
}

// upload writes the pass's inputs: each row's m-RoPE positions, the packed [M, hidden] residual stream, and a Gemma 4
// E-model's per-layer inputs.
func (pa *prefillPass) upload(xs [][]float32, mrope [][3]int) error {
	c, M, pp, hidden := pa.c, pa.M, pa.pp, pa.hidden
	if mrope != nil {
		flat := make([]uint32, 0, 3*M)
		for r := range mrope {
			src := mrope[r]
			if PrefillDefectForTest == 13 {
				src = mrope[(r+1)%M]
			}
			flat = append(flat, uint32(src[0]), uint32(src[1]), uint32(src[2]))
		}
		b, e := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(flat), Usage: wgpu.BufferUsageStorage})
		if e != nil {
			return e
		}
		pa.keepBuf(b)
		pa.mposBuf = b
	}

	rowLen := hidden + len(pp.layers)*pp.pleDim
	xdFlat := make([]float32, 0, M*hidden)
	for r := range xs {
		if len(xs[r]) != rowLen {
			return fmt.Errorf("gpu: the batched pass got a row of %d floats, want %d", len(xs[r]), rowLen)
		}
		xdFlat = append(xdFlat, xs[r][:hidden]...)
	}
	xd, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xdFlat), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	pa.keepBuf(xd)
	pa.xd = xd
	// A Gemma 4 E-model's per-layer inputs, laid out [layer][row][pleDim] so one layer's rows bind as one range. A
	// layer's range starts on a 256-byte boundary, as a binding's offset must: pleStride floats a layer.
	pa.pleStride = (M*pp.pleDim + 63) &^ 63
	if pp.pleDim > 0 {
		flat := make([]float32, len(pp.layers)*pa.pleStride)
		for l := range pp.layers {
			off := hidden + l*pp.pleDim
			for r := range xs {
				src := r
				if PrefillDefectForTest == 4 {
					src = (r + 1) % M
				}
				copy(flat[l*pa.pleStride+r*pp.pleDim:], xs[src][off:off+pp.pleDim])
			}
		}
		if pa.pleIn, err = c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(flat), Usage: wgpu.BufferUsageStorage}); err != nil {
			return err
		}
		pa.keepBuf(pa.pleIn)
	}
	return nil
}

// layer encodes one layer over every row: the q/k/v projections, their norms and rotation, the KV write, attention,
// the output projection and the feed-forward.
func (pa *prefillPass) layer(i int) {
	pa.recycle() // the previous layer's buffers back to the free list (xd is not one: it is created in upload, outside storF)
	l := pa.layerGeom(i)
	q, k, v, aGate := pa.layerQKV(l)
	pa.profBoundary(profGemm)
	pa.layerRopeKV(l, q, k, v)
	ctxv := pa.layerAttention(l, q, aGate)
	pa.layerOutFFN(l, ctxv)
}

func (pa *prefillPass) layerGeom(i int) prefillLayer {
	m := pa.m
	lw := &m.Layers[i]
	lp := &pa.pp.layers[i]
	lhd, lnKV := pa.hd, pa.nKV
	if lp.hd != 0 {
		lhd, lnKV = lp.hd, lp.nKV
	}
	lkvDim := lnKV * lhd
	qRow := pa.nH * lhd // floats in one row of q, and of the attention context
	kCache, vCache := lw.Attn.KCache.buf, lw.Attn.VCache.buf
	if lp.kvShared && PrefillDefectForTest == 11 {
		kCache, vCache = m.Layers[0].Attn.KCache.buf, m.Layers[0].Attn.VCache.buf
	}
	return prefillLayer{i: i, lw: lw, lp: lp, lhd: lhd, lnKV: lnKV, lkvDim: lkvDim, qRow: qRow, kCache: kCache, vCache: vCache}
}

// layerQKV projects the layer's normed input to q, k and v (q only on a KV-shared layer), and splits a gated query.
func (pa *prefillPass) layerQKV(l prefillLayer) (q, k, v, aGate *wgpu.Buffer) {
	c, M, lw, lp, lhd, lnKV, lkvDim, qRow := pa.c, pa.M, l.lw, l.lp, l.lhd, l.lnKV, l.lkvDim, l.qRow
	attnNorm := lw.Attn.Norm.buf
	normQ := func(kp int) (*wgpu.Buffer, *wgpu.Buffer) { return pa.rmsQuantB(pa.xd, attnNorm, kp) }
	var qBias, kBias, vBias *wgpu.Buffer
	if lw.Attn.QBias != nil { // Qwen2 q/k/v bias: an int8 projection's is fused into its GEMM's epilogue, as decode's gemvBias is
		qBias, kBias, vBias = lw.Attn.QBias.buf, lw.Attn.KBias.buf, lw.Attn.VBias.buf
	}
	switch {
	case lp.kvShared:
		// The source layer wrote these rows' K and V earlier in the pass; this layer projects its query only.
		aq, as := normQ(lp.q.kp())
		q = pa.projQB(aq, as, M, lp.q, qBias)
	case lp.kEqV:
		qk := pa.sharedQ(normQ, []prefillProj{lp.q, lp.k}, []*wgpu.Buffer{qBias, kBias})
		q, k = qk[0], qk[1]
		if PrefillDefectForTest == 10 {
			v = pa.storF(M * lkvDim)
			pa.cpy(k, 0, v, 0, uint64(M*lkvDim*4))
		} else {
			v = pa.vNormB(k, lnKV, lhd) // of the raw K, before its own norm and rotation
		}
	default:
		qkv := pa.sharedQ(normQ, []prefillProj{lp.q, lp.k, lp.v}, []*wgpu.Buffer{qBias, kBias, vBias})
		q, k, v = qkv[0], qkv[1], qkv[2]
		if lp.vNorm {
			v = pa.vNormB(v, lnKV, lhd)
		}
	}
	if lw.Attn.QGate {
		qRaw := q
		qN := M * qRow
		q = pa.storF(qN)
		aGate = pa.storF(qN)
		p := pa.uni([]uint32{uint32(qN), uint32(lhd), 0, 0})
		pa.disp(c.deltaQSplitPipeline, pa.bind(c.deltaQSplitLayout, qRaw, q, aGate, p), uint32(qN+63)/64, 1)
	}
	return q, k, v, aGate
}

// layerRopeKV norms and rotates q and k, then writes every row's k and v into the cache (not on a KV-shared layer).
func (pa *prefillPass) layerRopeKV(l prefillLayer, q, k, v *wgpu.Buffer) {
	c, m, M, positions, nH, lw, lp, lhd, lnKV, lkvDim, kCache, vCache := pa.c, pa.m, pa.M, pa.positions, pa.nH, l.lw, l.lp, l.lhd, l.lnKV, l.lkvDim, l.kCache, l.vCache
	if lw.Attn.QNorm != nil {
		pa.qkNormB(q, lw.Attn.QNorm.buf, nH, lhd)
		if !lp.kvShared {
			pa.qkNormB(k, lw.Attn.KNorm.buf, lnKV, lhd)
		}
	}
	pa.ropeB(q, lw.Attn.InvFreq.buf, nH, lhd)
	if !lp.kvShared {
		pa.ropeB(k, lw.Attn.InvFreq.buf, lnKV, lhd)
	}
	pa.profBoundary(profNormsRope)
	if !lp.kvShared {
		// Bulk KV-cache write: positions are contiguous (positions[0]..positions[0]+M-1, residency.go), so k/v's packed
		// [M, kvDim] rows land at one contiguous range of the cache: a single copy per k/v instead of M.
		kvRow, kvRows := positions[0], M
		if PrefillDefectForTest == 2 {
			kvRow, kvRows = positions[0]+1, M-1
		}
		if m.KVF16 {
			totalElems := kvRows * lkvDim
			totalWords := totalElems / 2
			p := pa.uni([]uint32{uint32(totalElems), uint32(kvRow * lkvDim), 0, 0})
			pa.disp(c.kvStoreF16Pipeline, pa.bind(c.kvStoreF16Layout, k, kCache, p), uint32(totalWords+63)/64, 1)
			pa.disp(c.kvStoreF16Pipeline, pa.bind(c.kvStoreF16Layout, v, vCache, p), uint32(totalWords+63)/64, 1)
		} else {
			pa.cpy(k, 0, kCache, uint64(kvRow*lkvDim*4), uint64(kvRows*lkvDim*4))
			pa.cpy(v, 0, vCache, uint64(kvRow*lkvDim*4), uint64(kvRows*lkvDim*4))
		}
	}
	pa.profBoundary(profKVWrite)
}

// layerAttention runs every row's attention over its causal prefix and returns the packed [M, nH*hd] context, gated
// when the layer has a query gate.
func (pa *prefillPass) layerAttention(l prefillLayer, q, aGate *wgpu.Buffer) *wgpu.Buffer {
	c, m, pp, M, positions, nH, scale, lp, lhd, lnKV, lkvDim, qRow, kCache, vCache := pa.c, pa.m, pa.pp, pa.M, pa.positions, pa.nH, pa.scale, l.lp, l.lhd, l.lnKV, l.lkvDim, l.qRow, l.kCache, l.vCache
	// All rows' K/V are now in the cache; each row attends to its causal prefix (including earlier rows of this
	// block), the ordering DecodeTokenFusedBatched's parity gate already proves.
	ctxv := pa.storF(M * qRow)
	// A sliding-window layer or a wide head has no hand-written batched kernel. Where decode's choice for the layer is
	// a key-split kernel, its rows form runs every row in one dispatch; otherwise decode's kernel runs once a row.
	window := 0
	if lp.local && PrefillDefectForTest != 6 {
		window = pp.window
	}
	rowsForm := -1
	if (lp.local || lhd > AttnWG) && !PrefillAttnPerRow && !PrefillAttnRowsOff && !AttnKeysDisabled {
		switch {
		case attnKeysEligible(lhd, lkvDim, m.KVF16, false):
			rowsForm = 0
		case attnKeysWideEligible(lhd, lkvDim, m.KVF16):
			rowsForm = 1
		}
	}
	if rowsForm >= 0 {
		if pa.buildErr == nil {
			pa.buildErr = c.ensureAttnKeysRows()
		}
		pa.ensureSinks0()
		f := c.attnKeysRows[rowsForm]
		rp := pa.uni([]uint32{uint32(nH), uint32(lnKV), uint32(lhd), uint32(positions[0] + 1), 0, uint32(nH / lnKV), F32bits(scale), uint32(window)})
		if pa.buildErr == nil {
			pa.disp(f.Pl, pa.bind(f.Layout, q, kCache, vCache, ctxv, pa.sinks0, rp, pa.hs0), uint32(nH), uint32(M))
		}
	} else if lp.local || lhd > AttnWG || PrefillAttnPerRow {
		// Decode's own single-row attention kernel, once a row, each row's q and context bound at its offset: the
		// batched kernels take neither a window's start nor a wide head.
		if qRow%64 != 0 && pa.buildErr == nil {
			pa.buildErr = fmt.Errorf("gpu: the batched pass declines a %d-float attention row: a row's range must start on a 256-byte boundary", qRow)
		}
		pl, ly := c.attnKernel(lhd, lkvDim, m.KVF16)
		pa.ensureSinks0()
		for r := range M {
			pos := positions[0] + r
			first := 0 // a sliding-window layer attends its last window positions, as decoder.KVCache.WindowStart
			if window > 0 {
				first = max(0, pos+1-window)
			}
			rp := pa.uni([]uint32{uint32(nH), uint32(lnKV), uint32(lhd), uint32(pos + 1), uint32(first), uint32(nH / lnKV), F32bits(scale), 0})
			pa.disp(pl, pa.bindOff(ly, prefillBindEntry{buf: q, off: uint32(r * qRow), n: uint32(qRow)}, whole(kCache), whole(vCache),
				prefillBindEntry{buf: ctxv, off: uint32(r * qRow), n: uint32(qRow)}, whole(pa.sinks0), whole(rp), whole(pa.hs0)), uint32(nH), 1)
		}
	} else {
		// ONE dispatch, grid (nH, M), against attnBatchedKernel's chosen kernel (the same tiled-vs-plain preference as
		// attnKernel, attention.go); q/ctxv are read and written at each row's own offset inside the shader.
		attnPl, attnLy := c.attnBatchedKernel(lhd, lkvDim, m.KVF16)
		ap := pa.uni([]uint32{uint32(nH), uint32(lnKV), uint32(lhd), uint32(positions[0]), uint32(nH / lnKV), F32bits(scale), uint32(M), 0})
		pa.disp(attnPl, pa.bind(attnLy, q, kCache, vCache, ctxv, ap), uint32(nH), uint32(M))
	}
	if aGate != nil {
		pa.dispFlat(c.deltaAttnGatePipeline, c.deltaAttnGateLayout, M*qRow, ctxv, aGate)
	}
	pa.profBoundary(profAttn)
	return ctxv
}

// ensureSinks0 creates, once a pass, the zero sinks and uniform the single-row attention kernels bind unused.
func (pa *prefillPass) ensureSinks0() {
	if pa.sinks0 == nil && pa.buildErr == nil {
		var e error
		if pa.sinks0, e = pa.c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(make([]float32, max(pa.nH, 4))), Usage: wgpu.BufferUsageStorage}); e != nil {
			pa.buildErr = e
		} else {
			pa.keepBuf(pa.sinks0)
		}
		pa.hs0 = pa.uni([]uint32{0, 0, 0, 0})
	}
}

// layerOutFFN projects the attention context out and adds it to the residual stream, then runs the feed-forward, a
// Gemma 4 E-model's per-layer-input branch and the layer's output scalar.
func (pa *prefillPass) layerOutFFN(l prefillLayer, ctxv *wgpu.Buffer) {
	M, pp, hidden, inter, i, lw, lp := pa.M, pa.pp, pa.hidden, pa.inter, l.i, l.lw, l.lp
	ctxQ, ctxS := pa.quantPackedM(ctxv, lp.o.cols(), lp.o.kp())
	attnOut := pa.projQB(ctxQ, ctxS, M, lp.o, nil)
	pa.profBoundary(profGemm)
	if lp.postAttnNorm != nil && PrefillDefectForTest != 7 {
		attnOut = pa.rmsF32B(attnOut, lp.postAttnNorm)
	}
	pa.xd = pa.residualB(pa.xd, attnOut) // in-place: xd += attnOut
	mlpNorm := lw.MLPNorm.buf
	gu := pa.sharedQ(func(kp int) (*wgpu.Buffer, *wgpu.Buffer) { return pa.rmsQuantB(pa.xd, mlpNorm, kp) }, []prefillProj{lp.gate, lp.up}, []*wgpu.Buffer{nil, nil})
	gate, up := gu[0], gu[1]
	pa.profBoundary(profGemm)
	width := lp.gate.rows() // the layer's own feed-forward width: a Gemma 4 E-model's layers do not share one
	if PrefillDefectForTest == 9 && inter < width {
		width = inter
	}
	midQ, midS := pa.gatedQuantB(gate, whole(up), width, lp.down.kp())
	pa.profBoundary(profNormsRope)
	down := pa.projQB(midQ, midS, M, lp.down, nil)
	pa.profBoundary(profGemm)
	if lp.postMLPNorm != nil {
		down = pa.rmsF32B(down, lp.postMLPNorm)
	}
	pa.xd = pa.residualB(pa.xd, down) // in-place: xd += down
	if lp.pleGate.set() && PrefillDefectForTest != 5 {
		// The per-layer-input branch: the residual gated down to pleDim, times this layer's slice of each row's tail
		// through the gated product, projected back, normed and added.
		hq, hs := pa.quantPackedM(pa.xd, hidden, lp.pleGate.kp())
		px := pa.projQB(hq, hs, M, lp.pleGate, nil)
		rowsIn := prefillBindEntry{buf: pa.pleIn, off: uint32(i * pa.pleStride), n: uint32(M * pp.pleDim)}
		pq, ps := pa.gatedQuantB(px, rowsIn, pp.pleDim, lp.pleProj.kp())
		sub := pa.rmsF32B(pa.projQB(pq, ps, M, lp.pleProj, nil), lp.postPLENorm)
		pa.xd = pa.residualB(pa.xd, sub)
	}
	if lp.scalar != 0 && lp.scalar != 1 && PrefillDefectForTest != 8 {
		pa.scaleB(pa.xd, lp.scalar)
	}
	pa.profBoundary(profNormsRope)
}

// logits runs the final norm and the LM head on the last row, submits the pass and reads the logits back.
func (pa *prefillPass) logits() ([]float32, error) {
	c, m, pp, M, hidden := pa.c, pa.m, pa.pp, pa.M, pa.hidden
	// Final norm + LM head on the LAST row ONLY (matches decoder/forwardn.go's
	// prefillLogits: the other M-1 rows' logits are never needed for prefill) — one
	// small copy to extract row M-1 from the packed buffer, then the same rms/
	// tiledProjB path at rowsM=1.
	xdLast := pa.storF(hidden)
	pa.cpy(pa.xd, uint64((M-1)*hidden*4), xdLast, 0, uint64(hidden*4))
	// The final norm and its quantization are decode's own fused kernel on the one row.
	headKP := pp.lmHead.kp()
	fq, fs := pa.storF(headKP/4), pa.storF(1)
	{
		p := pa.uni([]uint32{uint32(hidden), F32bits(pa.eps), BoolU32(pa.addOne), uint32(headKP)})
		pa.disp(c.rmsQuantPipeline, pa.bind(c.rmsQuantLayout, xdLast, m.FinalNorm.buf, fq, fs, p), 1, 1)
	}
	logitsBuf := pa.projQB(fq, fs, 1, pp.lmHead, nil)

	if pa.buildErr != nil {
		return nil, pa.buildErr
	}

	vocab := pp.lmHead.rows()
	stag, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	pa.keepBuf(stag)
	if err := pa.enc.TryCopyBufferToBuffer(logitsBuf, 0, stag, 0, uint64(vocab*4)); err != nil {
		return nil, err
	}
	cmd, err := pa.enc.TryFinish(nil)
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

// PrefillDefectForTest plants one defect in the batched pass, for its gate's planted-defect arm (tests only): 1 rotates
// every row one position late, 2 writes every row's K/V one slot late, 3 gives an int4 q projection the k projection's
// scales, 4 binds each row's per-layer inputs to the next row's, 5 drops the per-layer-input branch, 6 lets a
// sliding-window layer attend its whole prefix, 7 skips the post-attention norm, 8 skips the layer scalar, 9 uses the
// model's feed-forward width on every layer, 10 skips the value norm on a K=V layer, 11 has a shared-K/V layer read the
// first layer's cache, 12 rotates the m-RoPE axis table by one pair, 13 gives each row the next row's three positions,
// 14 swaps height and width.
var PrefillDefectForTest int

// PrefillAttnRowsOff makes the batched pass run decode's kernel once a row where it would run a key-split kernel's rows
// form, for the test that compares the two and the profile that times them. Tests only.
var PrefillAttnRowsOff = false

// PrefillAttnPerRow makes the batched pass run decode's single-row attention kernel once a row in place of the batched
// attention kernel. Tests only. It needs a row of q (nH*hd floats) to be a multiple of 64 floats, so each row's
// binding offset is a multiple of 256 bytes.
var PrefillAttnPerRow = false

// PrefillQuantPerProj restores the batched prefill's quantization per projection (sharedQ), for the test that
// compares the two and the A/B that times them. Tests only.
var PrefillQuantPerProj = false

// PrefillSharedQuants counts the quantizations sharedQ shared across projections (a test reads it to see the sharing ran).
var PrefillSharedQuants atomic.Int64

// prefillBufFresh and prefillBufPoison are tests-only switches on the pass's buffer recycling (storF): fresh
// buffers for every call, and NaN-filled recycled buffers. prefillBufReused counts recycled buffers.
var (
	PrefillBufFresh  = false
	prefillBufPoison = false
	prefillBufReused atomic.Int64
)

// PrefillBufSeenForTest (when set; tests only) is told the size of every device buffer a batched pass creates (recycled ones are not
// created again), so a test can hold PrefillScratchBytes to the pass.
var PrefillBufSeenForTest func(bytes int64)
