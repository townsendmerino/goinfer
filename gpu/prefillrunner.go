//go:build gpu

package gpu

import (
	"fmt"
	"sync/atomic"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// rmsnormBatchedShaderWGSL is rmsnormShaderWGSL (layer.go) widened to M rows in one
// dispatch: grid (1, M) instead of (1, 1), each workgroup owns row wid.y and reduces
// only that row (workgroupBarrier never crosses rows), so the per-row math and
// reduction order are identical to calling the M=1 kernel M times — bit-identical
// by construction, gated by TestRMSNormBatched_parity.
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

// ropeBatchedShaderWGSL is ropeShaderWGSL (attention.go) widened to M rows in one
// dispatch: grid (ceil(heads*half/64), M) instead of ropeShaderWGSL's one dispatch
// per row with a fixed scalar pos. Positions are always contiguous within one
// PrefillLastW8A8 call (positions[r] = start+r, residency.go's prefillLast closure),
// so row r's position is recovered as p.start+row — the same theta/cos/sin math per
// (row, head, d) as calling the M=1 kernel M times, bit-identical by construction,
// gated by TestRoPEBatched_parity.
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

// attnBatchedShaderWGSL is attnShaderWGSL (attention.go) widened to M query rows in
// one dispatch — grid (nH, M) instead of one dispatch per row with a fixed nKeys.
// Each row's causal bound is basePos+row+1 (positions are always contiguous within
// one PrefillLastW8A8 call — positions[r]=start+r, residency.go). No sink support:
// runModelToModelW already declines any model with FeatAttnSink. Not bit-identical
// to calling attnShaderWGSL M times in general (independent per-row reductions, so
// in practice it likely is) — gated on cosine/maxAbs like every other attention
// kernel pair in this file, not a bit-exact claim.
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

// attnKeysBatchedShaderWGSL is attnKeysShaderWGSL (attention.go) widened to M query
// rows the same way attnBatchedShaderWGSL widens the plain kernel — grid (nH, M),
// each row's own causal tile loop from key 0 to basePos+row (inclusive). Selected
// whenever attnKeysEligible(hd, kvDim, false, false) (attention.go), matching
// attnKernel's own preference for the tiled/key-split decomposition — most real
// dense architectures (hd and kvDim both multiples of 4) take this path, not the
// plain kernel above.
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

// ensurePrefillBatched lazily compiles the two batched-row kernels PrefillLastW8A8 needs beyond
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

// runModelToModelW narrows a resident runModel to the plain dense-W8A8 ModelW
// shape PrefillLastW8A8 accepts, returning ok=false if the model uses anything
// outside it (MoE, MLA, SSM, DeltaNet, sliding window, int8 KV, partial RoPE).
//
// q/k/v bias is accepted, but callers must also check ModelW.hasBias() against
// the backend: bias prefill is proven bit-exact on Vulkan only and is declined
// elsewhere until proven there. See BuildResident's prefillLast gate in
// residency.go; history in docs/code-notes/gpu.md#runModelToModelW.
//
// The result is a zero-copy view over rm's buffers; callers must not Close it.
// hd is the head dimension, used only to tell genuine partial RoPE (declined)
// from ropeHalf set explicitly to hd/2.
func runModelToModelW(rm *runModel, hd int) (ModelW, bool) {
	if rm.moe != nil || rm.mla != nil || rm.mamba != nil || rm.dnet != nil ||
		rm.kvI8 || rm.slidingWindow != 0 ||
		(rm.ropeHalf != 0 && rm.ropeHalf != hd/2) ||
		hd > attnWG { // ensureAttnWide is never in PrefillLastW8A8's ensure-list (attnKernel
		// would otherwise dispatch a nil c.attnWidePipeline); dnet/mamba above already
		// exclude the one real hd=256 family (Gated-DeltaNet hybrids), so this only
		// guards a theoretical wide-head-dim dense model, not a real gap.
		return ModelW{}, false
	}
	view := func(b *wgpu.Buffer) *DeviceBuffer {
		if b == nil {
			return nil
		}
		return &DeviceBuffer{buf: b}
	}
	layers := make([]LayerW, len(rm.layers))
	for i := range rm.layers {
		l := &rm.layers[i]
		if (l.qNorm == nil) != (l.kNorm == nil) ||
			l.oBias != nil || l.postAttnNorm != nil || l.postMLPNorm != nil ||
			l.hasSink || l.isLocal || (l.ropeScale != 0 && l.ropeScale != 1) ||
			l.ghd != 0 || l.gnKV != 0 || l.ghalf != 0 || l.gKEqV ||
			l.isMoE || l.shGate != nil ||
			l.mlaQA != nil || l.mlaQB != nil || l.mlaQ != nil || l.mlaKVA != nil || l.mlaO != nil ||
			l.isMamba || l.nemoKind != nemoNone || l.isDeltaNet {
			return ModelW{}, false
		}
		q, qOK := l.q.(*ResidentW8A8)
		k, kOK := l.k.(*ResidentW8A8)
		v, vOK := l.v.(*ResidentW8A8)
		o, oOK := l.o.(*ResidentW8A8)
		gate, gOK := l.gate.(*ResidentW8A8)
		up, uOK := l.up.(*ResidentW8A8)
		down, dOK := l.down.(*ResidentW8A8)
		if !qOK || !kOK || !vOK || !oOK || !gOK || !uOK || !dOK {
			return ModelW{}, false // a W4A8 (or other non-W8A8) projection — no tiled GEMM for it
		}
		layers[i] = LayerW{
			Attn: AttnWeights{
				Norm: view(l.attnNorm), QProj: q, KProj: k, VProj: v, OProj: o,
				InvFreq: view(l.invFreq), KCache: view(l.kCache), VCache: view(l.vCache),
				QBias: view(l.qBias), KBias: view(l.kBias), VBias: view(l.vBias),
				QNorm: view(l.qNorm), KNorm: view(l.kNorm),
				QGate: l.qGate,
			},
			MLPNorm: view(l.mlpNorm),
			Gate:    gate, Up: up, Down: down,
		}
	}
	lmHead, lmOK := rm.lmHead.(*ResidentW8A8)
	if !lmOK {
		return ModelW{}, false
	}
	return ModelW{
		Layers:    layers,
		FinalNorm: view(rm.finalNorm),
		LMHead:    lmHead,
		GatedGELU: rm.gatedGELU,
		KVF16:     rm.kvF16,
	}, true
}

// PrefillLastW8A8 is the batched-prefill analogue of DecodeTokenFusedBatched: it
// runs M prompt-token rows through the dense W8A8 layers with weight-heavy
// PROJECTIONS as ONE tiled GEMM each (weight streamed once across all M rows, via
// the unbounded-M kernel in gemm.go — DP4A-accelerated when Context.hasDP4A), and
// the cheap per-row ops (rmsnorm+quant, RoPE, KV-store, attention, SwiGLU,
// residual) looped over the rows, mirroring DecodeTokenFusedBatched's exact
// write-all-KV-then-attend structure and Metal command-buffer-cap handling
// (flushPasses) — the difference is M is NOT capped at gemmRowMaxM (real prompts
// run to hundreds/thousands of tokens, past gemmRow's array<i32,16> accumulator
// limit), and only the LAST row's logits are computed (h[M-1] → norm → LM head),
// matching decoder/forwardn.go's prefillLogits CPU contract and the
// decoder.Prefiller.PrefillLast interface this feeds — the other M-1 rows' KV
// entries are still written (the whole prompt's cache is filled either way), just
// their logits are never needed for prefill and so are never computed.
//
// Bit-equivalent to M sequential DecodeRunner.Run calls at positions
// start..start+M-1: same int8 inputs, same int32 accumulation (the tiled GEMM
// equals the GEMV, gated bit-exact by TestTiledDP4A_parity), same per-row ops, same
// KV-then-attend ordering. Gated by TestPrefillLastW8A8_parity.
func (c *Context) PrefillLastW8A8(xs [][]float32, m ModelW, hidden, nH, nKV, hd, inter int, positions []int, start int, eps, scale float32, addOne bool) ([]float32, error) {
	M := len(xs)
	if M == 0 {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 M=0")
	}
	if len(positions) != M {
		return nil, fmt.Errorf("gpu: PrefillLastW8A8 %d rows but %d positions", M, len(positions))
	}
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled, c.ensureTiledBias, c.ensurePrefillBatched, c.ensureQKNorm} {
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
	kvDim := nKV * hd

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

	// rmsB/ropeB/residualB/swigluB/quantPackedM/tiledProjB below all operate on PACKED
	// [M, width] row-major buffers — ONE dispatch across all M rows, not M separate
	// dispatches — the dispatch-count fix (ensurePrefillBatched's doc comment has the
	// measured before/after).

	rmsB := func(inPacked, w *wgpu.Buffer) *wgpu.Buffer {
		out := storF(M * hidden)
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormBatchedPipeline, bind(c.rmsnormBatchedLayout, inPacked, w, out, p), 1, uint32(M))
		return out
	}
	// qkNormB normalizes ALL M rows of a packed [M, heads*hd] buffer in place with
	// weight [hd], before RoPE (Qwen3/GLM). Each workgroup processes one head across
	// the batch.
	qkNormB := func(vecPacked, weight *wgpu.Buffer, heads int) {
		totalHeads := uint32(M * heads)
		p := uni([]uint32{totalHeads, uint32(hd), f32bits(eps), boolU32(addOne)})
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
	ropeB := func(vecPacked, invFreq *wgpu.Buffer, heads int) {
		half := hd / 2
		p := uni([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(positions[0]), f32bits(1), 0, 0, 0})
		disp(c.ropeBatchedPipeline, bind(c.ropeBatchedLayout, vecPacked, invFreq, p), uint32(heads*half+63)/64, uint32(M))
	}
	// residualB / swigluB dispatch their elementwise op as ONE call over the full
	// M*width flattened range instead of M separate width-sized dispatches — both
	// kernels already index by a flat global_invocation_id with no cross-row state
	// (residualShaderWGSL/swigluShaderWGSL, layer.go), so [M, width] IS [M*width]
	// to them; only the dispatch count and per-call uniform-buffer overhead drop.
	residualB := func(xPacked, yPacked *wgpu.Buffer) *wgpu.Buffer { // x += y, in place; returns x
		dispFlat(c.residualPipeline, c.residualLayout, M*hidden, xPacked, yPacked)
		return xPacked
	}
	swigluB := func(gatePacked, upPacked *wgpu.Buffer) *wgpu.Buffer {
		n := M * inter
		dst := storF(n)
		pl, ly := c.swigluPipeline, c.swigluLayout
		if m.GatedGELU {
			pl, ly = c.gegluPipeline, c.gegluLayout
		}
		dispFlat(pl, ly, n, gatePacked, upPacked, dst)
		return dst
	}
	// quantPackedM quantizes all M rows of a packed [M, K] buffer in ONE dispatch, straight into
	// the [M, kp/4] packed-int8 / [M] scales layout tiledProjB's GEMM reads (quantizeShaderWGSL,
	// device.go, takes M rows natively via QDims.m). No gather is needed.
	quantPackedM := func(inPacked *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer, int) {
		kp := padK(K)
		kw := kp / 4
		aqC := storF(M * kw)
		asC := storF(M)
		p := uni([]uint32{uint32(M), uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, inPacked, aqC, asC, p), uint32(M), 1)
		return aqC, asC, kw
	}
	// tiledProjB runs a projection over all M rows: quantPackedM (one dispatch) → ONE unbounded-M
	// tiled GEMM (weight streamed once, DP4A when available) → the packed [M, N] GEMM output is the
	// return value. rowsM lets the LM head reuse it at M=1.
	//
	// bias (nil for every projection except Qwen2's q/k/v) selects the bias-epilogue tiled kernel
	// instead of a separate residual-kernel add: the add has to be fused into the GEMM's expression
	// to stay bit-exact with production's gemvBias (docs/code-notes/gpu.md#runModelToModelW).
	//
	// R-25 (docs/tasks/task-recompute-audit.md): it is quantB then projQB, so the projections that read the same input
	// (q, k, v of xn; gate, up of xn2) quantize it once and share the result. Quantization is deterministic, so each GEMM
	// reads the same bytes it did when it quantized its own copy.
	quantB := func(xnPacked *wgpu.Buffer, rowsM, K int) (aqC, asC *wgpu.Buffer) {
		if rowsM == M {
			aqC, asC, _ = quantPackedM(xnPacked, K)
			return aqC, asC
		}
		// the M=1 LM-head call
		kp := padK(K)
		aqC = storF(kp / 4)
		asC = storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, xnPacked, aqC, asC, p), 1, 1)
		return aqC, asC
	}
	var projQB func(aqC, asC *wgpu.Buffer, rowsM int, rm *ResidentW8A8, bias *wgpu.Buffer) *wgpu.Buffer
	tiledProjB := func(xnPacked *wgpu.Buffer, rowsM int, rm *ResidentW8A8, bias *wgpu.Buffer) *wgpu.Buffer {
		aqC, asC := quantB(xnPacked, rowsM, rm.cols)
		return projQB(aqC, asC, rowsM, rm, bias)
	}
	// sharedQ returns the projections of one input x through rms, quantizing x once (R-25) when every projection has
	// the same K, which q/k/v and gate/up always do; prefillQuantPerProj (tests only) restores a quantization each.
	sharedQ := func(x *wgpu.Buffer, rms []*ResidentW8A8, biases []*wgpu.Buffer) []*wgpu.Buffer {
		out := make([]*wgpu.Buffer, len(rms))
		same := !prefillQuantPerProj
		for _, rm := range rms[1:] {
			same = same && rm.cols == rms[0].cols
		}
		var aqC, asC *wgpu.Buffer
		if same {
			aqC, asC = quantB(x, M, rms[0].cols)
			prefillSharedQuants.Add(1)
		}
		for i, rm := range rms {
			if same {
				out[i] = projQB(aqC, asC, M, rm, biases[i])
			} else {
				out[i] = tiledProjB(x, M, rm, biases[i])
			}
		}
		return out
	}
	projQB = func(aqC, asC *wgpu.Buffer, rowsM int, rm *ResidentW8A8, bias *wgpu.Buffer) *wgpu.Buffer {
		N := rm.rows
		dstC := storF(rowsM * N)
		p := uni([]uint32{uint32(rowsM), uint32(rm.kp), uint32(N), 0})
		gx, gy := c.gemmGrid(N, rowsM)
		if bias != nil {
			disp(c.tiledBiasPipeline, bind(c.tiledBiasLayout, aqC, rm.bq, asC, rm.bScales, dstC, p, bias), gx, gy)
		} else {
			disp(c.tiledPipeline, bind(c.tiledLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), gx, gy)
		}
		return dstC
	}

	xdFlat := make([]float32, 0, M*hidden)
	for r := range xs {
		xdFlat = append(xdFlat, xs[r]...)
	}
	xd, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(xdFlat), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	keepBuf(xd)

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
	for i := range m.Layers {
		recycle() // the previous layer's buffers back to the free list (xd is not one: it is created above, outside storF)
		lw := &m.Layers[i]
		xn := rmsB(xd, lw.Attn.Norm.buf)
		profBoundary(profNormsRope)
		var qBias, kBias, vBias *wgpu.Buffer
		if lw.Attn.QBias != nil { // Qwen2 q/k/v bias — fused into the GEMM epilogue,
			qBias, kBias, vBias = lw.Attn.QBias.buf, lw.Attn.KBias.buf, lw.Attn.VBias.buf // matching decoderunner.go's gemvBias, not a separate post-hoc add (see tiledProjB's doc comment)
		}
		var q *wgpu.Buffer
		var aGate *wgpu.Buffer
		qkv := sharedQ(xn, []*ResidentW8A8{lw.Attn.QProj, lw.Attn.KProj, lw.Attn.VProj}, []*wgpu.Buffer{qBias, kBias, vBias})
		if lw.Attn.QGate {
			qRaw := qkv[0]
			qN := M * nH * hd
			q = storF(qN)
			aGate = storF(qN)
			p := uni([]uint32{uint32(qN), uint32(hd), 0, 0})
			disp(c.deltaQSplitPipeline, bind(c.deltaQSplitLayout, qRaw, q, aGate, p), uint32(qN+63)/64, 1)
		} else {
			q = qkv[0]
		}
		k, v := qkv[1], qkv[2]
		profBoundary(profGemm)
		if lw.Attn.QNorm != nil {
			qkNormB(q, lw.Attn.QNorm.buf, nH)
			qkNormB(k, lw.Attn.KNorm.buf, nKV)
		}
		ropeB(q, lw.Attn.InvFreq.buf, nH)
		ropeB(k, lw.Attn.InvFreq.buf, nKV)
		profBoundary(profNormsRope)
		// Bulk KV-cache write: positions are contiguous (positions[0]..positions[0]+M-1,
		// residency.go), so k/v's packed [M, kvDim] rows land at one contiguous range of
		// the cache — a single copy per k/v instead of M.
		if m.KVF16 {
			totalElems := M * kvDim
			totalWords := totalElems / 2
			p := uni([]uint32{uint32(totalElems), uint32(positions[0] * kvDim), 0, 0})
			disp(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, k, lw.Attn.KCache.buf, p), uint32(totalWords+63)/64, 1)
			disp(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, v, lw.Attn.VCache.buf, p), uint32(totalWords+63)/64, 1)
		} else {
			cpy(k, 0, lw.Attn.KCache.buf, uint64(positions[0]*kvDim*4), uint64(M*kvDim*4))
			cpy(v, 0, lw.Attn.VCache.buf, uint64(positions[0]*kvDim*4), uint64(M*kvDim*4))
		}
		profBoundary(profKVWrite)
		// All rows' K/V are now in the cache; each row attends to its causal prefix (including earlier rows
		// of this block), the ordering DecodeTokenFusedBatched's parity gate already proves. ONE dispatch, grid
		// (nH, M), against attnBatchedKernel's chosen kernel (the same tiled-vs-plain preference as attnKernel,
		// attention.go); q/ctxv are read and written at each row's own offset inside the shader
		// (docs/completed/task-gpu-batched-prefill.md, Increment 1).
		attnPl, attnLy := c.attnBatchedKernel(hd, kvDim, m.KVF16)
		ctxv := storF(M * nH * hd)
		ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(positions[0]), uint32(nH / nKV), f32bits(scale), uint32(M), 0})
		disp(attnPl, bind(attnLy, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, ctxv, ap), uint32(nH), uint32(M))
		if aGate != nil {
			dispFlat(c.deltaAttnGatePipeline, c.deltaAttnGateLayout, M*nH*hd, ctxv, aGate)
		}
		profBoundary(profAttn)
		attnOut := tiledProjB(ctxv, M, lw.Attn.OProj, nil)
		profBoundary(profGemm)
		xd = residualB(xd, attnOut) // in-place: xd += attnOut
		xn2 := rmsB(xd, lw.MLPNorm.buf)
		profBoundary(profNormsRope)
		gu := sharedQ(xn2, []*ResidentW8A8{lw.Gate, lw.Up}, []*wgpu.Buffer{nil, nil})
		gate, up := gu[0], gu[1]
		profBoundary(profGemm)
		mid := swigluB(gate, up)
		profBoundary(profNormsRope)
		down := tiledProjB(mid, M, lw.Down, nil)
		profBoundary(profGemm)
		xd = residualB(xd, down) // in-place: xd += down
		profBoundary(profNormsRope)
	}

	// Final norm + LM head on the LAST row ONLY (matches decoder/forwardn.go's
	// prefillLogits: the other M-1 rows' logits are never needed for prefill) — one
	// small copy to extract row M-1 from the packed buffer, then the same rms/
	// tiledProjB path at rowsM=1.
	xdLast := storF(hidden)
	cpy(xd, uint64((M-1)*hidden*4), xdLast, 0, uint64(hidden*4))
	xnLastP := storF(hidden)
	{
		p := uni([]uint32{uint32(hidden), f32bits(eps), boolU32(addOne), 0})
		disp(c.rmsnormPipeline, bind(c.rmsnormLayout, xdLast, m.FinalNorm.buf, xnLastP, p), 1, 1)
	}
	logitsBuf := tiledProjB(xnLastP, 1, m.LMHead, nil)

	if buildErr != nil {
		return nil, buildErr
	}

	vocab := m.LMHead.rows
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
