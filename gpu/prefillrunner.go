//go:build gpu

package gpu

import (
	"fmt"

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

// attnBatchedKernel picks the batched attention pipeline+layout for a geometry,
// mirroring attnKernel's (attention.go) own preference for the tiled key-split
// decomposition — restricted to the two cases PrefillLastW8A8 can ever reach
// (runModelToModelW already declines kvF16/kvI8/hd>attnWG), unlike attnKernel's
// full 5-way switch.
func (c *Context) attnBatchedKernel(hd, kvDim int) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	if !attnKeysDisabled && attnKeysEligible(hd, kvDim, false, false) {
		return c.attnKeysBatchedPipeline, c.attnKeysBatchedLayout
	}
	return c.attnBatchedPipeline, c.attnBatchedLayout
}

// ensurePrefillBatched lazily compiles the two batched-row kernels PrefillLastW8A8
// needs beyond its existing ensure-list (M-08's dispatch-count fix, audit-metal
// class finding but on WebGPU: the original implementation issued one dispatch PER
// ROW for RMSNorm/RoPE/quantize-gather/GEMM-scatter, ~39 dispatches × M per layer,
// measured 0.03-0.14x SLOWER than the sequential loop it was meant to replace at
// P=64..1024 — see docs/measurements/ and TestResidentPrefillLast_TTFT). Kept
// separate from ensureLayer/ensureAttn's M=1 pipelines: the decode path must not
// pay for or risk these at all.
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
	return nil
}

// runModelToModelW narrows a resident runModel (the general polymorphic
// representation DecodeRunner uses — W8A8/W4A8, MoE, MLA, Mamba, DeltaNet,
// sliding window, QK-norm, bias, per-layer geometry overrides, …) down to the
// plain dense-W8A8 ModelW shape PrefillLastW8A8 accepts, or reports ok=false when
// this model uses ANY feature outside that shape. The same scope
// DecodeTokenFusedBatched declares ("no MoE/MLA/SSM/bias/QK-norm").
//
// q/k/v bias (Qwen2) IS accepted here (unlike DecodeTokenFusedBatched) — see
// AttnWeights.QBias/KBias/VBias and tiledProj's bias parameter — but callers must
// additionally check ModelW.hasBias() against the backend before trusting the
// result; see residency.go's prefillLast closure for that gate and the full
// history below. This function's own scope guard is architecture-only.
//
// HISTORY (2026-09-12, real-checkpoint debugging, qwen2.5-coder-0.5b): enabling
// bias originally diverged from sequential Forward() at nKeys>=3 (cosine ~0.99).
// Two real, independent causes were found and fixed:
//
//  1. Kernel mismatch: this function unconditionally dispatched c.attnPipeline
//     (the plain f32 attention kernel), while decoderunner.go's sequential decode
//     picks per-geometry via attnKernel (attention.go) — for this checkpoint's
//     geometry (hd=64, kvDim=128, f32 KV) that's the key-split kernel. Two
//     different kernels computing the same attention is not guaranteed
//     bit-identical. Fix: call the same c.attnKernel(hd, kvDim, false)
//     decoderunner.go uses (both now share one implementation).
//
//  2. Epilogue fusion: production's gemvBias computes
//     `f32(acc)*aScale*bScale + bias[n]` in ONE expression inside ONE dispatch;
//     this function did a plain GEMM into a fresh buffer, scattered it out via a
//     byte-exact copy, THEN a separate residualShaderWGSL dispatch added the bias
//     — mathematically the same formula, but forced through an f32 round-trip
//     between the multiply and the add that a single WGSL expression may
//     evaluate with a different (FMA-contracted) rounding. Measured directly
//     (TestLocalize_BiasEpilogue, layer 0's real Q weight+bias): 98 of 896
//     elements differed by up to 2.4e-7 between the two forms — tiny in f32
//     terms, but enough that a subsequent int8 requantize can flip a rounding
//     bucket for an element sitting on the boundary, and 24 layers of that
//     compounds into the observed divergence. Fix: matmulTiledW8A8BiasKernelWGSL
//     (gemm.go) — the SAME tiled GEMM with a fused per-column bias epilogue,
//     textually matching gemvBias's expression.
//
// Together these two fixes take real-checkpoint bias-enabled parity to BIT-EXACT
// (cosine 1.0, maxAbsDiff LITERALLY 0, not float noise) on Vulkan/RTX 2070 SUPER
// at every nKeys measured from 1 to 50.
//
// STILL OPEN, Vulkan-vs-Metal only: on Metal/M1 Pro, the SAME fixes leave a
// smaller but real residual divergence past nKeys~15 (cosine ~0.997-0.999,
// maxAbs ~0.3-0.65 — not float noise, and NOT monotonic with nKeys). A same
// analogy — production fuses the O-proj/down-proj GEMV WITH the residual add
// (decoderunner.go's gemvAdd / gemvW8A8ShaderWGSL's addResidual epilogue), while
// this function does a plain GEMM then a separate residualShaderWGSL add, same
// forced-round-trip shape as the bias case — was the obvious next suspect and is
// almost certainly PART of the real mechanism, but a first attempt at a fused
// residual-epilogue kernel (mirroring matmulTiledW8A8BiasKernelWGSL) introduced a
// NEW regression on Vulkan too (removed rather than shipped broken — see git
// history around 2026-09-12 for the attempt, which gathered the residual stream
// into a contiguous buffer, ran a read-write accumulate kernel, then scattered it
// back — the bug was not found before time ran out on that investigation).
// Whoever picks this up next: rebuild that attempt carefully (gather/accumulate/
// scatter, matching tiledProj's bias-parameter shape but for O-proj/down-proj),
// verify it against TestLocalize_BiasEpilogue-style isolated tests BEFORE wiring
// it into the main loop, and re-measure both backends. Also worth checking: this
// repo's real ship gate for a fast-prefill path is the §3.2 pooled fidelity gate
// against the CPU-f32 reference (docs/task-prefill-gap.md), not bit-exactness
// against sequential GPU decode — Metal's current gap might already clear that
// bar even before a further fix, which would change the urgency here.
//
// Until Metal is resolved or independently cleared, residency.go's prefillLast
// gates bias to backends where it's actually proven (Vulkan only) — a decline
// there falls back to the slower-but-correct sequential loop, never serving a
// silently-wrong result.
//
// It is a zero-copy view: every *wgpu.Buffer is wrapped, not duplicated, and the
// caller must not Close() the resulting ModelW (ModelW.Release would double-free
// buffers rd.rm still owns) — the wrapper DeviceBuffers exist only so ModelW's
// field types match; PrefillLastW8A8 never calls Close on them either.
// hd is the model's head dimension — needed only to tell genuine partial RoPE
// (ropeHalf set to something less than hd/2) apart from ropeHalf simply being SET
// to the full-rotation value instead of left at its 0 "use hd/2" sentinel; both
// PrefillLastW8A8's and DecodeTokenFusedBatched's rope() closures hardcode
// half:=hd/2 (full rotation, no partial-RoPE support), so the latter is fine and
// only the former must decline.
func runModelToModelW(rm *runModel, hd int) (ModelW, bool) {
	if rm.moe != nil || rm.mla != nil || rm.mamba != nil || rm.dnet != nil ||
		rm.kvF16 || rm.kvI8 || rm.slidingWindow != 0 || rm.gatedGELU ||
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
		if l.qNorm != nil || l.kNorm != nil ||
			l.oBias != nil || l.postAttnNorm != nil || l.postMLPNorm != nil ||
			l.hasSink || l.isLocal || (l.ropeScale != 0 && l.ropeScale != 1) ||
			l.ghd != 0 || l.gnKV != 0 || l.ghalf != 0 || l.gKEqV ||
			l.isMoE || l.shGate != nil ||
			l.mlaQA != nil || l.mlaQB != nil || l.mlaQ != nil || l.mlaKVA != nil || l.mlaO != nil ||
			l.isMamba || l.nemoKind != nemoNone || l.isDeltaNet || l.qGate {
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
			},
			MLPNorm: view(l.mlpNorm),
			Gate:    gate, Up: up, Down: down,
		}
	}
	lmHead, lmOK := rm.lmHead.(*ResidentW8A8)
	if !lmOK {
		return ModelW{}, false
	}
	return ModelW{Layers: layers, FinalNorm: view(rm.finalNorm), LMHead: lmHead}, true
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
	for _, f := range []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureTiled, c.ensureTiledBias, c.ensurePrefillBatched} {
		if err := f(); err != nil {
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
	storF := func(n int) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
			return nil
		}
		keepBuf(b)
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
	// maxChunkElems: WebGPU's per-dimension workgroup-COUNT limit is 65535
	// (maxComputeWorkgroupsPerDimension); each workgroup here covers 64 elements, so a
	// single 1-D dispatch tops out at 65535*64 elements. Hit for real at M=1024 on
	// qwen2.5-coder-0.5b (inter=4864: M*inter=4,980,736 > 4,194,240) — surfaced as a
	// WebGPU validation error PrefillLast would otherwise decline into (safe, just
	// slow), not a silent wrong-output risk, but worth actually fixing.
	const maxChunkElems = 65535 * 64
	dispFlat := func(pl *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout, total int, bufs ...*wgpu.Buffer) {
		off := 0
		for off < total {
			n := total - off
			if n > maxChunkElems {
				n = maxChunkElems
			}
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
		dispFlat(c.swigluPipeline, c.swigluLayout, n, gatePacked, upPacked, dst)
		return dst
	}
	// quantPackedM quantizes ALL M rows of a packed [M, K] buffer in ONE dispatch,
	// straight into the [M, kp/4] packed-int8 / [M] scales layout tiledProjB's GEMM
	// wants — quantizeShaderWGSL (device.go) was ALREADY written for M rows in one
	// dispatch (QDims.m, workgroup_id.x = row); the old per-row call site (quant1,
	// dispatched with m=1, M times) never used that capability. No gather needed:
	// the GEMM's aq/aScale inputs ARE this call's direct output.
	quantPackedM := func(inPacked *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer, int) {
		kp := padK(K)
		kw := kp / 4
		aqC := storF(M * kw)
		asC := storF(M)
		p := uni([]uint32{uint32(M), uint32(K), uint32(kp), 0})
		disp(c.quantizePipeline, bind(c.quantizeLayout, inPacked, aqC, asC, p), uint32(M), 1)
		return aqC, asC, kw
	}
	// tiledProjB runs a projection over all M rows: quantPackedM (one dispatch, no
	// gather) → ONE unbounded-M tiled GEMM (weight streamed once, DP4A-accelerated
	// when available) → the packed [M, N] GEMM output IS the return value, no
	// scatter into per-row buffers. rowsM lets the LM head reuse this at M=1 (a
	// one-row "batch" is just the M=1 case of the same dispatch shape).
	//
	// bias (nil for every projection except Qwen2's q/k/v) selects the bias-epilogue
	// tiled kernel instead of a separate post-hoc residual-kernel add — see the
	// original tiledProj's doc comment (git history) for the measured reason this
	// matters for bit-exactness (TestLocalize_BiasEpilogue).
	tiledProjB := func(xnPacked *wgpu.Buffer, rowsM int, rm *ResidentW8A8, bias *wgpu.Buffer) *wgpu.Buffer {
		K, N := rm.cols, rm.rows
		var aqC, asC *wgpu.Buffer
		if rowsM == M {
			aqC, asC, _ = quantPackedM(xnPacked, K)
		} else { // the M=1 LM-head call
			kp := padK(K)
			aqC = storF(kp / 4)
			asC = storF(1)
			p := uni([]uint32{1, uint32(K), uint32(kp), 0})
			disp(c.quantizePipeline, bind(c.quantizeLayout, xnPacked, aqC, asC, p), 1, 1)
		}
		dstC := storF(rowsM * N)
		p := uni([]uint32{uint32(rowsM), uint32(rm.kp), uint32(N), 0})
		gx, gy := (uint32(N)+15)/16, (uint32(rowsM)+15)/16
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

	for i := range m.Layers {
		lw := &m.Layers[i]
		xn := rmsB(xd, lw.Attn.Norm.buf)
		var qBias, kBias, vBias *wgpu.Buffer
		if lw.Attn.QBias != nil { // Qwen2 q/k/v bias — fused into the GEMM epilogue,
			qBias, kBias, vBias = lw.Attn.QBias.buf, lw.Attn.KBias.buf, lw.Attn.VBias.buf // matching decoderunner.go's gemvBias, not a separate post-hoc add (see tiledProjB's doc comment)
		}
		q := tiledProjB(xn, M, lw.Attn.QProj, qBias)
		k := tiledProjB(xn, M, lw.Attn.KProj, kBias)
		v := tiledProjB(xn, M, lw.Attn.VProj, vBias)
		ropeB(q, lw.Attn.InvFreq.buf, nH)
		ropeB(k, lw.Attn.InvFreq.buf, nKV)
		// Bulk KV-cache write: positions are contiguous (positions[0]..positions[0]+M-1,
		// residency.go), so k/v's packed [M, kvDim] rows land at one contiguous range of
		// the cache — a single copy per k/v instead of M.
		cpy(k, 0, lw.Attn.KCache.buf, uint64(positions[0]*kvDim*4), uint64(M*kvDim*4))
		cpy(v, 0, lw.Attn.VCache.buf, uint64(positions[0]*kvDim*4), uint64(M*kvDim*4))
		// All rows' K/V are now in the cache; each row attends to its causal prefix
		// (including earlier rows of this same prefill block) — the same ordering
		// DecodeTokenFusedBatched's parity gate already proves correct.
		//
		// task-gpu-batched-prefill.md Increment 1: ONE dispatch, grid (nH, M), against
		// attnBatchedKernel's chosen kernel (mirrors attnKernel's own tiled-vs-plain
		// preference, attention.go) — replaces what used to be M per-row dispatches
		// into the M=1 kernel (the one dispatch-count cost the earlier fix in this
		// function's history did not eliminate). q/ctxv are read/written directly at
		// each row's own offset inside the shader now, so the qRow-extract/cv-scatter
		// copy pair this loop used to need is gone entirely, not just the M-1 spare
		// allocations of it.
		attnPl, attnLy := c.attnBatchedKernel(hd, kvDim)
		ctxv := storF(M * nH * hd)
		ap := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(positions[0]), uint32(nH / nKV), f32bits(scale), uint32(M), 0})
		disp(attnPl, bind(attnLy, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, ctxv, ap), uint32(nH), uint32(M))
		attnOut := tiledProjB(ctxv, M, lw.Attn.OProj, nil)
		xd = residualB(xd, attnOut) // in-place: xd += attnOut
		xn2 := rmsB(xd, lw.MLPNorm.buf)
		gate := tiledProjB(xn2, M, lw.Gate, nil)
		up := tiledProjB(xn2, M, lw.Up, nil)
		mid := swigluB(gate, up)
		down := tiledProjB(mid, M, lw.Down, nil)
		xd = residualB(xd, down) // in-place: xd += down
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
