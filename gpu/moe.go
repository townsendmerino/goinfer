//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// MoE residency (Lever C3) — sparse expert routing + dispatch on the GPU so the MoE
// families (Mixtral / Qwen2-MoE / GLM / DeepSeek) run on the resident DecodeRunner
// instead of the staged path. This file is C3a: the router top-k SELECTION kernel.
// C3b adds the indexed sparse-expert GEMV; C3c wires both into the runner.
//
// The selection mirrors the CPU routeExperts (decoder/mlp.go): score the router logits
// (softmax for Mixtral/Qwen2-MoE, or per-expert sigmoid for DeepSeek/GLM), optionally add
// a per-expert selection bias, take the top-k by selection score, set each chosen
// expert's WEIGHT to its un-biased score, optionally renormalize the k weights to sum 1
// (Mixtral norm_topk_prob) and scale them (DeepSeek routed_scaling_factor). The
// group-limited variant (DeepSeek nGroup>1) is deferred to C3d; this kernel is the
// nGroup==1 path. nE is tiny (8–256) so one single-lane workgroup is plenty — selection
// is not the cost, the expert GEMVs are.
const moeRouteWGSL = `
const MAXE: u32 = 256u;
struct P { nE: u32, k: u32, sigmoid: u32, norm: u32, scale: f32, hasBias: u32, nGroup: u32, topkGroup: u32 };
@group(0) @binding(0) var<storage, read>       logits: array<f32>;  // [nE] router logits
@group(0) @binding(1) var<storage, read>       bias:   array<f32>;  // [nE] selection bias (read only if hasBias)
@group(0) @binding(2) var<storage, read_write> outIdx: array<u32>;  // [k] chosen expert indices
@group(0) @binding(3) var<storage, read_write> outWgt: array<f32>;  // [k] chosen expert weights
@group(0) @binding(4) var<uniform>             p:      P;
@compute @workgroup_size(1)
fn main() {
    let nE = min(p.nE, MAXE);
    var score: array<f32, 256>;  // un-biased score (softmax prob or sigmoid)
    var sel:   array<f32, 256>;  // selection score (score + bias)
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

// Indexed sparse-expert GEMV (Lever C3b). Identical int8 GEMV math to gemvW8A8, but the
// weight ROW base is computed from a DYNAMIC expert index read out of the routing buffer
// (idx[slot], produced on-GPU by moeRoute) into a STACKED [nE,N,kp] weight buffer — so the
// resident plan records a FIXED k dispatches (one per selected expert slot) while the
// actual expert is chosen at run time, no host round-trip. mode 1 folds the router weight
// into the epilogue (dst[n] += wgt[slot]·r) for the down-projection combine; mode 0
// overwrites (gate/up into their own scratch).
const moeExpertGEMVWGSL = `
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

// Gated shared-expert combine (Lever C3d). Qwen2-MoE adds an always-on shared expert
// scaled by a per-token sigmoid gate g = sigmoid(SharedGate·h): xd[n] += g·sharedDown[n].
// gl is the 1-element gate logit (the SharedGate GEMV result); each thread reads it and
// sigmoids inline. GLM/DeepSeek add the shared expert UNGATED (a plain residual add via
// gemvAdd), so they never reach this kernel.
const sharedGateWGSL = `
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

func (c *Context) ensureSharedGate() error {
	if c.sharedGatePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "sharedGate", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: sharedGateWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile sharedGate: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "sharedGate", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline sharedGate: %w", err)
	}
	c.sharedGateShader, c.sharedGatePipeline, c.sharedGateLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

// ResidentStackedW8A8 holds nE experts' W8A8 weight for ONE projection (gate, up, or
// down) packed back-to-back in the gemvW8A8 layout, indexable by expert: row (e*N+n).
type ResidentStackedW8A8 struct {
	bq, bScales *wgpu.Buffer
	nE, rows    int
	cols, kp    int
	// w4 ⇒ this stack is int4 (W4A8): bq holds packed nibbles, bScales holds f16
	// group scales (kp = padK32), and moeExpert dispatches the int4 kernel.
	w4 bool
}

// Release frees the stacked buffers.
func (s *ResidentStackedW8A8) Release() {
	if s.bq != nil {
		s.bq.Release()
	}
	if s.bScales != nil {
		s.bScales.Release()
	}
}

// UploadStackedExperts packs nE experts' int8 weights [each N,K] + per-row scales [each N]
// into one resident buffer (expert e's row n at packed-row e*N+n), the layout the indexed
// expert GEMV reads. q8[e] / scales[e] are expert e's quantized projection.
func (c *Context) UploadStackedExperts(q8 [][]int8, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(q8) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExperts bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	kp := padK(K)
	words := kp / 4
	packed := make([]uint32, nE*N*words)
	allScales := make([]float32, nE*N)
	for e := 0; e < nE; e++ {
		if len(q8[e]) < N*K || len(scales[e]) < N {
			return nil, fmt.Errorf("gpu: UploadStackedExperts expert %d too small", e)
		}
		copy(packed[e*N*words:(e+1)*N*words], packInt8(q8[e], N, K))
		copy(allScales[e*N:(e+1)*N], scales[e][:N])
	}
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-experts", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked experts buffer: %w", err)
	}
	sc, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-expert-scales", Contents: wgpu.ToBytes(allScales), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: stacked expert scales buffer: %w", err)
	}
	return &ResidentStackedW8A8{bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp}, nil
}

func (c *Context) ensureMoEExpert() error {
	if c.moeExpertPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGEMV", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: moeExpertGEMVWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGEMV: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGEMV", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGEMV: %w", err)
	}
	c.moeExpertShader, c.moeExpertPipeline, c.moeExpertLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

// IndexedGEMVForTest runs ONE indexed expert GEMV standalone (overwrite mode): dst[N] =
// (aq·expert[idx[slot]]ᵀ). The C3b unit-test seam — isolates the stacked-index addressing
// from the full MoE block (C3c). aq is the quantized activation [K], aScale its scale.
func (c *Context) IndexedGEMVForTest(s *ResidentStackedW8A8, aq []int8, aScale float32, idx []int, slot int) ([]float32, error) {
	if err := c.ensureMoEExpert(); err != nil {
		return nil, err
	}
	N := s.rows
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer aBuf.Release()
	asBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer asBuf.Release()
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer idxBuf.Release()
	wgtBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-wgt", Contents: wgpu.ToBytes(make([]float32, len(idx))), Usage: wgpu.BufferUsageStorage})
	defer wgtBuf.Release()
	dstBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "ig-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer dstBuf.Release()
	dims, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer dims.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeExpertLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeExpertPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := gemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.End()
	pass.Release()
	stg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stg.Release()
	enc.CopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.BufferMapAsyncStatus
	if err := stg.MapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: indexed gemv map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.Unmap()
	return out, nil
}

func (c *Context) ensureMoERoute() error {
	if c.moeRoutePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeRoute", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: moeRouteWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeRoute: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeRoute", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeRoute: %w", err)
	}
	c.moeRouteShader, c.moeRoutePipeline, c.moeRouteLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

// RouteExpertsForTest runs the router top-k selection kernel standalone (upload logits,
// dispatch, read back) — the C3a unit-test seam mirroring decoder.routeExperts for
// nGroup==1. bias may be nil. Returns the k chosen expert indices and their weights.
func (c *Context) RouteExpertsForTest(logits, bias []float32, k int, sigmoid, norm bool, scale float32) ([]int, []float32, error) {
	if err := c.ensureMoERoute(); err != nil {
		return nil, nil, err
	}
	nE := len(logits)
	lb, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-logits", Contents: wgpu.ToBytes(logits), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer lb.Release()
	hasBias := uint32(0)
	bb := bias
	if bb == nil {
		bb = make([]float32, nE) // a bound (unused) buffer; hasBias=0 keeps it out of the math
	} else {
		hasBias = 1
	}
	bbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-bias", Contents: wgpu.ToBytes(bb), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer bbuf.Release()
	idxBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "moe-idx", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer idxBuf.Release()
	wgtBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "moe-wgt", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer wgtBuf.Release()
	sig, nrm := uint32(0), uint32(0)
	if sigmoid {
		sig = 1
	}
	if norm {
		nrm = 1
	}
	pbuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-p", Contents: wgpu.ToBytes([]uint32{uint32(nE), uint32(k), sig, nrm, f32bits(scale), hasBias, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer pbuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeRouteLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: lb, Size: lb.GetSize()}, {Binding: 1, Buffer: bbuf, Size: bbuf.GetSize()},
		{Binding: 2, Buffer: idxBuf, Size: idxBuf.GetSize()}, {Binding: 3, Buffer: wgtBuf, Size: wgtBuf.GetSize()},
		{Binding: 4, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return nil, nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeRoutePipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	pass.End()
	pass.Release()
	idxStg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer idxStg.Release()
	wgtStg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer wgtStg.Release()
	enc.CopyBufferToBuffer(idxBuf, 0, idxStg, 0, uint64(k*4))
	enc.CopyBufferToBuffer(wgtBuf, 0, wgtStg, 0, uint64(k*4))
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var si, sw wgpu.BufferMapAsyncStatus
	if err := idxStg.MapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.BufferMapAsyncStatus) { si = s }); err != nil {
		return nil, nil, err
	}
	if err := wgtStg.MapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.BufferMapAsyncStatus) { sw = s }); err != nil {
		return nil, nil, err
	}
	c.device.Poll(true, nil)
	if si != wgpu.BufferMapAsyncStatusSuccess || sw != wgpu.BufferMapAsyncStatusSuccess {
		return nil, nil, fmt.Errorf("gpu: moe route map failed: idx=%v wgt=%v", si, sw)
	}
	idx := make([]int, k)
	for i, v := range wgpu.FromBytes[uint32](idxStg.GetMappedRange(0, uint(k*4))) {
		idx[i] = int(v)
	}
	idxStg.Unmap()
	wts := make([]float32, k)
	copy(wts, wgpu.FromBytes[float32](wgtStg.GetMappedRange(0, uint(k*4))))
	wgtStg.Unmap()
	return idx, wts, nil
}
