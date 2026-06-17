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
struct P { nE: u32, k: u32, sigmoid: u32, norm: u32, scale: f32, hasBias: u32, _a: u32, _b: u32 };
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
