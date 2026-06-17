//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// MLA residency (Lever C4) — DeepSeek / Kimi Multi-head Latent Attention on the GPU
// resident decode path. The compressed-KV latent ([kv_lora_rank | qk_rope_head_dim]
// per position) is the only cached payload; per-head K/V are never materialized.
// This file is the absorb-path core primitive (mirrors decoder.mlaAttentionAbsorb):
// the query is pre-absorbed through W_UK once per head (→ rank-space), each cached
// latent is scored by a rank+rope-dim dot, and V is collapsed to a rank-space weighted
// sum (lifted by W_UV later, on the host side of the runner). The surrounding GEMVs
// (q-LoRA, kv-down, W_UK absorb, W_UV lift, o-proj) reuse the W8A8 path; this kernel
// is the genuinely-new attention shape — a single-query attend with a KV "head" shared
// across all query heads and a score width (latDim) wider than the value width (rank).
//
// Layout: qAbs[h] = qNopeAbs_h[rank] ‖ qRope_h[qkRope] (the absorbed query, width
// latDim = rank+qkRope); lat[j] = cn_j[rank] ‖ krj_j[qkRope] (the stored latent —
// normalized + rope-keyed at store time). Score_j = scale·(qAbs[h]·lat[j]) over the
// full latDim; value = cn_j = lat[j][:rank]; output wsum[h] = Σ_j softmax_j·cn_j.
//
// One workgroup per query head, online (FlashAttention-style) softmax over keys so no
// per-key score row is materialized. The score dot and the value accumulate are both
// STRIDED over the 128-lane workgroup because rank (≤1024) and latDim exceed the lane
// count — each lane owns dims {t, t+128, …}; m/l/corr/pe are scalar (identical across
// lanes, all see the same reduced score) so no extra reduction beyond the score dot.
const mlaAttnWGSL = `
const WG: u32 = 128u;
struct P { nH: u32, latDim: u32, rank: u32, nKeys: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qAbs: array<f32>;  // [nH*latDim] qNopeAbs ‖ qRope per head
@group(0) @binding(1) var<storage, read>       lat:  array<f32>;  // [nKeys*latDim] cn ‖ krj per key
@group(0) @binding(2) var<storage, read_write> wsum: array<f32>;  // [nH*rank] rank-space V collapse
@group(0) @binding(3) var<uniform>             p:    P;
var<workgroup> red: array<f32, WG>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.nH) { return; }
    let t = lid.x;
    let latDim = p.latDim;
    let rank = p.rank;
    let qbase = h * latDim;
    var acc: array<f32, 8>;  // per-lane value accumulators (rank ≤ 8*WG = 1024)
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { acc[i] = 0.0; }
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    for (var j: u32 = 0u; j < p.nKeys; j = j + 1u) {
        let kbase = j * latDim;
        // score = qAbs[h] · lat[j] over the full latDim (strided across lanes).
        var partial: f32 = 0.0;
        for (var d: u32 = t; d < latDim; d = d + WG) {
            partial = partial + qAbs[qbase + d] * lat[kbase + d];
        }
        red[t] = partial;
        workgroupBarrier();
        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let x = red[0] * p.scale;
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        // value = cn_j = lat[j][:rank]; accumulate the rank dims this lane owns.
        var i: u32 = 0u;
        for (var d: u32 = t; d < rank; d = d + WG) {
            acc[i] = acc[i] * corr + pe * lat[kbase + d];
            i = i + 1u;
        }
        l = l * corr + pe;
        m = mnew;
        workgroupBarrier();  // all lanes consumed red[0] before next key overwrites red[t]
    }
    var i: u32 = 0u;
    for (var d: u32 = t; d < rank; d = d + WG) {
        wsum[h * rank + d] = acc[i] / l;
        i = i + 1u;
    }
}
`

func (c *Context) ensureMLAAttn() error {
	if c.mlaAttnPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaAttn", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mlaAttnWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaAttn: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaAttn", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaAttn: %w", err)
	}
	c.mlaAttnShader, c.mlaAttnPipeline, c.mlaAttnLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}

// MLAAttn runs the absorb-path rank-space attention on host inputs and returns the
// rank-space V collapse wsum [nH*rank]. qAbs is [nH*latDim] (qNopeAbs ‖ qRope per
// head), lat is [nKeys*latDim] (cn ‖ krj per key), latDim = rank + qkRope. Standalone
// op for parity (the resident runner records the device kernel directly).
func (c *Context) MLAAttn(qAbs, lat []float32, nH, latDim, rank, nKeys int, scale float32) ([]float32, error) {
	if err := c.ensureMLAAttn(); err != nil {
		return nil, err
	}
	if rank > 8*128 {
		return nil, fmt.Errorf("gpu: MLAAttn rank %d exceeds 1024 (per-lane acc cap)", rank)
	}
	qBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-qabs", Contents: wgpu.ToBytes(qAbs), Usage: wgpu.BufferUsageStorage})
	defer qBuf.Release()
	lBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-lat", Contents: wgpu.ToBytes(lat), Usage: wgpu.BufferUsageStorage})
	defer lBuf.Release()
	wBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mla-wsum", Size: uint64(nH * rank * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer wBuf.Release()
	pBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "mla-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(latDim), uint32(rank), uint32(nKeys), f32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer pBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaAttnLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()},
		{Binding: 1, Buffer: lBuf, Size: lBuf.GetSize()},
		{Binding: 2, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.mlaAttnPipeline)
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
	return c.Readback(&DeviceBuffer{buf: wBuf, n: nH * rank})
}
