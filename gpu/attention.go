//go:build gpu

package gpu

import (
	"fmt"
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

// Single-query attention (decode): one workgroup per query head, an online
// (FlashAttention-style) softmax over keys [start, nKeys) so it is numerically
// stable and needs no scratch for the full score row. GQA maps kvh = qh/group.
// First-cut: one active thread per head (correctness over occupancy).
const attnShaderWGSL = `
struct P { nH: u32, nKV: u32, hd: u32, nKeys: u32, start: u32, group: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       q:    array<f32>;  // [nH*hd]  (RoPE'd)
@group(0) @binding(1) var<storage, read>       keys: array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(2) var<storage, read>       vals: array<f32>;  // [nKeys*nKV*hd]
@group(0) @binding(3) var<storage, read_write> ctx:  array<f32>;  // [nH*hd]
@group(0) @binding(4) var<uniform>             p:    P;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let qh = wid.x;
    if (qh >= p.nH) { return; }
    let hd = p.hd;
    let kvDim = p.nKV * hd;
    let kvh = qh / p.group;
    let qbase = qh * hd;
    let kvbase = kvh * hd;
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: array<f32, 256>;  // hd ≤ 256
    for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = 0.0; }
    for (var s: u32 = p.start; s < p.nKeys; s = s + 1u) {
        let kbase = s * kvDim + kvbase;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { dot = dot + q[qbase + d] * keys[kbase + d]; }
        let x = dot * p.scale;
        let mnew = max(m, x);
        let corr = exp(m - mnew);
        let pe = exp(x - mnew);
        l = l * corr + pe;
        for (var d: u32 = 0u; d < hd; d = d + 1u) { acc[d] = acc[d] * corr + pe * vals[kbase + d]; }
        m = mnew;
    }
    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { ctx[qbase + d] = acc[d] * inv; }
}
`

func (c *Context) ensureAttn() error {
	if c.ropePipeline != nil {
		return nil
	}
	mk := func(label, code string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
		sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: label, WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code}})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("gpu: compile %s: %w", label, err)
		}
		pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: label, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"}})
		if err != nil {
			sh.Release()
			return nil, nil, nil, fmt.Errorf("gpu: pipeline %s: %w", label, err)
		}
		return sh, pl, pl.GetBindGroupLayout(0), nil
	}
	var err error
	if c.ropeShader, c.ropePipeline, c.ropeLayout, err = mk("rope", ropeShaderWGSL); err != nil {
		return err
	}
	if c.attnShader, c.attnPipeline, c.attnLayout, err = mk("attn", attnShaderWGSL); err != nil {
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
