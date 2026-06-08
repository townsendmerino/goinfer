//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Coalesced W8A8 GEMV (the M=1 decode kernel). The naive matmul assigns thread
// (m,n) to one output and has it stream a whole weight row — so adjacent threads
// (n, n+1) read rows kw words apart: UNCOALESCED, ~6% of bandwidth, slower than
// the CPU at decode. This kernel instead assigns one WORKGROUP per output column
// n: its 64 lanes cooperatively stride the contiguous weight row bq[n], so a warp
// reads consecutive words (coalesced), then a tree-reduce sums the partials.
// Decode is bandwidth-bound, so coalescing the weight stream is the actual unlock.
const gemvW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

// vec4<u32> view of the packed int8 (16 int8 / 16-byte transaction): wider loads
// raise memory throughput vs scalar u32. kp is a multiple of 16, so kp/16 vec4s.
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [1, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotw(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}

var<workgroup> partial: array<i32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x;  // one workgroup per output column
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kv = dims.kp / 16u;     // vec4s per row
    let bBase = n * kv;
    var acc: i32 = 0;
    for (var v: u32 = t; v < kv; v = v + 64u) {   // warp reads consecutive vec4s ⇒ coalesced
        let a4 = aq[v];
        let b4 = bq[bBase + v];
        acc = acc + dotw(a4.x, b4.x) + dotw(a4.y, b4.y) + dotw(a4.z, b4.z) + dotw(a4.w, b4.w);
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        dst[n] = f32(partial[0]) * aScales[0] * bScales[n];
    }
}
`

func (c *Context) ensureGEMV() error {
	if c.gemvPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "gemvW8A8",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: gemvW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile GEMV shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "gemvW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create GEMV pipeline: %w", err)
	}
	c.gemvShader = sh
	c.gemvPipeline = pl
	c.gemvLayout = pl.GetBindGroupLayout(0)
	return nil
}

// GEMVRunner is the steady-state decode path: it allocates the activation /
// scale / output / staging buffers and the bind group ONCE for a given resident
// weight, then each Run only WriteBuffers the new activation and dispatches —
// avoiding the per-call buffer churn that dominated the one-shot MatmulW8A8GEMV
// (GPU sync is cheap; wgpu buffer creation is not). This is what the decoder
// caches per weight matrix.
type GEMVRunner struct {
	c                                  *Context
	rm                                 *ResidentW8A8
	aBuf, asBuf, dstBuf, dimsBuf, stag *wgpu.Buffer
	bg                                 *wgpu.BindGroup
	n                                  int
}

// NewGEMVRunner builds a reusable decode runner for a resident weight.
func (c *Context) NewGEMVRunner(rm *ResidentW8A8) (*GEMVRunner, error) {
	if err := c.ensureGEMV(); err != nil {
		return nil, err
	}
	N := rm.rows
	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	aBuf, err := mk("gemvr-act", uint64(rm.kp/4*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return nil, err
	}
	asBuf, _ := mk("gemvr-ascale", 4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	dstBuf, _ := mk("gemvr-dst", uint64(N*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	stag, _ := mk("gemvr-stage", uint64(N*4), wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "gemvr-dims", Contents: wgpu.ToBytes([]uint32{1, uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		aBuf.Release()
		asBuf.Release()
		dstBuf.Release()
		stag.Release()
		return nil, err
	}
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.gemvLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
			{Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		aBuf.Release()
		asBuf.Release()
		dstBuf.Release()
		stag.Release()
		dimsBuf.Release()
		return nil, err
	}
	return &GEMVRunner{c: c, rm: rm, aBuf: aBuf, asBuf: asBuf, dstBuf: dstBuf, dimsBuf: dimsBuf, stag: stag, bg: bg, n: N}, nil
}

// Run computes dst[N] = (aq quantized int8) · rmᵀ, reusing the runner's buffers.
func (r *GEMVRunner) Run(aq []int8, aScale float32) ([]float32, error) {
	c := r.c
	if err := c.queue.WriteBuffer(r.aBuf, 0, wgpu.ToBytes(packInt8(aq, 1, r.rm.cols))); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner write act: %w", err)
	}
	if err := c.queue.WriteBuffer(r.asBuf, 0, wgpu.ToBytes([]float32{aScale})); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner write scale: %w", err)
	}
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.gemvPipeline)
	pass.SetBindGroup(0, r.bg, nil)
	pass.DispatchWorkgroups(uint32(r.n), 1, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: GEMVRunner pass: %w", err)
	}
	pass.Release()
	if err := enc.CopyBufferToBuffer(r.dstBuf, 0, r.stag, 0, uint64(r.n*4)); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner copy: %w", err)
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := r.stag.MapAsync(wgpu.MapModeRead, 0, uint64(r.n*4), func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: GEMVRunner map failed: %v", status)
	}
	out := make([]float32, r.n)
	copy(out, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.n*4))))
	if err := r.stag.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner unmap: %w", err)
	}
	return out, nil
}

// Release frees the runner's buffers (not the resident weight).
func (r *GEMVRunner) Release() {
	r.aBuf.Release()
	r.asBuf.Release()
	r.dstBuf.Release()
	r.dimsBuf.Release()
	r.stag.Release()
	r.bg.Release()
}

// MatmulW8A8GEMV computes dst[N] = (aq quantized int8, 1×K) · rmᵀ with the
// coalesced one-workgroup-per-column kernel — the decode (M=1) path. aq is the
// int8 activation row + its single scale.
func (c *Context) MatmulW8A8GEMV(aq []int8, aScale float32, rm *ResidentW8A8) ([]float32, error) {
	if err := c.ensureGEMV(); err != nil {
		return nil, err
	}
	K, N := rm.cols, rm.rows
	if len(aq) < K {
		return nil, fmt.Errorf("gpu: MatmulW8A8GEMV act too small: %d < %d", len(aq), K)
	}
	packed := packInt8(aq, 1, K)
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "gemv-act", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV act buffer: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "gemv-ascale", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV ascale buffer: %w", err)
	}
	defer asBuf.Release()

	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gemv-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV dst buffer: %w", err)
	}
	defer dstBuf.Release()
	dims := []uint32{1, uint32(rm.kp), uint32(N), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "gemv-dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV dims buffer: %w", err)
	}
	defer dimsBuf.Release()
	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "gemv-stage", Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV staging: %w", err)
	}
	defer stage.Release()

	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.gemvLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
			{Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV bind: %w", err)
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.gemvPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(N), 1, 1) // one workgroup per output column
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: GEMV pass: %w", err)
	}
	pass.Release()
	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, uint64(N*4)); err != nil {
		return nil, fmt.Errorf("gpu: GEMV copy: %w", err)
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: GEMV map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: GEMV map failed: %v", status)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(N*4))))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: GEMV unmap: %w", err)
	}
	return out, nil
}
