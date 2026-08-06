//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Thin-M W8A8 GEMM for the Stage-B verify (docs/spec/07). The 16×16 tiled GEMM
// (gemm.go) is the PREFILL kernel; at the small M of a speculative block (M≈K+1≈8)
// it wastes half its 16-row tile and loses to per-row GEMV (measured 0.88×). This
// kernel keeps the GEMV's structure — one WORKGROUP per output column n, 64 lanes
// coalesce-stride the contiguous weight row — but loads each weight vec4 ONCE and
// accumulates it against ALL M activation rows (M register accumulators). So the
// weight matrix streams once for the whole block (the Stage-B win) with NO wasted
// tile dimension. Arithmetic is identical to M separate GEMVs ⇒ bit-parity.
const gemmRowMaxM = 16

const gemmRowW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [M, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [M*N] row-major (m*N + n)
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
    let n = wid.x + wid.y * 32768u;   // 2D grid: N can exceed the 65535/dim cap
    if (n >= dims.n) { return; }      // n is workgroup-uniform ⇒ barrier-safe
    let t = lid.x;
    let kv = dims.kp / 16u;
    let bBase = n * kv;
    let M = dims.m;                   // uniform ⇒ the M-loops are uniform control flow

    var acc: array<i32, 16>;          // gemmRowMaxM accumulators
    for (var m: u32 = 0u; m < M; m = m + 1u) { acc[m] = 0; }

    for (var v: u32 = t; v < kv; v = v + 64u) {   // warp reads consecutive weight vec4s ⇒ coalesced
        let b4 = bq[bBase + v];                    // weight loaded ONCE, reused across all M rows
        for (var m: u32 = 0u; m < M; m = m + 1u) {
            let a4 = aq[m * kv + v];
            acc[m] = acc[m] + dotw(a4.x, b4.x) + dotw(a4.y, b4.y) + dotw(a4.z, b4.z) + dotw(a4.w, b4.w);
        }
    }

    for (var m: u32 = 0u; m < M; m = m + 1u) {   // reduce each row, reusing the one shared buffer
        partial[t] = acc[m];
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) {
            dst[m * dims.n + n] = f32(partial[0]) * aScales[m] * bScales[n];
        }
        workgroupBarrier();
    }
}
`

func (c *Context) ensureGemmRow() error {
	if c.gemmRowPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "gemmRowW8A8",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: gemmRowW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile gemmRow shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "gemmRowW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create gemmRow pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.gemmRowShader = sh
	c.gemmRowPipeline = pl
	c.gemmRowLayout = pl.GetBindGroupLayout(0)
	return nil
}

// MatmulW8A8GemmRow computes M activation rows against rm with the thin-M kernel,
// returning [M*N] row-major. M ≤ gemmRowMaxM. One-shot (per-call buffers) — the
// parity-gated primitive; DecodeTokenFusedBatched records it inline for the verify.
func (c *Context) MatmulW8A8GemmRow(aq []int8, aScales []float32, rm *ResidentW8A8, M int) ([]float32, error) {
	if M <= 0 || M > gemmRowMaxM {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow M=%d out of range (1..%d)", M, gemmRowMaxM)
	}
	K, N := rm.cols, rm.rows
	if len(aq) < M*K || len(aScales) < M {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow input too small")
	}
	if err := c.ensureGemmRow(); err != nil {
		return nil, err
	}
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "gemmrow-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer aBuf.Release()
	asBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "gemmrow-ascale", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer asBuf.Release()
	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "gemmrow-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer dstBuf.Release()
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "gemmrow-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer dimsBuf.Release()
	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "gemmrow-stage", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer stage.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.gemmRowLayout,
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
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.gemmRowPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := gemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.End()
	pass.Release()
	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, uint64(M*N*4)); err != nil {
		return nil, err
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, uint64(M*N*4), func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow map failed: %v", status)
	}
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(M*N*4))))
	if err := stage.Unmap(); err != nil {
		return nil, err
	}
	return out, nil
}
