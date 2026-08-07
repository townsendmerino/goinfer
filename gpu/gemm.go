//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Tiled W8A8 GEMM (the prefill, M>1 kernel). The naive matmul re-streams a full
// weight row and activation row from global memory per output element — fine when
// bandwidth-bound at M=1, but at prefill it's compute/traffic-bound and wastes
// reuse. This stages 16×16 int8 tiles of the activation and weight into workgroup
// shared memory, so each loaded element is reused across the tile (arithmetic
// intensity ↑). One workgroup computes a 16×16 output block; K is walked in
// 16-wide strips (kp is a multiple of 16). Math identical to MatmulBTW8A8.
const matmulTiledW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

// Tile PACKED WORDS, not int8: each u32 (4 int8 = 4 K) is loaded into shared once
// and reused across the 16×16 output tile, then unpacked in the inner product.
// (Loading individual int8 would re-fetch each word 4× and lose to the naive
// kernel, which already gets free 4-way reuse per word.) Strip = 16 words = 64 K.
const TS: u32 = 16u;
var<workgroup> As: array<u32, 256>;  // [16 rows][16 words]
var<workgroup> Bs: array<u32, 256>;  // [16 cols][16 words]

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dot4(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;                 // words per row
    let row = wid.y * TS + lid.y;          // output row (m)
    let col = wid.x * TS + lid.x;          // output col (n)
    let nStrips = (kw + TS - 1u) / TS;     // 16 words per strip
    var acc: i32 = 0;
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let wcol = s * TS + lid.x;         // word column in this strip
        var aw: u32 = 0u;
        if (row < dims.m && wcol < kw) { aw = aq[row * kw + wcol]; }
        As[lid.y * TS + lid.x] = aw;
        let brow = wid.x * TS + lid.y;     // Bs[lid.y][lid.x] = bq word (colBase+lid.y, wcol)
        var bw: u32 = 0u;
        if (brow < dims.n && wcol < kw) { bw = bq[brow * kw + wcol]; }
        Bs[lid.y * TS + lid.x] = bw;
        workgroupBarrier();
        for (var w: u32 = 0u; w < TS; w = w + 1u) {
            acc = acc + dot4(As[lid.y * TS + w], Bs[lid.x * TS + w]);
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        dst[row * dims.n + col] = f32(acc) * aScales[row] * bScales[col];
    }
}
`

func (c *Context) ensureTiled() error {
	if c.tiledPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "matmulTiledW8A8",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: matmulTiledW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile tiled shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "matmulTiledW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create tiled pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.tiledShader = sh
	c.tiledPipeline = pl
	c.tiledLayout = c.bgl(pl)
	return nil
}

// BatchTiled runs several tiled GEMMs that share one activation aq[M,K] (the
// prefill qkv / gate-up projections) as ONE submit with ONE Poll — the prefill
// analogue of BatchGEMV. The activation is uploaded once; each op dispatches the
// tiled kernel into its own dst, all in one command buffer. Returns each op's
// [M, rms[i].rows] result.
func (c *Context) BatchTiled(aq []int8, aScales []float32, M int, rms []*ResidentW8A8) ([][]float32, error) {
	if err := c.ensureTiled(); err != nil {
		return nil, err
	}
	K := rms[0].cols
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchTiled act: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchTiled ascales: %w", err)
	}
	defer asBuf.Release()

	type perOp struct {
		dst, dims, stag *wgpu.Buffer
		bg              *wgpu.BindGroup
		n               int
	}
	pops := make([]perOp, len(rms))
	release := func() {
		for _, p := range pops {
			for _, b := range []*wgpu.Buffer{p.dst, p.dims, p.stag} {
				if b != nil {
					b.Release()
				}
			}
			if p.bg != nil {
				p.bg.Release()
			}
		}
	}
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	for i, rm := range rms {
		N := rm.rows
		dst, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "btiled-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if err != nil {
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled dst: %w", err)
		}
		dims, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
		if err != nil { // nil dims → nil-panic in CreateBindGroup / the cleanup Release below (audit R-06)
			dst.Release()
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled dims: %w", err)
		}
		bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.tiledLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dst, Size: dst.GetSize()}, {Binding: 5, Buffer: dims, Size: dims.GetSize()},
		}})
		if err != nil {
			dst.Release()
			dims.Release()
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled bind: %w", err)
		}
		pops[i] = perOp{dst: dst, dims: dims, bg: bg, n: N}
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups((uint32(N)+15)/16, (uint32(M)+15)/16, 1)
	}
	if err := pass.End(); err != nil {
		pass.Release()
		release()
		return nil, fmt.Errorf("gpu: BatchTiled pass: %w", err)
	}
	pass.Release()
	for i := range pops {
		stag, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "btiled-stage", Size: uint64(M * pops[i].n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled stage: %w", err)
		}
		pops[i].stag = stag
		enc.CopyBufferToBuffer(pops[i].dst, 0, stag, 0, uint64(M*pops[i].n*4))
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	statuses := make([]wgpu.BufferMapAsyncStatus, len(pops))
	for i := range pops {
		idx := i
		if err := pops[i].stag.MapAsync(wgpu.MapModeRead, 0, uint64(M*pops[i].n*4), func(s wgpu.BufferMapAsyncStatus) { statuses[idx] = s }); err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled map: %w", err)
		}
	}
	c.device.Poll(true, nil) // ONE sync for the whole batch
	outs := make([][]float32, len(pops))
	for i := range pops {
		if statuses[i] != wgpu.BufferMapAsyncStatusSuccess {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled map[%d] failed: %v", i, statuses[i])
		}
		out := make([]float32, M*pops[i].n)
		copy(out, wgpu.FromBytes[float32](pops[i].stag.GetMappedRange(0, uint(M*pops[i].n*4))))
		pops[i].stag.Unmap()
		outs[i] = out
	}
	release()
	return outs, nil
}

// MatmulW8A8Tiled computes dst[M,N] = (aq int8 [M,K]) · rmᵀ with the shared-memory
// tiled kernel — the prefill (M>1) path. aq + per-row scales are uploaded each
// call; the weight rm stays resident.
func (c *Context) MatmulW8A8Tiled(aq []int8, aScales []float32, rm *ResidentW8A8, M int) ([]float32, error) {
	if err := c.ensureTiled(); err != nil {
		return nil, err
	}
	K, N := rm.cols, rm.rows
	if len(aq) < M*K || len(aScales) < M {
		return nil, fmt.Errorf("gpu: MatmulW8A8Tiled input too small")
	}
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled act buffer: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled ascales buffer: %w", err)
	}
	defer asBuf.Release()

	dstSize := uint64(M * N * 4)
	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "tiled-dst", Size: dstSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dst buffer: %w", err)
	}
	defer dstBuf.Release()
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dims buffer: %w", err)
	}
	defer dimsBuf.Release()
	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "tiled-stage", Size: dstSize, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled staging buffer: %w", err)
	}
	defer stage.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.tiledLayout,
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
		return nil, fmt.Errorf("gpu: tiled bind: %w", err)
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(N)+15)/16, (uint32(M)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: tiled pass: %w", err)
	}
	pass.Release()
	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: tiled copy: %w", err)
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: tiled map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: tiled map failed: %v", status)
	}
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(dstSize))))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: tiled unmap: %w", err)
	}
	return out, nil
}
