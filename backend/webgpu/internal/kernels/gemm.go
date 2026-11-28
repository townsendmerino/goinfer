//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// gemmPipes holds the Context's GEMM, GEMV, quantization, and multi-row W4A8/W8A8 pipelines.
type gemmPipes struct {
	quantShader       *wgpu.ShaderModule
	quantPipeline     *wgpu.ComputePipeline
	quantLayout       *wgpu.BindGroupLayout
	quantizeShader    *wgpu.ShaderModule
	quantizePipeline  *wgpu.ComputePipeline
	quantizeLayout    *wgpu.BindGroupLayout
	gemvShader        *wgpu.ShaderModule
	gemvPipeline      *wgpu.ComputePipeline
	gemvLayout        *wgpu.BindGroupLayout
	gemvBiasShader    *wgpu.ShaderModule
	gemvBiasPipeline  *wgpu.ComputePipeline
	gemvBiasLayout    *wgpu.BindGroupLayout
	gemmTile          int
	tiledShader       *wgpu.ShaderModule
	tiledPipeline     *wgpu.ComputePipeline
	tiledLayout       *wgpu.BindGroupLayout
	tiledByTile       map[int]tiledPipes
	tiledBiasShader   *wgpu.ShaderModule
	tiledBiasPipeline *wgpu.ComputePipeline
	tiledBiasLayout   *wgpu.BindGroupLayout
	tiledBiasByTile   map[int]tiledPipes
	gemmRowShader     *wgpu.ShaderModule
	gemmRowPipeline   *wgpu.ComputePipeline
	gemmRowLayout     *wgpu.BindGroupLayout
	gemvW4Shader      *wgpu.ShaderModule
	gemvW4Pipeline    *wgpu.ComputePipeline
	gemvW4Layout      *wgpu.BindGroupLayout
	w4Multi           [2]tiledPipes
	w4MultiPin        int
	biasRowsPipeline  *wgpu.ComputePipeline
	biasRowsLayout    *wgpu.BindGroupLayout
}

// ensureTiledBias compiles the bias-epilogue tiled GEMM (see
// matmulTiledW8A8BiasKernelWGSL) — a separate pipeline from ensureTiled's, chosen
// the same way (c.hasDP4A), since a bias-taking kernel needs its own bind group
// layout (7 bindings vs 6).
func (c *Context) ensureTiledBias() error {
	if c.tiledBiasPipeline != nil {
		return nil
	}
	if c.gemmTile == 0 {
		c.gemmTile = gemmTileDefault()
	}
	p, ok := c.tiledBiasByTile[c.gemmTile]
	if !ok {
		var err error
		if p, err = c.compileTiled(c.gemmTile, true); err != nil {
			return err
		}
		if c.tiledBiasByTile == nil {
			c.tiledBiasByTile = map[int]tiledPipes{}
		}
		c.tiledBiasByTile[c.gemmTile] = p
	}
	c.tiledBiasShader = p.sh
	c.tiledBiasPipeline = p.Pl
	c.tiledBiasLayout = p.Layout
	return nil
}

func (c *Context) ensureTiled() error {
	if c.tiledPipeline != nil {
		return nil
	}
	if c.gemmTile == 0 {
		c.gemmTile = gemmTileDefault()
	}
	p, ok := c.tiledByTile[c.gemmTile]
	if !ok {
		var err error
		if p, err = c.compileTiled(c.gemmTile, false); err != nil {
			return err
		}
		if c.tiledByTile == nil {
			c.tiledByTile = map[int]tiledPipes{}
		}
		c.tiledByTile[c.gemmTile] = p
	}
	c.tiledShader = p.sh
	c.tiledPipeline = p.Pl
	c.tiledLayout = p.Layout
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
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "btiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchTiled act: %w", err)
	}
	defer releaseBuf(aBuf)
	asBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "btiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchTiled ascales: %w", err)
	}
	defer releaseBuf(asBuf)

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
					releaseBuf(b)
				}
			}
			if p.bg != nil {
				p.bg.Release()
			}
		}
	}
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		release()
		return nil, fmt.Errorf("gpu: BatchTiled: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	for i, rm := range rms {
		N := rm.rows
		dst, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "btiled-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if err != nil {
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled dst: %w", err)
		}
		dims, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "btiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
		if err != nil { // nil dims → nil-panic in CreateBindGroup / the cleanup Release below
			releaseBuf(dst)
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled dims: %w", err)
		}
		bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.tiledLayout, Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dst, Size: dst.GetSize()}, {Binding: 5, Buffer: dims, Size: dims.GetSize()},
		}})
		if err != nil {
			releaseBuf(dst)
			releaseBuf(dims)
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled bind: %w", err)
		}
		pops[i] = perOp{dst: dst, dims: dims, bg: bg, n: N}
		pass.SetBindGroup(0, bg, nil)
		gx, gy := c.gemmGrid(N, M)
		pass.DispatchWorkgroups(gx, gy, 1)
	}
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		release()
		return nil, fmt.Errorf("gpu: BatchTiled pass: %w", err)
	}
	pass.Release()
	for i := range pops {
		stag, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "btiled-stage", Size: uint64(M * pops[i].n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled stage: %w", err)
		}
		pops[i].stag = stag
		if err := enc.TryCopyBufferToBuffer(pops[i].dst, 0, stag, 0, uint64(M*pops[i].n*4)); err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled copy[%d]: %w", i, err)
		}
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		release()
		return nil, fmt.Errorf("gpu: BatchTiled: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	statuses := make([]wgpu.MapAsyncStatus, len(pops))
	for i := range pops {
		idx := i
		if err := pops[i].stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(M*pops[i].n*4), func(s wgpu.MapAsyncStatus) { statuses[idx] = s }); err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled map: %w", err)
		}
	}
	c.device.Poll(true, nil) // ONE sync for the whole batch
	outs := make([][]float32, len(pops))
	for i := range pops {
		if statuses[i] != wgpu.MapAsyncStatusSuccess {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled map[%d] failed: %v", i, statuses[i])
		}
		out := make([]float32, M*pops[i].n)
		copy(out, wgpu.FromBytes[float32](pops[i].stag.GetMappedRange(0, uint(M*pops[i].n*4))))
		pops[i].stag.TryUnmap()
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
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "tiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled act buffer: %w", err)
	}
	defer releaseBuf(aBuf)
	asBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "tiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled ascales buffer: %w", err)
	}
	defer releaseBuf(asBuf)

	dstSize := uint64(M * N * 4)
	dstBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{
		Label: "tiled-dst", Size: dstSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dst buffer: %w", err)
	}
	defer releaseBuf(dstBuf)
	dimsBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "tiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dims buffer: %w", err)
	}
	defer releaseBuf(dimsBuf)
	stage, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{
		Label: "tiled-stage", Size: dstSize, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled staging buffer: %w", err)
	}
	defer releaseBuf(stage)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{
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
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8Tiled: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := c.gemmGrid(N, M)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: tiled pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: tiled copy: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8Tiled: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := stage.TryMapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: tiled map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: tiled map failed: %v", status)
	}
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(dstSize))))
	if err := stage.TryUnmap(); err != nil {
		return nil, fmt.Errorf("gpu: tiled unmap: %w", err)
	}
	return out, nil
}
