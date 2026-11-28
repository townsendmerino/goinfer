//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// mlaPipes holds the Context's MLA absorb-path attention, latent store, per-head matvec, and query RoPE pipelines.
type mlaPipes struct {
	mlaAttnShader     *wgpu.ShaderModule
	mlaAttnPipeline   *wgpu.ComputePipeline
	mlaAttnLayout     *wgpu.BindGroupLayout
	mlaStoreShader    *wgpu.ShaderModule
	mlaStorePipeline  *wgpu.ComputePipeline
	mlaStoreLayout    *wgpu.BindGroupLayout
	mlaHeadMVShader   *wgpu.ShaderModule
	mlaHeadMVPipeline *wgpu.ComputePipeline
	mlaHeadMVLayout   *wgpu.BindGroupLayout
	mlaQRopeShader    *wgpu.ShaderModule
	mlaQRopePipeline  *wgpu.ComputePipeline
	mlaQRopeLayout    *wgpu.BindGroupLayout
}

func (c *Context) ensureMLAStore() error {
	if c.mlaStorePipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaLatentStore", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MlaLatentStoreWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaLatentStore: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaLatentStore", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaLatentStore: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mlaStoreShader, c.mlaStorePipeline, c.mlaStoreLayout = sh, pl, c.bgl(pl)
	return nil
}

// MLALatentStore runs the latent append on host inputs: kvDown [rank+qkRope] → the
// normalized+roped latent entry [rank+qkRope]. Standalone op for parity (the runner
// records the device kernel with base = pos*latDim into the resident latent cache).
func (c *Context) MLALatentStore(kvDown, normW, invFreq []float32, rank, qkRope, pos int, eps, ropeScale float32, interleave bool) ([]float32, error) {
	if err := c.ensureMLAStore(); err != nil {
		return nil, err
	}
	latDim := rank + qkRope
	dBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-kvdown", Contents: wgpu.ToBytes(kvDown), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLALatentStore: mla-kvdown: %w", err)
	}
	defer releaseBuf(dBuf)
	nBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-normw", Contents: wgpu.ToBytes(normW), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLALatentStore: mla-normw: %w", err)
	}
	defer releaseBuf(nBuf)
	fBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-invfreq", Contents: wgpu.ToBytes(invFreq), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLALatentStore: mla-invfreq: %w", err)
	}
	defer releaseBuf(fBuf)
	lBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "mla-latcache", Size: uint64(latDim * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(lBuf)
	pBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-store-p", Contents: wgpu.ToBytes([]uint32{uint32(rank), uint32(qkRope), uint32(pos), F32bits(eps), 0, F32bits(ropeScale), BoolU32(interleave), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLALatentStore: mla-store-p: %w", err)
	}
	defer releaseBuf(pBuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaStoreLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: dBuf, Size: dBuf.GetSize()},
		{Binding: 1, Buffer: nBuf, Size: nBuf.GetSize()},
		{Binding: 2, Buffer: fBuf, Size: fBuf.GetSize()},
		{Binding: 3, Buffer: lBuf, Size: lBuf.GetSize()},
		{Binding: 4, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	if err := c.submitUnary(c.mlaStorePipeline, bg, 64); err != nil {
		return nil, err
	}
	return c.readbackRaw(lBuf, latDim)
}

func (c *Context) ensureMLAHeadMV() error {
	if c.mlaHeadMVPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaHeadMatvec", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MlaHeadMatvecWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaHeadMatvec: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaHeadMatvec", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaHeadMatvec: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mlaHeadMVShader, c.mlaHeadMVPipeline, c.mlaHeadMVLayout = sh, pl, c.bgl(pl)
	return nil
}

// MLAHeadMatvec runs the per-head block-diagonal matvec on host inputs and returns dst
// [nH*N]. a is [nH*aStride], w is [nH*N*K] (per head, row-major). Standalone op for parity.
func (c *Context) MLAHeadMatvec(a, w []float32, nH, N, K, aStride int) ([]float32, error) {
	if err := c.ensureMLAHeadMV(); err != nil {
		return nil, err
	}
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-hmv-a", Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAHeadMatvec: mla-hmv-a: %w", err)
	}
	defer releaseBuf(aBuf)
	wBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-hmv-w", Contents: wgpu.ToBytes(w), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAHeadMatvec: mla-hmv-w: %w", err)
	}
	defer releaseBuf(wBuf)
	dBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "mla-hmv-dst", Size: uint64(nH * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(dBuf)
	pBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-hmv-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(N), uint32(K), uint32(aStride), uint32(N), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAHeadMatvec: mla-hmv-p: %w", err)
	}
	defer releaseBuf(pBuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaHeadMVLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
		{Binding: 1, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 2, Buffer: dBuf, Size: dBuf.GetSize()},
		{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAHeadMatvec: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.mlaHeadMVPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(nH * N)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAHeadMatvec: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.readbackRaw(dBuf, nH*N)
}

func (c *Context) ensureMLAQRope() error {
	if c.mlaQRopePipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaQRope", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MlaQRopeWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaQRope: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaQRope", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaQRope: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mlaQRopeShader, c.mlaQRopePipeline, c.mlaQRopeLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMLAAttn() error {
	if c.mlaAttnPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mlaAttn", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MlaAttnWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mlaAttn: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mlaAttn", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mlaAttn: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mlaAttnShader, c.mlaAttnPipeline, c.mlaAttnLayout = sh, pl, c.bgl(pl)
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
	qBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-qabs", Contents: wgpu.ToBytes(qAbs), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAAttn: mla-qabs: %w", err)
	}
	defer releaseBuf(qBuf)
	lBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-lat", Contents: wgpu.ToBytes(lat), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAAttn: mla-lat: %w", err)
	}
	defer releaseBuf(lBuf)
	wBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "mla-wsum", Size: uint64(nH * rank * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(wBuf)
	pBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "mla-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(latDim), uint32(rank), uint32(nKeys), F32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAAttn: mla-p: %w", err)
	}
	defer releaseBuf(pBuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.mlaAttnLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()},
		{Binding: 1, Buffer: lBuf, Size: lBuf.GetSize()},
		{Binding: 2, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAAttn: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.mlaAttnPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), 1, 1) // one workgroup per query head
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MLAAttn: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.readbackRaw(wBuf, nH*rank)
}
