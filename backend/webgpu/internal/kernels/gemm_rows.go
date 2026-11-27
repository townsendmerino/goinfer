//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Thin-M W8A8 GEMM for the Stage-B verify (docs/spec/07). The 16×16 tiled GEMM
// (gemm.go) is the PREFILL kernel; at the small M of a speculative block (M≈K+1≈8)
// it wastes half its 16-row tile and loses to per-row GEMV. This
// kernel keeps the GEMV's structure — one WORKGROUP per output column n, 64 lanes
// coalesce-stride the contiguous weight row — but loads each weight vec4 ONCE and
// accumulates it against ALL M activation rows (M register accumulators). So the
// weight matrix streams once for the whole block (the Stage-B win) with NO wasted
// tile dimension. Arithmetic is identical to M separate GEMVs ⇒ bit-parity.
const gemmRowMaxM = 16

func (c *Context) ensureGemmRow() error {
	if c.gemmRowPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      "gemmRowW8A8",
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.GemmRowW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile gemmRow shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "gemmRowW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create gemmRow pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.gemmRowShader = sh
	c.gemmRowPipeline = pl
	c.gemmRowLayout = c.bgl(pl)
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
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "gemmrow-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(aBuf)
	asBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "gemmrow-ascale", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(asBuf)
	dstBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "gemmrow-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(dstBuf)
	dimsBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "gemmrow-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(dimsBuf)
	stage, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "gemmrow-stage", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(stage)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{
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
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.gemmRowPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow: pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(dstBuf, 0, stage, 0, uint64(M*N*4)); err != nil {
		return nil, err
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := stage.TryMapAsync(wgpu.MapModeRead, 0, uint64(M*N*4), func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: MatmulW8A8GemmRow map failed: %v", status)
	}
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(M*N*4))))
	if err := stage.TryUnmap(); err != nil {
		return nil, err
	}
	return out, nil
}
