//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

func (c *Context) ensureGEMVBias() error {
	if c.gemvBiasPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      "gemvW8A8Bias",
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.GemvW8A8BiasShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile GEMV-bias shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "gemvW8A8Bias",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create GEMV-bias pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.gemvBiasShader = sh
	c.gemvBiasPipeline = pl
	c.gemvBiasLayout = c.bgl(pl)
	return nil
}

func (c *Context) ensureGEMV() error {
	if c.gemvPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      "gemvW8A8",
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.GemvW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile GEMV shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "gemvW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create GEMV pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.gemvShader = sh
	c.gemvPipeline = pl
	c.gemvLayout = c.bgl(pl)
	return nil
}

// BatchGEMV runs several decode GEMVs that share one activation (the fused
// qkv / gate-up projections) as ONE submit with ONE Poll — the sync-floor cut.
// aq+aScale are the quantized shared activation; rms are the resident weights
// (all the same K). Returns each op's [rms[i].nRows()] output. Buffers are per-call
// (the win here is collapsing N syncs to one, not buffer reuse).
//
// rms is decodeWeight: W8A8 and W4A8 share the same 6-binding layout (decodeWeight's own comment, gemv_w4a8.go), so one
// generic dispatch loop serves both; MatmulW8A8Batch and MatmulW4A8Batch each wrap their concrete resident type before
// calling in. A mixed-precision batch works too (each op's own pipeline is bound just before its dispatch), though callers
// only build homogeneous ones.
func (c *Context) BatchGEMV(aq []int8, aScale float32, rms []DecodeWeight) ([][]float32, error) {
	if len(rms) == 0 {
		return nil, fmt.Errorf("gpu: BatchGEMV: no ops")
	}
	for _, rm := range rms {
		switch rm.(type) {
		case *ResidentW4A8:
			if err := c.ensureGEMVW4(); err != nil {
				return nil, err
			}
		default:
			if err := c.ensureGEMV(); err != nil {
				return nil, err
			}
		}
	}
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "batch-act", Contents: wgpu.ToBytes(packInt8(aq, 1, len(aq))), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchGEMV act: %w", err)
	}
	defer releaseBuf(aBuf)
	asBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "batch-ascale", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchGEMV ascale: %w", err)
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
			if p.dst != nil {
				releaseBuf(p.dst)
			}
			if p.dims != nil {
				releaseBuf(p.dims)
			}
			if p.stag != nil {
				releaseBuf(p.stag)
			}
			if p.bg != nil {
				p.bg.Release()
			}
		}
	}

	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		release()
		return nil, fmt.Errorf("gpu: BatchGEMV: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	for i, rm := range rms {
		N := rm.nRows()
		dst, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "batch-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if err != nil {
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV dst: %w", err)
		}
		dims, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "batch-dims", Contents: wgpu.ToBytes([]uint32{1, uint32(rm.kPad()), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
		if err != nil { // nil dims → nil-panic in CreateBindGroup / the cleanup Release below
			releaseBuf(dst)
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV dims: %w", err)
		}
		bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout: rm.gLayout(c),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
				{Binding: 1, Buffer: rm.wbuf(), Size: rm.wbuf().GetSize()},
				{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
				{Binding: 3, Buffer: rm.sbuf(), Size: rm.sbuf().GetSize()},
				{Binding: 4, Buffer: dst, Size: dst.GetSize()},
				{Binding: 5, Buffer: dims, Size: dims.GetSize()},
			},
		})
		if err != nil {
			releaseBuf(dst)
			releaseBuf(dims)
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV bind: %w", err)
		}
		pops[i] = perOp{dst: dst, dims: dims, bg: bg, n: N}
		pass.SetPipeline(rm.gPipe(c)) // per-op: a mixed-precision batch needs each op's own pipeline
		pass.SetBindGroup(0, bg, nil)
		gx, gy := GemvGrid(N)
		pass.DispatchWorkgroups(gx, gy, 1)
	}
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		release()
		return nil, fmt.Errorf("gpu: BatchGEMV pass: %w", err)
	}
	pass.Release()
	for i := range pops {
		stag, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "batch-stage", Size: uint64(pops[i].n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV stage: %w", err)
		}
		pops[i].stag = stag
		if err := enc.TryCopyBufferToBuffer(pops[i].dst, 0, stag, 0, uint64(pops[i].n*4)); err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV copy: %w", err)
		}
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		release()
		return nil, fmt.Errorf("gpu: BatchGEMV: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	statuses := make([]wgpu.MapAsyncStatus, len(pops))
	for i := range pops {
		idx := i
		if err := pops[i].stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(pops[i].n*4), func(s wgpu.MapAsyncStatus) { statuses[idx] = s }); err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV map: %w", err)
		}
	}
	c.device.Poll(true, nil) // ONE sync for the whole batch
	outs := make([][]float32, len(pops))
	for i := range pops {
		if statuses[i] != wgpu.MapAsyncStatusSuccess {
			release()
			return nil, fmt.Errorf("gpu: BatchGEMV map[%d] failed: %v", i, statuses[i])
		}
		out := make([]float32, pops[i].n)
		copy(out, wgpu.FromBytes[float32](pops[i].stag.GetMappedRange(0, uint(pops[i].n*4))))
		pops[i].stag.TryUnmap()
		outs[i] = out
	}
	release()
	return outs, nil
}

// GEMVRunner is the steady-state decode path: it allocates the activation /
// scale / output / staging buffers and the bind group ONCE for a given resident
// weight, then each Run only WriteBuffers the new activation and dispatches —
// avoiding the per-call buffer churn that dominated the one-shot MatmulW8A8GEMV
// (GPU sync is cheap; wgpu buffer creation is not). This is what the decoder
// caches per weight matrix.
//
// rm is decodeWeight, either W8A8 or W4A8: both kernels share the same 6-binding layout (decodeWeight's own comment in
// gemv_w4a8.go), so this runner is precision-agnostic the way DecodeRunner's own gemv/gemvAdd builders are.
type GEMVRunner struct {
	c                                  *Context
	rm                                 DecodeWeight
	aBuf, asBuf, dstBuf, dimsBuf, stag *wgpu.Buffer
	bg                                 *wgpu.BindGroup
	n                                  int
}

// NewGEMVRunner builds a reusable decode runner for a resident weight (W8A8 or W4A8).
func (c *Context) NewGEMVRunner(rm DecodeWeight) (*GEMVRunner, error) {
	// Each precision compiles its own pipeline lazily; gPipe(c)/gLayout(c) below assume
	// whichever one this rm needs has already been ensured.
	switch rm.(type) {
	case *ResidentW4A8:
		if err := c.ensureGEMVW4(); err != nil {
			return nil, err
		}
	default:
		if err := c.ensureGEMV(); err != nil {
			return nil, err
		}
	}
	N := rm.nRows()
	kp := rm.kPad()
	mk := func(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	aBuf, err := mk("gemvr-act", uint64(kp/4*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return nil, err
	}
	// Check each alloc: a failed CreateBuffer returns nil, which would flow into CreateBindGroup and then nil-panic in the
	// error-cleanup Release() below. Release the already-allocated buffers on the way out.
	asBuf, err := mk("gemvr-ascale", 4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		releaseBuf(aBuf)
		return nil, err
	}
	dstBuf, err := mk("gemvr-dst", uint64(N*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		releaseBuf(aBuf)
		releaseBuf(asBuf)
		return nil, err
	}
	stag, err := mk("gemvr-stage", uint64(N*4), wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	if err != nil {
		releaseBuf(aBuf)
		releaseBuf(asBuf)
		releaseBuf(dstBuf)
		return nil, err
	}
	dimsBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "gemvr-dims", Contents: wgpu.ToBytes([]uint32{1, uint32(kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		releaseBuf(aBuf)
		releaseBuf(asBuf)
		releaseBuf(dstBuf)
		releaseBuf(stag)
		return nil, err
	}
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: rm.gLayout(c),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: rm.wbuf(), Size: rm.wbuf().GetSize()},
			{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
			{Binding: 3, Buffer: rm.sbuf(), Size: rm.sbuf().GetSize()},
			{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		releaseBuf(aBuf)
		releaseBuf(asBuf)
		releaseBuf(dstBuf)
		releaseBuf(stag)
		releaseBuf(dimsBuf)
		return nil, err
	}
	return &GEMVRunner{c: c, rm: rm, aBuf: aBuf, asBuf: asBuf, dstBuf: dstBuf, dimsBuf: dimsBuf, stag: stag, bg: bg, n: N}, nil
}

// Run computes dst[N] = (aq quantized int8) · rmᵀ, reusing the runner's buffers. aq must be
// exactly K elements (no caller-side padding) — linalg.QuantizeRowsInt8(a, 1, K) already
// produces that, and aBuf was sized to kPad() bytes at construction time, so the untouched tail
// stays correctly zero (CreateBuffer zero-inits) regardless of how kPad() rounds K.
func (r *GEMVRunner) Run(aq []int8, aScale float32) ([]float32, error) {
	c := r.c
	if err := c.queue.TryWriteBuffer(r.aBuf, 0, wgpu.ToBytes(packInt8(aq, 1, len(aq)))); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner write act: %w", err)
	}
	if err := c.queue.TryWriteBuffer(r.asBuf, 0, wgpu.ToBytes([]float32{aScale})); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner write scale: %w", err)
	}
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: Run: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(r.rm.gPipe(c))
	pass.SetBindGroup(0, r.bg, nil)
	gx, gy := GemvGrid(r.n)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: GEMVRunner pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(r.dstBuf, 0, r.stag, 0, uint64(r.n*4)); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner copy: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: Run: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := r.stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(r.n*4), func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: GEMVRunner map failed: %v", status)
	}
	out := make([]float32, r.n)
	copy(out, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.n*4))))
	if err := r.stag.TryUnmap(); err != nil {
		return nil, fmt.Errorf("gpu: GEMVRunner unmap: %w", err)
	}
	return out, nil
}

// Close frees the runner's buffers (not the resident weight).
func (r *GEMVRunner) Close() error {
	r.c.releaseOwned(func() {
		releaseBuf(r.aBuf)
		releaseBuf(r.asBuf)
		releaseBuf(r.dstBuf)
		releaseBuf(r.dimsBuf)
		releaseBuf(r.stag)
		r.bg.Release()
	})
	return nil
}

// GemvGrid maps N output columns to a 2D workgroup grid (X capped at 32768 so
// vocab-sized N stays under the 65535/dimension limit). The kernel reconstructs
// n = x + y*32768.
func GemvGrid(n int) (uint32, uint32) {
	if n <= 32768 {
		return uint32(n), 1
	}
	return 32768, uint32((n + 32767) / 32768)
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
	aBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "gemv-act", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV act buffer: %w", err)
	}
	defer releaseBuf(aBuf)
	asBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "gemv-ascale", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV ascale buffer: %w", err)
	}
	defer releaseBuf(asBuf)

	dstBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{
		Label: "gemv-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV dst buffer: %w", err)
	}
	defer releaseBuf(dstBuf)
	dims := []uint32{1, uint32(rm.kp), uint32(N), 0}
	dimsBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{
		Label: "gemv-dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV dims buffer: %w", err)
	}
	defer releaseBuf(dimsBuf)
	stage, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{
		Label: "gemv-stage", Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: GEMV staging: %w", err)
	}
	defer releaseBuf(stage)

	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{
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
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8GEMV: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.gemvPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: GEMV pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(dstBuf, 0, stage, 0, uint64(N*4)); err != nil {
		return nil, fmt.Errorf("gpu: GEMV copy: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: MatmulW8A8GEMV: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := stage.TryMapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: GEMV map: %w", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: GEMV map failed: %v", status)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(N*4))))
	if err := stage.TryUnmap(); err != nil {
		return nil, fmt.Errorf("gpu: GEMV unmap: %w", err)
	}
	return out, nil
}
