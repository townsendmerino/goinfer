//go:build gpu

package gpu

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Tiled W8A8 GEMM (the prefill, M>1 kernel). The naive matmul re-streams a full
// weight row and activation row from global memory per output element — fine when
// bandwidth-bound at M=1, but at prefill it's compute/traffic-bound and wastes
// reuse. This stages 16×16 int8 tiles of the activation and weight into workgroup
// shared memory, so each loaded element is reused across the tile (arithmetic
// intensity ↑). One workgroup computes a 16×16 output block; K is walked in
// 16-wide strips (kp is a multiple of 16). Math identical to MatmulBTW8A8.
//
// The inner product over each 4-int8-packed word is factored out as dot4 so the
// tiling/scheduling code is written once and shared by two dot4 implementations
// (see below) rather than duplicating the whole kernel.
const matmulTiledW8A8KernelWGSL = `
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

%s

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

// dot4UnpackedWGSL: every backend accepts this — four sign-extended int8 unpacked
// from the unpackged word and multiply-accumulated in scalar ALU ops.
const dot4UnpackedWGSL = `
fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dot4(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}
`

// dot4PackedWGSL: the native WGSL builtin (gfx-rs/wgpu PR #7494/#7595), only used
// when Context.hasDP4A (probed live in New()) — on a backend that lowers it to real
// hardware DP4A (e.g. Vulkan's VK_KHR_shader_integer_dot_product on a DP4A-capable
// NVIDIA part), this is what clears the compute/bandwidth wall the tiled GEMM was
// gated on (docs/task-gpu-batched-prefill.md). Same math as the unpacked fallback —
// dot4I8Packed treats the u32 as four little-endian sign-extended int8 lanes,
// verified against the unpacked path's parity gate (TestMatmulW8A8Tiled_dp4aParity).
const dot4PackedWGSL = `
fn dot4(a: u32, b: u32) -> i32 {
    return dot4I8Packed(a, b);
}
`

var (
	matmulTiledW8A8ShaderWGSL     = fmt.Sprintf(matmulTiledW8A8KernelWGSL, dot4UnpackedWGSL)
	matmulTiledW8A8DP4AShaderWGSL = fmt.Sprintf(matmulTiledW8A8KernelWGSL, dot4PackedWGSL)
)

// matmulTiledW8A8BiasKernelWGSL is matmulTiledW8A8KernelWGSL with one addition: a
// per-output-column bias, added in the SAME expression as the dequant multiply
// (`f32(acc) * aScales[row] * bScales[col] + bias[col]`) — textually identical
// operand order to gemvW8A8BiasShaderWGSL's epilogue (gemv.go), which is the
// point. PrefillLastW8A8 used to do this as TWO dispatches (this kernel without
// bias, then a separate residualShaderWGSL add) — mathematically the same
// formula, but forced through an f32 round-trip through memory between the
// multiply and the add, which a single WGSL expression may evaluate with an FMA
// contraction the compiler cannot apply across two separate dispatches. Measured
// real effect (TestLocalize_BiasEpilogue, qwen2.5-coder-0.5b layer 0, real
// weights): 98 of 896 elements differed between the two forms, all within 2.4e-7
// absolute — tiny in f32 terms, but large enough that a subsequent int8 quantize
// of an element sitting near a rounding boundary can flip which bucket it lands
// in, and 24 layers of that compounds into the observed cosine ~0.99x /
// maxAbs ~1 divergence in final logits. This kernel exists so PrefillLastW8A8
// can dispatch the identical single-expression epilogue gemvBias does, for M
// rows at once instead of M separate GEMV calls.
const matmulTiledW8A8BiasKernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;
@group(0) @binding(6) var<storage, read>       bias:    array<f32>;  // [N]

const TS: u32 = 16u;
var<workgroup> As: array<u32, 256>;
var<workgroup> Bs: array<u32, 256>;

%s

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;
    let row = wid.y * TS + lid.y;
    let col = wid.x * TS + lid.x;
    let nStrips = (kw + TS - 1u) / TS;
    var acc: i32 = 0;
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let wcol = s * TS + lid.x;
        var aw: u32 = 0u;
        if (row < dims.m && wcol < kw) { aw = aq[row * kw + wcol]; }
        As[lid.y * TS + lid.x] = aw;
        let brow = wid.x * TS + lid.y;
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
        dst[row * dims.n + col] = f32(acc) * aScales[row] * bScales[col] + bias[col];
    }
}
`

var (
	matmulTiledW8A8BiasShaderWGSL     = fmt.Sprintf(matmulTiledW8A8BiasKernelWGSL, dot4UnpackedWGSL)
	matmulTiledW8A8DP4ABiasShaderWGSL = fmt.Sprintf(matmulTiledW8A8BiasKernelWGSL, dot4PackedWGSL)
)

// ensureTiledBias compiles the bias-epilogue tiled GEMM (see
// matmulTiledW8A8BiasKernelWGSL) — a separate pipeline from ensureTiled's, chosen
// the same way (c.hasDP4A), since a bias-taking kernel needs its own bind group
// layout (7 bindings vs 6).
func (c *Context) ensureTiledBias() error {
	if c.tiledBiasPipeline != nil {
		return nil
	}
	code, label := matmulTiledW8A8BiasShaderWGSL, "matmulTiledW8A8Bias"
	if c.hasDP4A {
		code, label = matmulTiledW8A8DP4ABiasShaderWGSL, "matmulTiledW8A8DP4ABias"
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      label,
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile tiled-bias shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   label,
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create tiled-bias pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.tiledBiasShader = sh
	c.tiledBiasPipeline = pl
	c.tiledBiasLayout = c.bgl(pl)
	return nil
}

func (c *Context) ensureTiled() error {
	if c.tiledPipeline != nil {
		return nil
	}
	code, label := matmulTiledW8A8ShaderWGSL, "matmulTiledW8A8"
	if c.hasDP4A {
		code, label = matmulTiledW8A8DP4AShaderWGSL, "matmulTiledW8A8DP4A"
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      label,
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile tiled shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   label,
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
	aBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: BatchTiled act: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
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
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	for i, rm := range rms {
		N := rm.rows
		dst, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Label: "btiled-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if err != nil {
			pass.Release()
			release()
			return nil, fmt.Errorf("gpu: BatchTiled dst: %w", err)
		}
		dims, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "btiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
		if err != nil { // nil dims → nil-panic in CreateBindGroup / the cleanup Release below (audit R-06)
			dst.Release()
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
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		release()
		return nil, fmt.Errorf("gpu: BatchTiled pass: %w", err)
	}
	pass.Release()
	for i := range pops {
		stag, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Label: "btiled-stage", Size: uint64(M * pops[i].n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			release()
			return nil, fmt.Errorf("gpu: BatchTiled stage: %w", err)
		}
		pops[i].stag = stag
		enc.TryCopyBufferToBuffer(pops[i].dst, 0, stag, 0, uint64(M*pops[i].n*4))
	}
	cmd, _ := enc.TryFinish(nil)
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
	aBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-act", Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled act buffer: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled ascales buffer: %w", err)
	}
	defer asBuf.Release()

	dstSize := uint64(M * N * 4)
	dstBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{
		Label: "tiled-dst", Size: dstSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dst buffer: %w", err)
	}
	defer dstBuf.Release()
	dimsBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "tiled-dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled dims buffer: %w", err)
	}
	defer dimsBuf.Release()
	stage, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{
		Label: "tiled-stage", Size: dstSize, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: tiled staging buffer: %w", err)
	}
	defer stage.Release()
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
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.tiledPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(N)+15)/16, (uint32(M)+15)/16, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: tiled pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: tiled copy: %w", err)
	}
	cmd, _ := enc.TryFinish(nil)
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
