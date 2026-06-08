//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// W8A8 (int8×int8) matmul on the GPU — the Stage-1 "quantized matmul" unlock.
// Decode is memory-bandwidth-bound (every token reads all resident weights), so
// the win is storing weights as 1-byte int8 PACKED 4-per-u32 (≈4× less traffic
// than f32), not the multiply itself. WebGPU has no i8 type and (in this binding)
// no dot4I8Packed builtin, so the kernel unpacks four sign-extended int8 per u32
// and accumulates in i32 — a few extra ALU ops that don't matter when the kernel
// is bandwidth-bound. Math matches linalg.MatmulBTW8A8 exactly:
//
//	dst[m,n] = (Σ_k aq[m,k]·bq[n,k]) · aScale[m] · bScale[n]
//
// with per-row int8 quantization on both sides. K is padded to a multiple of 4
// (the pad int8s are 0 on both sides → contribute nothing).
const matmulW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };  // kp = K padded to mult of 4

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8 activations
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4] packed int8 weights
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

// unpack four little-endian sign-extended int8 from a u32 (arithmetic >> on i32
// sign-extends per WGSL spec).
fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(
        i32(w << 24u) >> 24u,
        i32(w << 16u) >> 24u,
        i32(w <<  8u) >> 24u,
        i32(w)        >> 24u,
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;  // m
    let col = gid.y;  // n
    if (row >= dims.m || col >= dims.n) { return; }
    let kw = dims.kp / 4u;
    let aBase = row * kw;
    let bBase = col * kw;
    var acc: i32 = 0;
    for (var w: u32 = 0u; w < kw; w = w + 1u) {
        let av = unpack_i8x4(aq[aBase + w]);
        let bv = unpack_i8x4(bq[bBase + w]);
        acc = acc + av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
    }
    dst[row * dims.n + col] = f32(acc) * aScales[row] * bScales[col];
}
`

// ensureQuant lazily compiles the W8A8 pipeline (kept off the New() hot path so
// f32-only callers don't pay for it).
func (c *Context) ensureQuant() error {
	if c.quantPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "matmulW8A8",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: matmulW8A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile W8A8 shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "matmulW8A8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create W8A8 pipeline: %w", err)
	}
	c.quantShader = sh
	c.quantPipeline = pl
	c.quantLayout = pl.GetBindGroupLayout(0)
	return nil
}

// ResidentW8A8 is an int8 weight [N,K] uploaded once (packed 4-per-u32) with its
// per-row scales — reused across every token's matmul against it.
type ResidentW8A8 struct {
	bq      *wgpu.Buffer // [N, kp/4] packed int8
	bScales *wgpu.Buffer // [N] f32
	rows    int          // N
	cols    int          // K (unpadded)
	kp      int          // K padded to mult of 4
}

// Release frees the resident GPU buffers.
func (rm *ResidentW8A8) Release() {
	if rm.bq != nil {
		rm.bq.Release()
		rm.bq = nil
	}
	if rm.bScales != nil {
		rm.bScales.Release()
		rm.bScales = nil
	}
}

// padK rounds K up to a multiple of 4.
func padK(k int) int { return (k + 3) &^ 3 }

// packInt8 packs a [rows, cols] int8 matrix into [rows, padK(cols)/4] u32 words,
// little-endian, zero-padding each row to a multiple of 4.
func packInt8(q []int8, rows, cols int) []uint32 {
	kp := padK(cols)
	kw := kp / 4
	out := make([]uint32, rows*kw)
	for r := 0; r < rows; r++ {
		src := q[r*cols : r*cols+cols]
		dst := out[r*kw : r*kw+kw]
		for w := 0; w < kw; w++ {
			var word uint32
			for j := 0; j < 4; j++ {
				if k := w*4 + j; k < cols {
					word |= uint32(uint8(src[k])) << (8 * j)
				}
			}
			dst[w] = word
		}
	}
	return out
}

// UploadW8A8 uploads an int8 weight matrix [N,K] + per-row scales [N] to resident
// GPU buffers (packed int8). Reused by MatmulW8A8; the caller must Release it.
func (c *Context) UploadW8A8(q8 []int8, scales []float32, N, K int) (*ResidentW8A8, error) {
	if N <= 0 || K <= 0 {
		return nil, fmt.Errorf("gpu: UploadW8A8 non-positive dim N=%d K=%d", N, K)
	}
	if len(q8) < N*K || len(scales) < N {
		return nil, fmt.Errorf("gpu: UploadW8A8 input too small: q8=%d (need %d) scales=%d (need %d)", len(q8), N*K, len(scales), N)
	}
	packed := packInt8(q8, N, K)
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w8a8-weight", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 weight buffer: %w", err)
	}
	sc, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w8a8-bscales", Contents: wgpu.ToBytes(scales[:N]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: create W8A8 scales buffer: %w", err)
	}
	return &ResidentW8A8{bq: bq, bScales: sc, rows: N, cols: K, kp: padK(K)}, nil
}

// MatmulW8A8 computes dst[M,N] = (aq quantized int8) · rm.bᵀ, dequantized — the
// activation int8 aq [M,K] + per-row scales [M] are uploaded each call (tiny at
// M=1 decode); the weight rm stays resident. Returns the [M,N] f32 result.
func (c *Context) MatmulW8A8(aq []int8, aScales []float32, rm *ResidentW8A8, M int) ([]float32, error) {
	if err := c.ensureQuant(); err != nil {
		return nil, err
	}
	K, N := rm.cols, rm.rows
	if M <= 0 {
		return nil, fmt.Errorf("gpu: MatmulW8A8 non-positive M=%d", M)
	}
	if len(aq) < M*K || len(aScales) < M {
		return nil, fmt.Errorf("gpu: MatmulW8A8 input too small: aq=%d (need %d) aScales=%d (need %d)", len(aq), M*K, len(aScales), M)
	}
	packed := packInt8(aq, M, K)
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w8a8-act", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 act buffer: %w", err)
	}
	defer aBuf.Release()
	asBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w8a8-ascales", Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 ascales buffer: %w", err)
	}
	defer asBuf.Release()
	return c.runQuant(aBuf, asBuf, rm, M, N)
}

// runQuant dispatches the W8A8 pipeline and reads back the [M,N] f32 result.
func (c *Context) runQuant(aBuf, asBuf *wgpu.Buffer, rm *ResidentW8A8, M, N int) ([]float32, error) {
	dstSize := uint64(M * N * 4)
	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "w8a8-dst", Size: dstSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 dst buffer: %w", err)
	}
	defer dstBuf.Release()

	dims := []uint32{uint32(M), uint32(rm.kp), uint32(N), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w8a8-dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 dims buffer: %w", err)
	}
	defer dimsBuf.Release()

	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "w8a8-stage", Size: dstSize, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 staging buffer: %w", err)
	}
	defer stage.Release()

	bindGroup, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.quantLayout,
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
		return nil, fmt.Errorf("gpu: create W8A8 bind group: %w", err)
	}
	defer bindGroup.Release()

	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: create W8A8 encoder: %w", err)
	}
	defer enc.Release()

	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.quantPipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: end W8A8 pass: %w", err)
	}
	pass.Release()

	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: copy W8A8 dst→stage: %w", err)
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: finish W8A8 encoder: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	mapStatus := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.BufferMapAsyncStatus) { mapStatus = s }); err != nil {
		return nil, fmt.Errorf("gpu: W8A8 map async: %w", err)
	}
	c.device.Poll(true, nil)
	if mapStatus != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: W8A8 staging map failed: %v", mapStatus)
	}
	raw := stage.GetMappedRange(0, uint(dstSize))
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](raw))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: W8A8 unmap staging: %w", err)
	}
	return out, nil
}
