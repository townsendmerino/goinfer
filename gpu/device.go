//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Stage 2 — device-resident activations. At M=1 decode the activation is a few
// KB, so the per-token cost isn't data movement, it's LATENCY: the Stage-1
// MatmulW8A8 calls Poll(true) to map its readback, and ~7 matmuls × N layers =
// hundreds of synchronous round-trips per token. Keeping activations in device
// buffers lets a chain of matmuls submit back-to-back and sync ONCE.
//
// Chaining W8A8→W8A8 needs the one glue op that was on the CPU — int8
// re-quantization of a matmul's f32 output — moved onto the GPU. quantizeShader
// does it: one workgroup per row computes the row max-abs (naive serial reduce
// on lane 0 — trivial at decode; the rows run in parallel), then all lanes
// quantize+pack to the same 4×int8/u32 layout MatmulW8A8 consumes.

const quantizeShaderWGSL = `
struct QDims { m: u32, n: u32, np: u32, _p: u32 };  // np = N padded to mult of 4

@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [M, N]
@group(0) @binding(1) var<storage, read_write> qout:   array<u32>;  // [M, np/4] packed int8
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;  // [M]
@group(0) @binding(3) var<uniform>             d:      QDims;

var<workgroup> shScale: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = wid.x;
    if (m >= d.m) { return; }
    let base = m * d.n;
    if (lid.x == 0u) {
        var mx: f32 = 0.0;
        for (var i: u32 = 0u; i < d.n; i = i + 1u) {
            let v = abs(src[base + i]);
            if (v > mx) { mx = v; }
        }
        var s: f32 = mx / 127.0;
        if (s == 0.0) { s = 1.0; }
        shScale = s;
        scales[m] = s;
    }
    workgroupBarrier();
    let inv = 1.0 / shScale;
    let nw = d.np / 4u;
    let obase = m * nw;
    for (var w: u32 = lid.x; w < nw; w = w + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = w * 4u + j;
            var q: i32 = 0;
            if (k < d.n) {
                q = i32(round(src[base + k] * inv));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[obase + w] = word;
    }
}
`

// DeviceBuffer is a GPU storage buffer that stays resident between ops (an
// activation flowing through a layer). n is its element count (f32 or u32
// depending on the producer). Release frees it.
type DeviceBuffer struct {
	buf *wgpu.Buffer
	n   int
}

func (d *DeviceBuffer) Close() error {
	if d != nil && d.buf != nil {
		d.buf.Release()
		d.buf = nil
	}
	return nil
}

func (c *Context) ensureQuantize() error {
	if c.quantizePipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "quantizeRowsInt8",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: quantizeShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile quantize shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "quantizeRowsInt8",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create quantize pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.quantizeShader = sh
	c.quantizePipeline = pl
	c.quantizeLayout = pl.GetBindGroupLayout(0)
	return nil
}

// UploadF32 puts a host f32 slice into a resident device buffer (the chain's
// input activation). The caller Releases it.
func (c *Context) UploadF32(a []float32) (*DeviceBuffer, error) {
	buf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "act-f32", Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: UploadF32: %w", err)
	}
	return &DeviceBuffer{buf: buf, n: len(a)}, nil
}

// quantizeDevice quantizes an f32 device activation [M,K] to packed int8 + per-row
// scales, both resident. It only Submits (no Poll) so it pipelines with the
// matmul that consumes it.
func (c *Context) quantizeDevice(src *DeviceBuffer, M, K int) (*DeviceBuffer, *DeviceBuffer, error) {
	if err := c.ensureQuantize(); err != nil {
		return nil, nil, err
	}
	kp := padK(K)
	qWords := M * (kp / 4)
	qBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "act-q8", Size: uint64(qWords * 4), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("gpu: quantizeDevice q buffer: %w", err)
	}
	scBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "act-scales", Size: uint64(M * 4), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		qBuf.Release()
		return nil, nil, fmt.Errorf("gpu: quantizeDevice scales buffer: %w", err)
	}
	dims := []uint32{uint32(M), uint32(K), uint32(kp), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "qdims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		qBuf.Release()
		scBuf.Release()
		return nil, nil, fmt.Errorf("gpu: quantizeDevice dims: %w", err)
	}
	defer dimsBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.quantizeLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: src.buf, Size: src.buf.GetSize()},
			{Binding: 1, Buffer: qBuf, Size: qBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		qBuf.Release()
		scBuf.Release()
		return nil, nil, fmt.Errorf("gpu: quantizeDevice bind: %w", err)
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.quantizePipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(M), 1, 1) // one workgroup per row
	if err := pass.End(); err != nil {
		pass.Release()
		qBuf.Release()
		scBuf.Release()
		return nil, nil, fmt.Errorf("gpu: quantizeDevice pass: %w", err)
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd) // no Poll — pipelines with the consumer
	return &DeviceBuffer{buf: qBuf, n: qWords}, &DeviceBuffer{buf: scBuf, n: M}, nil
}

// matmulW8A8Device computes dst = (int8 act) · rmᵀ into a resident f32 device
// buffer, Submitting only (no Poll). The output feeds the next layer's quantize.
func (c *Context) matmulW8A8Device(aq, aScales *DeviceBuffer, rm *ResidentW8A8, M int) (*DeviceBuffer, error) {
	if err := c.ensureQuant(); err != nil {
		return nil, err
	}
	N := rm.rows
	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "mm-dst", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: matmulW8A8Device dst: %w", err)
	}
	dims := []uint32{uint32(M), uint32(rm.kp), uint32(N), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "mm-dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		dstBuf.Release()
		return nil, fmt.Errorf("gpu: matmulW8A8Device dims: %w", err)
	}
	defer dimsBuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.quantLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aq.buf, Size: aq.buf.GetSize()},
			{Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
			{Binding: 2, Buffer: aScales.buf, Size: aScales.buf.GetSize()},
			{Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
			{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		dstBuf.Release()
		return nil, fmt.Errorf("gpu: matmulW8A8Device bind: %w", err)
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.quantPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		dstBuf.Release()
		return nil, fmt.Errorf("gpu: matmulW8A8Device pass: %w", err)
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd) // no Poll
	return &DeviceBuffer{buf: dstBuf, n: M * N}, nil
}

// Readback copies a device f32 buffer to the host — the single sync point at the
// end of a chain.
func (c *Context) Readback(db *DeviceBuffer) ([]float32, error) {
	size := uint64(db.n * 4)
	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "readback", Size: size, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: Readback staging: %w", err)
	}
	defer stage.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	if err := enc.CopyBufferToBuffer(db.buf, 0, stage, 0, size); err != nil {
		return nil, fmt.Errorf("gpu: Readback copy: %w", err)
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, size, func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return nil, fmt.Errorf("gpu: Readback map: %w", err)
	}
	c.device.Poll(true, nil) // the one sync
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: Readback map failed: %v", status)
	}
	out := make([]float32, db.n)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(size))))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: Readback unmap: %w", err)
	}
	return out, nil
}

// ChainW8A8 runs act through a sequence of resident W8A8 weights — each step
// quantizes the running activation on-device and matmuls — keeping everything
// resident and syncing only once at the end. weights[i] must have cols == the
// output width of weights[i-1] (square, in the proof). This is the Stage-2 path:
// one Poll for the whole chain instead of one per matmul.
func (c *Context) ChainW8A8(act []float32, M int, weights []*ResidentW8A8) ([]float32, error) {
	cur, err := c.UploadF32(act)
	if err != nil {
		return nil, err
	}
	for _, rm := range weights {
		qb, sb, err := c.quantizeDevice(cur, M, rm.cols)
		if err != nil {
			cur.Close()
			return nil, err
		}
		out, err := c.matmulW8A8Device(qb, sb, rm, M)
		qb.Close()
		sb.Close()
		cur.Close()
		if err != nil {
			return nil, err
		}
		cur = out
	}
	defer cur.Close()
	return c.Readback(cur)
}
