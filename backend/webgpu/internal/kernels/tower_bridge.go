//go:build gpu

package kernels

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// TowerPipe is a compiled single-bind-group compute pipeline used by backend/webgpu/internal/towers.
type TowerPipe struct {
	pl  *wgpu.ComputePipeline
	lay *wgpu.BindGroupLayout
}

// Raw returns the underlying WebGPU buffer handle.
func (b *DeviceBuffer) Raw() *wgpu.Buffer { return b.buf }

// EnsureVision compiles the shared vision primitive pipelines on c if not already built.
func (c *Context) EnsureVision() error { return c.ensureVision() }

// MakeTowerPipe compiles a standalone compute pipeline from WGSL source for a tower.
func (c *Context) MakeTowerPipe(label, code string) (TowerPipe, error) {
	_, pl, lay, err := c.mkPipeline(label, code)
	return TowerPipe{pl: pl, lay: lay}, err
}

// KernelN runs k over bufs (bindings 0..len(bufs)-1) and a uniform holding p (the next binding), gx by gy
// workgroups. No Poll.
func (c *Context) KernelN(k TowerPipe, bufs []*wgpu.Buffer, p []uint32, gx, gy int) error {
	pbuf, err := c.tryCreateBufferInit(mem.LineCompanion, &wgpu.BufferInitDescriptor{Label: "kn-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer releaseBuf(pbuf)
	entries := make([]wgpu.BindGroupEntry, 0, len(bufs)+1)
	for i, b := range bufs {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()})
	}
	entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(bufs)), Buffer: pbuf, Size: pbuf.GetSize()})
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: k.lay, Entries: entries})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("gpu: kernelN: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(k.pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(gx), uint32(gy), 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return fmt.Errorf("gpu: kernelN: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// NewDevBuf allocates a float32 storage+copy-src DeviceBuffer of n elements on the companion plan line.
func (c *Context) NewDevBuf(n int) (*DeviceBuffer, error) {
	b, err := c.tryCreateBuffer(mem.LineCompanion, &wgpu.BufferDescriptor{Label: "vbuf", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	return NewDeviceBuffer(b, n), nil
}

// NewU32StorageBuf uploads a uint32 slice into a storage buffer on the companion plan line.
func (c *Context) NewU32StorageBuf(label string, data []uint32) (*DeviceBuffer, error) {
	b, err := c.tryCreateBufferInit(mem.LineCompanion, &wgpu.BufferInitDescriptor{
		Label: label, Contents: wgpu.ToBytes(data), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}
	return NewDeviceBuffer(b, len(data)), nil
}

// AddRowsDevice runs the elementwise add/bias-add kernel in place on dst.
func (c *Context) AddRowsDevice(dst, add *wgpu.Buffer, n, cols, mode int) error {
	return c.addRowsDevice(dst, add, n, cols, mode)
}

// CopyHeadDevice gathers or scatters one attention head between interleaved and contiguous layouts.
func (c *Context) CopyHeadDevice(src, dst *wgpu.Buffer, rows, hd, srcStride, off, dir int) error {
	return c.copyHeadDevice(src, dst, rows, hd, srcStride, off, dir)
}

// LayerNormRowsDevice runs row-wise LayerNorm into dst.
func (c *Context) LayerNormRowsDevice(src, weight, bias, dst *wgpu.Buffer, rows, h int, eps float32) error {
	return c.layerNormRowsDevice(src, weight, bias, dst, rows, h, eps)
}

// MatmulF32Device computes a [M, K] * b^T [N, K] into a new [M, N] DeviceBuffer.
func (c *Context) MatmulF32Device(a, b *wgpu.Buffer, M, K, N int) (*DeviceBuffer, error) {
	return c.matmulF32Device(a, b, M, K, N)
}

// SoftmaxInplace runs the shared row-wise softmax kernel in place on x.
func (c *Context) SoftmaxInplace(x *wgpu.Buffer, p []uint32, dispatch int) error {
	return c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, x, p, dispatch)
}

// GELUInplace runs the shared tanh-GELU kernel in place on x.
func (c *Context) GELUInplace(x *wgpu.Buffer, p []uint32, dispatch int) error {
	return c.inplaceKernel(c.geluPipeline, c.geluLayout, x, p, dispatch)
}

// ReleaseOwned runs fn while the device waits for in-flight work before releasing owned buffers.
func (c *Context) ReleaseOwned(fn func()) { c.releaseOwned(fn) }

// PollWait blocks until submitted GPU work on c completes.
func (c *Context) PollWait() { c.device.Poll(true, nil) }

// EnsureQuant compiles the quantized-projection pipelines on c if not already built.
func (c *Context) EnsureQuant() error { return c.ensureQuant() }

// QuantizeDevice quantizes a float32 [M, K] buffer to int8 activations and per-row scales.
func (c *Context) QuantizeDevice(src *DeviceBuffer, M, K int) (*DeviceBuffer, *DeviceBuffer, error) {
	return c.quantizeDevice(src, M, K)
}

// MatmulW8A8Device computes a W8A8 matmul against a resident int8 weight matrix.
func (c *Context) MatmulW8A8Device(aq, as *DeviceBuffer, rm *ResidentW8A8, M int) (*DeviceBuffer, error) {
	return c.matmulW8A8Device(aq, as, rm, M)
}

// Rows returns the output dimension of the resident matrix.
func (rm *ResidentW8A8) Rows() int { return rm.rows }

// Cols returns the input dimension of the resident matrix.
func (rm *ResidentW8A8) Cols() int { return rm.cols }

// Len returns the element count of b.
func (b *DeviceBuffer) Len() int { return b.n }

// NewResidentW8A8Shape constructs a metadata-only ResidentW8A8 for shape validation tests.
func NewResidentW8A8Shape(rows, cols int) *ResidentW8A8 { return &ResidentW8A8{rows: rows, cols: cols} }
