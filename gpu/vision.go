//go:build gpu

package gpu

import (
	"math"

	"github.com/cogentcore/webgpu/wgpu"
)

// Resident SigLIP vision encoder — the path to a GPU-fast image prefill
// (docs/completed/task-gpu-vision-tower.md). Per-call matmul offload was a measured dead
// end (WebGPU's ~1s submit+readback overhead × ~162 matmuls/forward ≈ the whole
// runtime). The fix is residency: keep the [np, hidden] activation in device
// buffers through all layers, chaining each op as a Submit with NO Poll (queue
// order guarantees the dependency), so the forward syncs once. This file builds
// the batched (M = np patches) kernels the encoder needs that the decode path
// lacks: standard LayerNorm (mean/var, vs RMSNorm), gelu-tanh (vs silu-gated),
// and a bidirectional softmax — composed with the existing device matmul /
// quantize / residual primitives.

// layerNormRowsWGSL: standard LayerNorm over each of `rows` independent rows of
// width `h`: y = (x-mean)/sqrt(var+eps) * weight + bias, population variance
// (var = mean(x²) - mean(x)²), matching HF nn.LayerNorm. One workgroup per row.
const layerNormRowsWGSL = `
struct P { rows: u32, h: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>; // [rows, h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>; // [h]
@group(0) @binding(2) var<storage, read>       bias:   array<f32>; // [h]
@group(0) @binding(3) var<storage, read_write> dst:    array<f32>; // [rows, h]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> smean: array<f32, 64>;
var<workgroup> svar:  array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.h;
    let t = lid.x;
    var s: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[base + i]; s = s + v; s2 = s2 + v*v; }
    smean[t] = s; svar[t] = s2;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { smean[t] = smean[t] + smean[t + stride]; svar[t] = svar[t] + svar[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = smean[0] / f32(p.h);
    let varr = svar[0] / f32(p.h) - mean * mean;
    let inv = 1.0 / sqrt(varr + p.eps);
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        dst[base + i] = (src[base + i] - mean) * inv * weight[i] + bias[i];
    }
}
`

// geluTanhWGSL: the tanh approximation of GELU (SigLIP's hidden_act), elementwise.
// 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))).
const geluTanhWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    let c = 0.7978845608028654; // sqrt(2/pi)
    for (var i: u32 = gid.x; i < p.n; i = i + total) {
        let v = x[i];
        x[i] = 0.5 * v * (1.0 + tanh(c * (v + 0.044715 * v * v * v)));
    }
}
`

// softmaxRowsWGSL: numerically-stable row softmax over `rows`×`n`, applying
// `scale` to each element first (the attention 1/√d). In place. One workgroup
// per row (e.g. one per (head, query) attention row).
const softmaxRowsWGSL = `
struct P { rows: u32, n: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>; // [rows, n]
@group(0) @binding(1) var<uniform>             p: P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.n;
    let t = lid.x;
    // row max (of scaled scores)
    var m: f32 = -3.4e38;
    for (var i: u32 = t; i < p.n; i = i + 64u) { let v = x[base + i] * p.scale; if (v > m) { m = v; } }
    sh[t] = m;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop { if (stride == 0u) { break; } if (t < stride) { sh[t] = max(sh[t], sh[t + stride]); } workgroupBarrier(); stride = stride / 2u; }
    let rmax = sh[0];
    workgroupBarrier();
    // exp + sum
    var s: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { let e = exp(x[base + i] * p.scale - rmax); x[base + i] = e; s = s + e; }
    sh[t] = s;
    workgroupBarrier();
    stride = 32u;
    loop { if (stride == 0u) { break; } if (t < stride) { sh[t] = sh[t] + sh[t + stride]; } workgroupBarrier(); stride = stride / 2u; }
    let inv = 1.0 / sh[0];
    for (var i: u32 = t; i < p.n; i = i + 64u) { x[base + i] = x[base + i] * inv; }
}
`

// ensureVision compiles the vision-only kernels (lazily, once per Context).
func (c *Context) ensureVision() error {
	if c.lnRowsPipeline != nil {
		return nil
	}
	// This closure used to DISCARD its *wgpu.ShaderModule (returning only pipeline+layout), so the
	// five vision shader modules were unreachable for the rest of the process — not merely missing
	// from Close, but impossible to release from anywhere (audit C-26a). Delegating to the tracked
	// constructor registers both objects at creation; the shader is dropped here only after that.
	mk := func(label, code string) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
		_, pl, lay, err := c.mkPipeline(label, code)
		if err != nil {
			return nil, nil, err
		}
		return pl, lay, nil
	}
	var err error
	if c.lnRowsPipeline, c.lnRowsLayout, err = mk("layerNormRows", layerNormRowsWGSL); err != nil {
		return err
	}
	if c.geluPipeline, c.geluLayout, err = mk("geluTanh", geluTanhWGSL); err != nil {
		return err
	}
	if c.softmaxPipeline, c.softmaxLayout, err = mk("softmaxRows", softmaxRowsWGSL); err != nil {
		return err
	}
	if c.addRowsPipeline, c.addRowsLayout, err = mk("addRows", addRowsWGSL); err != nil {
		return err
	}
	if c.copyHeadPipeline, c.copyHeadLayout, err = mk("copyHead", copyHeadWGSL); err != nil {
		return err
	}
	return nil
}

// maxWG caps a 1D workgroup count at the WebGPU per-dimension limit; the
// grid-strided kernels (addRows, gelu) cover the rest by looping.
func maxWG(groups int) int {
	if groups > 65535 {
		return 65535
	}
	return groups
}

// addRowsDevice: dst += add, broadcasting add[cols] over dst when mode=0, else
// elementwise (mode=1). n = len(dst). No Poll.
func (c *Context) addRowsDevice(dst, add *wgpu.Buffer, n, cols, mode int) error {
	return c.twoBufKernel(c.addRowsPipeline, c.addRowsLayout, dst, add, []uint32{uint32(n), uint32(cols), uint32(mode), 0}, maxWG((n+63)/64))
}

// copyHeadDevice: per-head strided copy (dir 0 gather / 1 scatter / 2 gatherT).
func (c *Context) copyHeadDevice(src, dst *wgpu.Buffer, rows, hd, srcStride, off, dir int) error {
	return c.twoBufKernel(c.copyHeadPipeline, c.copyHeadLayout, src, dst, []uint32{uint32(rows), uint32(hd), uint32(srcStride), uint32(off), uint32(dir), 0, 0, 0}, (rows*hd+63)/64)
}

// twoBufKernel runs a kernel with two storage buffers (binding 0,1) + a uniform.
func (c *Context) twoBufKernel(pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, b0, b1 *wgpu.Buffer, p []uint32, dispatch int) error {
	pbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "k-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: b0, Size: b0.GetSize()},
		{Binding: 1, Buffer: b1, Size: b1.GetSize()},
		{Binding: 2, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(dispatch), 1, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// inplaceKernel runs an in-place compute op (one read_write buffer + a uniform),
// no Poll. dispatch = number of workgroups. Shared by gelu (n/64 groups) and
// softmaxRows (one group per row).
func (c *Context) inplaceKernel(pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, x *wgpu.Buffer, p []uint32, dispatch int) error {
	pbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ip-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: x, Size: x.GetSize()},
		{Binding: 1, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(dispatch), 1, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// softmaxRowsHost / geluHost: host wrappers for the parity tests.
func (c *Context) softmaxRowsHost(x []float32, rows, n int, scale float32) ([]float32, error) {
	if err := c.ensureVision(); err != nil {
		return nil, err
	}
	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	defer xd.buf.Release()
	if err := c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, xd.buf, []uint32{uint32(rows), uint32(n), math.Float32bits(scale), 0}, rows); err != nil {
		return nil, err
	}
	return c.Readback(&DeviceBuffer{buf: xd.buf, n: rows * n})
}

func (c *Context) geluHost(x []float32) ([]float32, error) {
	if err := c.ensureVision(); err != nil {
		return nil, err
	}
	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	defer xd.buf.Release()
	if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, xd.buf, []uint32{uint32(len(x)), 0, 0, 0}, (len(x)+63)/64); err != nil {
		return nil, err
	}
	return c.Readback(&DeviceBuffer{buf: xd.buf, n: len(x)})
}

// layerNormRowsDevice runs LayerNorm over `rows`×`h`, src→dst device buffers, no
// Poll (chains into the next op). weight/bias are resident [h] buffers.
func (c *Context) layerNormRowsDevice(src *wgpu.Buffer, weight, bias *wgpu.Buffer, dst *wgpu.Buffer, rows, h int, eps float32) error {
	pbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ln-p", Contents: wgpu.ToBytes([]uint32{uint32(rows), uint32(h), math.Float32bits(eps), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.lnRowsLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: src, Size: src.GetSize()},
		{Binding: 1, Buffer: weight, Size: weight.GetSize()},
		{Binding: 2, Buffer: bias, Size: bias.GetSize()},
		{Binding: 3, Buffer: dst, Size: dst.GetSize()},
		{Binding: 4, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.lnRowsPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(rows), 1, 1) // one workgroup per row
	if err := pass.End(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// matmulF32Device runs dst[M,N] = a[M,K] · b[N,K]ᵀ on the naive f32 matmul
// pipeline (c.pipeline), output to a fresh device buffer, NO Poll. Used for the
// patch-embed and the attention QKᵀ / scores·V (activation×activation, no resident
// weight). a,b are device buffers.
func (c *Context) matmulF32Device(a, b *wgpu.Buffer, M, K, N int) (*DeviceBuffer, error) {
	dst, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mm", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	// dst is handed to the caller only on success; release it on EVERY early error return below
	// (dims-create, bind-group-create, pass.End) — before this it leaked on those paths (audit M-16).
	ok := false
	defer func() {
		if !ok {
			dst.Release()
		}
	}()
	dims, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(K), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer dims.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: a, Size: a.GetSize()},
		{Binding: 1, Buffer: b, Size: b.GetSize()},
		{Binding: 2, Buffer: dst, Size: dst.GetSize()},
		{Binding: 3, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	ok = true // dst now owned by the returned DeviceBuffer
	return &DeviceBuffer{buf: dst, n: M * N}, nil
}

// addRowsWGSL: dst[i] += bias[i % cols] (broadcast a [cols] bias over [rows,cols])
// when mode=0; dst[i] += other[i] (elementwise, posEmb / residual) when mode=1.
const addRowsWGSL = `
struct P { n: u32, cols: u32, mode: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read>       add: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = ng.x * 64u;
    for (var i: u32 = gid.x; i < p.n; i = i + total) {
        if (p.mode == 0u) { dst[i] = dst[i] + add[i % p.cols]; }
        else { dst[i] = dst[i] + add[i]; }
    }
}
`

// copyHeadWGSL: per-head strided copy between [rows, srcStride] and [rows, hd]
// (gather a head's slice / scatter it back / build vᵀ), driven by dir:
//
//	0 gather : dst[i*hd + d]      = src[i*srcStride + off + d]
//	1 scatter: dst[i*srcStride+off+d] = src[i*hd + d]
//	2 gatherT: dst[d*rows + i]    = src[i*srcStride + off + d]   (transpose to [hd,rows])
const copyHeadWGSL = `
struct P { rows: u32, hd: u32, srcStride: u32, off: u32, dir: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = p.rows * p.hd;
    let k = gid.x;
    if (k >= n) { return; }
    let i = k / p.hd;
    let d = k % p.hd;
    if (p.dir == 0u)      { dst[i*p.hd + d]              = src[i*p.srcStride + p.off + d]; }
    else if (p.dir == 1u) { dst[i*p.srcStride + p.off + d] = src[i*p.hd + d]; }
    else                  { dst[d*p.rows + i]            = src[i*p.srcStride + p.off + d]; }
}
`

// LayerNormRowsHost is a host-in/host-out wrapper used by the parity test.
func (c *Context) LayerNormRowsHost(src, weight, bias []float32, rows, h int, eps float32) ([]float32, error) {
	if err := c.ensureVision(); err != nil {
		return nil, err
	}
	sd, err := c.UploadF32(src)
	if err != nil {
		return nil, err
	}
	defer sd.buf.Release()
	wd, err := c.UploadF32(weight)
	if err != nil {
		return nil, err
	}
	defer wd.buf.Release()
	bd, err := c.UploadF32(bias)
	if err != nil {
		return nil, err
	}
	defer bd.buf.Release()
	out, err := c.newF32("ln-out", rows*h)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	if err := c.layerNormRowsDevice(sd.buf, wd.buf, bd.buf, out, rows, h, eps); err != nil {
		return nil, err
	}
	return c.Readback(&DeviceBuffer{buf: out, n: rows * h})
}
