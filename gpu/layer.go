//go:build gpu

package gpu

import (
	"math"

	"github.com/cogentcore/webgpu/wgpu"
)

// Stage 3 — whole-sub-block residency. The decode floor is syncs per token
// (Stage 2 finding); the fused-batch cut qkv/gate-up to ~4 syncs/layer. This goes
// further for the MLP: keep the activation in device buffers through
// RMSNorm → gate/up matmul → SwiGLU → down matmul → residual, chaining each step
// as a Submit with NO Poll (queue order guarantees the dependency), so the whole
// MLP block is ONE sync (one upload, one readback) instead of two synced matmuls
// plus CPU norm/SwiGLU/residual. Attention + KV-on-device (the harder half, with
// the resident growing KV cache) is the remaining Stage-3 work.

const rmsnormShaderWGSL = `
struct P { h: u32, eps: f32, addone: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;
@group(0) @binding(3) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var s: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; s = s + v*v; }
    sh[t] = s;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps);
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        dst[i] = src[i] * inv * w;
    }
}
`

const swigluShaderWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform>             p:    P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = gate[i];
    dst[i] = (g / (1.0 + exp(-g))) * up[i];   // silu(gate)·up
}
`

const residualShaderWGSL = `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;  // x += y
@group(0) @binding(1) var<storage, read>       y: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] + y[i];
}
`

func (c *Context) ensureLayer() error {
	// Guard EACH pipeline independently, not just the first: a mid-build failure (transient OOM)
	// used to leave rmsnormPipeline set but swiglu/residual nil, and the next call saw the first-field
	// guard satisfied and dispatched a nil pipeline (audit R-30). Per-field guards retry only what's
	// missing — the ensureGEMVW8A16 shape. Shared tracked constructor (gpu.go) registers for release (C-26).
	mk := c.mkPipeline
	var err error
	if c.rmsnormPipeline == nil {
		if c.rmsnormShader, c.rmsnormPipeline, c.rmsnormLayout, err = mk("rmsnorm", rmsnormShaderWGSL); err != nil {
			return err
		}
	}
	if c.swigluPipeline == nil {
		if c.swigluShader, c.swigluPipeline, c.swigluLayout, err = mk("swiglu", swigluShaderWGSL); err != nil {
			return err
		}
	}
	if c.residualPipeline == nil {
		if c.residualShader, c.residualPipeline, c.residualLayout, err = mk("residual", residualShaderWGSL); err != nil {
			return err
		}
	}
	return nil
}

// submitUnary encodes a 1-output elementwise/reduction op (no Poll) and returns
// the output device buffer. binds = the bind-group entries (output already
// included by the caller); n is the dispatch element count (ceil/64 workgroups).
func (c *Context) submitUnary(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, n int) error {
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(n)+63)/64, 1, 1)
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

func (c *Context) newF32(label string, n int) (*wgpu.Buffer, error) {
	return c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
}

func (c *Context) dims4(label string, a uint32, bf float32) (*wgpu.Buffer, error) {
	// pack {a, bf-as-bits-or-0, 0, 0}; for rmsnorm bf is eps, else unused.
	return c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: label, Contents: wgpu.ToBytes([]uint32{a, math.Float32bits(bf), 0, 0}), Usage: wgpu.BufferUsageUniform})
}

// FusedMLP runs RMSNorm → gate/up → SwiGLU → down → residual entirely on-device,
// syncing once. x is [H]; rmsW is the resident norm weight; gate/up/down are
// resident W8A8 weights. Returns x + MLP(rmsnorm(x)) ([H]).
func (c *Context) FusedMLP(x []float32, rmsW *DeviceBuffer, gate, up, down *ResidentW8A8, eps float32, addOne bool) ([]float32, error) {
	if err := c.ensureLayer(); err != nil {
		return nil, err
	}
	H, I := len(x), gate.rows
	var keep []interface{ Release() }
	rel := func() {
		for _, k := range keep {
			k.Release()
		}
	}
	defer rel()

	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	keep = append(keep, xd)

	// 1. RMSNorm(x) → xn
	xn, err := c.newF32("xn", H)
	if err != nil {
		return nil, err
	}
	// V-22 (docs/review-2026-09-04.md): wrapped ONCE, into xnDB, and reused at the quantize step
	// below — a second newDeviceBuffer(xn, H) there double-accounted the same buffer's bytes in
	// liveBufferBytes, since only the wrapper kept here ever gets Close()d by rel().
	xnDB := newDeviceBuffer(xn, H)
	keep = append(keep, xnDB)
	pbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rms-p", Contents: wgpu.ToBytes([]uint32{uint32(H), math.Float32bits(eps), boolU32(addOne), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	keep = append(keep, pbuf)
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.rmsnormLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: xd.buf, Size: xd.buf.GetSize()},
		{Binding: 1, Buffer: rmsW.buf, Size: rmsW.buf.GetSize()},
		{Binding: 2, Buffer: xn, Size: xn.GetSize()},
		{Binding: 3, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	keep = append(keep, bg)
	if err := c.submitUnary(c.rmsnormPipeline, bg, 64); err != nil { // 1 workgroup (reduction)
		return nil, err
	}

	// 2. quantize(xn) → aq/aScale, gate/up matmuls (device)
	qb, sb, err := c.quantizeDevice(xnDB, 1, H)
	if err != nil {
		return nil, err
	}
	keep = append(keep, qb, sb)
	gd, err := c.matmulW8A8Device(qb, sb, gate, 1)
	if err != nil {
		return nil, err
	}
	keep = append(keep, gd)
	ud, err := c.matmulW8A8Device(qb, sb, up, 1)
	if err != nil {
		return nil, err
	}
	keep = append(keep, ud)

	// 3. SwiGLU(gate, up) → mid
	mid, err := c.newF32("mid", I)
	if err != nil {
		return nil, err
	}
	// V-22 (docs/review-2026-09-04.md): same reuse-not-rewrap fix as xnDB above.
	midDB := newDeviceBuffer(mid, I)
	keep = append(keep, midDB)
	sp, err := c.dims4("swiglu-p", uint32(I), 0)
	if err != nil {
		return nil, err
	}
	keep = append(keep, sp)
	sbg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.swigluLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: gd.buf, Size: gd.buf.GetSize()},
		{Binding: 1, Buffer: ud.buf, Size: ud.buf.GetSize()},
		{Binding: 2, Buffer: mid, Size: mid.GetSize()},
		{Binding: 3, Buffer: sp, Size: sp.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	keep = append(keep, sbg)
	if err := c.submitUnary(c.swigluPipeline, sbg, I); err != nil {
		return nil, err
	}

	// 4. quantize(mid) → down matmul
	qb2, sb2, err := c.quantizeDevice(midDB, 1, I)
	if err != nil {
		return nil, err
	}
	keep = append(keep, qb2, sb2)
	dd, err := c.matmulW8A8Device(qb2, sb2, down, 1)
	if err != nil {
		return nil, err
	}
	keep = append(keep, dd)

	// 5. residual x += down
	rp, err := c.dims4("res-p", uint32(H), 0)
	if err != nil {
		return nil, err
	}
	keep = append(keep, rp)
	rbg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.residualLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: xd.buf, Size: xd.buf.GetSize()},
		{Binding: 1, Buffer: dd.buf, Size: dd.buf.GetSize()},
		{Binding: 2, Buffer: rp, Size: rp.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	keep = append(keep, rbg)
	if err := c.submitUnary(c.residualPipeline, rbg, H); err != nil {
		return nil, err
	}

	return c.Readback(xd) // the single sync
}

func boolU32(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}
