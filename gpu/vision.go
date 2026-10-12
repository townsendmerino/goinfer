//go:build gpu

package gpu

import (
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Resident SigLIP vision encoder, the path to a GPU-fast image prefill (docs/completed/task-gpu-vision-tower.md). Per-call
// matmul offload is a dead end: the submit+readback overhead per matmul is about the whole runtime. The fix is residency:
// keep the [np, hidden] activation in device buffers through all layers, chaining each op as a Submit with NO Poll (queue
// order guarantees the dependency), so the forward syncs once. This file builds the batched (M = np patches) kernels the
// encoder needs that the decode path lacks: standard LayerNorm (mean/var, vs RMSNorm), gelu-tanh (vs silu-gated), and a
// bidirectional softmax, composed with the existing device matmul / quantize / residual primitives.
// History: docs/code-notes/gpu.md#Resident SigLIP vision encoder.

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
// 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))). The tanh argument is clamped to ±15, where tanh is already ±1 in float32: a
// device tanh may return NaN for a large argument, and a float32 tower's activations reach one at real size.
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
        x[i] = 0.5 * v * (1.0 + tanh(clamp(c * (v + 0.044715 * v * v * v), -15.0, 15.0)));
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
	// Guard on the LAST pipeline built (copyHead), not the first (lnRows): a mid-build failure would leave the first-field guard
	// satisfied with later pipelines nil, and the next call would dispatch a nil pipeline. On a partial-build retry the earlier
	// fields are rebuilt (old ones stay tracked for release at Close).
	if c.copyHeadPipeline != nil {
		return nil
	}
	// Delegates to the tracked constructor so both the shader module and the pipeline are registered for release at creation; a
	// closure that discarded the shader module would make it impossible to release from anywhere.
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
	if c.mmTiledF32Pipeline, c.mmTiledF32Layout, err = mk("matmulTiledF32", matmulTiledF32WGSL); err != nil {
		return err
	}
	if c.mmRB64F32Pipeline, c.mmRB64F32Layout, err = mk("matmulRB64F32", matmulRB64F32WGSL); err != nil {
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
	pbuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "k-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: b0, Size: b0.GetSize()},
		{Binding: 1, Buffer: b1, Size: b1.GetSize()},
		{Binding: 2, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(dispatch), 1, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// inplaceKernel runs an in-place compute op (one read_write buffer + a uniform),
// no Poll. dispatch = number of workgroups. Shared by gelu (n/64 groups) and
// softmaxRows (one group per row).
func (c *Context) inplaceKernel(pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, x *wgpu.Buffer, p []uint32, dispatch int) error {
	pbuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ip-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: x, Size: x.GetSize()},
		{Binding: 1, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(dispatch), 1, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
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
	defer xd.Close() // V-22 (docs/review-2026-09-04.md): was xd.buf.Release(), bypassing accounting
	if err := c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, xd.buf, []uint32{uint32(rows), uint32(n), math.Float32bits(scale), 0}, rows); err != nil {
		return nil, err
	}
	return c.Readback(xd)
}

func (c *Context) geluHost(x []float32) ([]float32, error) {
	if err := c.ensureVision(); err != nil {
		return nil, err
	}
	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	defer xd.Close() // V-22 (docs/review-2026-09-04.md): was xd.buf.Release(), bypassing accounting
	if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, xd.buf, []uint32{uint32(len(x)), 0, 0, 0}, (len(x)+63)/64); err != nil {
		return nil, err
	}
	return c.Readback(xd)
}

// layerNormRowsDevice runs LayerNorm over `rows`×`h`, src→dst device buffers, no
// Poll (chains into the next op). weight/bias are resident [h] buffers.
func (c *Context) layerNormRowsDevice(src *wgpu.Buffer, weight, bias *wgpu.Buffer, dst *wgpu.Buffer, rows, h int, eps float32) error {
	pbuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ln-p", Contents: wgpu.ToBytes([]uint32{uint32(rows), uint32(h), math.Float32bits(eps), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer pbuf.Release()
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.lnRowsLayout, Entries: []wgpu.BindGroupEntry{
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
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.lnRowsPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(rows), 1, 1) // one workgroup per row
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// matmulTiledF32WGSL computes dst[m,n] = Σ_k a[m,k]·b[n,k], the naive kernel's contract and bindings, with both
// operands staged through workgroup memory: one workgroup computes a 16×16 block of dst and walks K in strips of 16, so
// each element it loads is used by 16 outputs where the naive kernel streams a whole row of each operand per output.
// A strip's tail past K, and rows past M or N, are zero-filled. The sum runs over k in order, as the naive kernel's does.
const matmulTiledF32WGSL = `
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       a:    array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read>       b:    array<f32>;  // [N, K]
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;  // [M, N]
@group(0) @binding(3) var<uniform>             dims: Dims;
const TS: u32 = 16u;
var<workgroup> As: array<f32, 256>;  // [16 rows][16 k]
var<workgroup> Bs: array<f32, 256>;  // [16 cols][16 k]
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * TS + lid.y;
    let col = wid.x * TS + lid.x;
    let brow = wid.x * TS + lid.y;
    let nStrips = (dims.k + TS - 1u) / TS;
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let kc = s * TS + lid.x;
        var av: f32 = 0.0;
        if (row < dims.m && kc < dims.k) { av = a[row * dims.k + kc]; }
        As[lid.y * TS + lid.x] = av;
        var bv: f32 = 0.0;
        if (brow < dims.n && kc < dims.k) { bv = b[brow * dims.k + kc]; }
        Bs[lid.y * TS + lid.x] = bv;
        workgroupBarrier();
        for (var w: u32 = 0u; w < TS; w = w + 1u) {
            acc = acc + As[lid.y * TS + w] * Bs[lid.x * TS + w];
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        dst[row * dims.n + col] = acc;
    }
}
`

// matmulRB64F32WGSL is the same product with each thread owning a 4×4 block of outputs: a workgroup computes a 64×64
// block of dst, stages 16 K of 64 rows of each operand at a time, and every staged element feeds four multiply-adds
// from registers where the 16×16 kernel's feeds one. The outputs a thread owns are strided (rows lid.y + {0,16,32,48},
// columns lid.x + {0,16,32,48}) and the staged tiles are K-major with a padded row stride, as in the int8 kernel this
// follows (gemm_rb.go). The sum runs over k in order.
const matmulRB64F32WGSL = `
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       a:    array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read>       b:    array<f32>;  // [N, K]
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;  // [M, N]
@group(0) @binding(3) var<uniform>             dims: Dims;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const LD: u32 = 65u;
var<workgroup> As: array<f32, 1040>;  // [BK][LD]: As[w*LD + r] = a(rowBase + r, strip*BK + w)
var<workgroup> Bs: array<f32, 1040>;  // [BK][LD]: Bs[w*LD + c] = b(colBase + c, strip*BK + w)

fn store4(acc: vec4<f32>, row: u32, col0: u32) {
    if (row >= dims.m) { return; }
    let base = row * dims.n;
    if (col0       < dims.n) { dst[base + col0]       = acc.x; }
    if (col0 + 16u < dims.n) { dst[base + col0 + 16u] = acc.y; }
    if (col0 + 32u < dims.n) { dst[base + col0 + 32u] = acc.z; }
    if (col0 + 48u < dims.n) { dst[base + col0 + 48u] = acc.w; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rowBase = wid.y * BM;
    let colBase = wid.x * BN;
    let t = lid.y * 16u + lid.x;
    let nStrips = (dims.k + BK - 1u) / BK;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let w0 = s * BK;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = t + q * 256u;
            let r = idx >> 4u;
            let w = idx & 15u;
            let kc = w0 + w;
            var av: f32 = 0.0;
            let ar = rowBase + r;
            if (ar < dims.m && kc < dims.k) { av = a[ar * dims.k + kc]; }
            As[w * LD + r] = av;
            var bv: f32 = 0.0;
            let br = colBase + r;
            if (br < dims.n && kc < dims.k) { bv = b[br * dims.k + kc]; }
            Bs[w * LD + r] = bv;
        }
        workgroupBarrier();
        for (var w: u32 = 0u; w < BK; w = w + 1u) {
            let ab = w * LD + lid.y;
            let bb = w * LD + lid.x;
            let bvec = vec4<f32>(Bs[bb], Bs[bb + 16u], Bs[bb + 32u], Bs[bb + 48u]);
            acc0 = acc0 + As[ab] * bvec;
            acc1 = acc1 + As[ab + 16u] * bvec;
            acc2 = acc2 + As[ab + 32u] * bvec;
            acc3 = acc3 + As[ab + 48u] * bvec;
        }
        workgroupBarrier();
    }
    let col0 = colBase + lid.x;
    store4(acc0, rowBase + lid.y,       col0);
    store4(acc1, rowBase + lid.y + 16u, col0);
    store4(acc2, rowBase + lid.y + 32u, col0);
    store4(acc3, rowBase + lid.y + 48u, col0);
}
`

// The float32 matmul kernels matmulF32Device can run. visionMatmulForTest picks one for the kernels' gates and timing
// comparisons; outside them it is the default.
const (
	visionMatmulRB64    = iota // matmulRB64F32WGSL, the default
	visionMatmulTiled16        // matmulTiledF32WGSL
	visionMatmulNaive          // the naive kernel
)

var visionMatmulForTest = visionMatmulRB64

// matmulF32Device runs dst[M,N] = a[M,K] · b[N,K]ᵀ on a float32 kernel (the register-blocked one unless a test says
// otherwise), output to a fresh device buffer, NO Poll. Used for the tower's patch embed, its float32 projections, and
// the attention QKᵀ / scores·V. a,b are device buffers.
func (c *Context) matmulF32Device(a, b *wgpu.Buffer, M, K, N int) (*DeviceBuffer, error) {
	dst, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Label: "mm", Size: uint64(M * N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	// dst is handed to the caller only on success; release it on EVERY early error return below (dims-create, bind-group-create,
	// pass.End).
	ok := false
	defer func() {
		if !ok {
			dst.Release()
		}
	}()
	dims, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "dims", Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(K), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer dims.Release()
	// The naive kernel indexes rows by its x dimension and the tiled ones by y.
	pl, ly, gx, gy := c.mmRB64F32Pipeline, c.mmRB64F32Layout, (uint32(N)+63)/64, (uint32(M)+63)/64
	if visionMatmulForTest == visionMatmulTiled16 {
		pl, ly, gx, gy = c.mmTiledF32Pipeline, c.mmTiledF32Layout, (uint32(N)+15)/16, (uint32(M)+15)/16
	}
	if visionMatmulForTest == visionMatmulNaive || pl == nil {
		pl, ly, gx, gy = c.pipeline, c.layout, (uint32(M)+15)/16, (uint32(N)+15)/16
	}
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ly, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: a, Size: a.GetSize()},
		{Binding: 1, Buffer: b, Size: b.GetSize()},
		{Binding: 2, Buffer: dst, Size: dst.GetSize()},
		{Binding: 3, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	ok = true // dst now owned by the returned DeviceBuffer
	return newDeviceBuffer(dst, M*N), nil
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
	defer sd.Close() // V-22 (docs/review-2026-09-04.md): was sd.buf.Release(), bypassing accounting
	wd, err := c.UploadF32(weight)
	if err != nil {
		return nil, err
	}
	defer wd.Close() // same as sd
	bd, err := c.UploadF32(bias)
	if err != nil {
		return nil, err
	}
	defer bd.Close() // same as sd
	out, err := c.newF32("ln-out", rows*h)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	if err := c.layerNormRowsDevice(sd.buf, wd.buf, bd.buf, out, rows, h, eps); err != nil {
		return nil, err
	}
	return c.readbackRaw(out, rows*h)
}
