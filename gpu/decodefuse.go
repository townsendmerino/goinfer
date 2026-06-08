//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// §2 fused decode kernels. The §5 finding: decode is glue-serialization-bound —
// the per-token cost is a deep RAW dependency chain of ~535 small dispatches,
// each forcing a barrier the GPU can't hide. The lever is critical-PATH length,
// so every standalone glue op that can be folded into the kernel it borders
// removes a link from the serialized spine. These kernels do that folding; each
// is bit-exact with the unfused pair it replaces (same f32 math, same int8 pack).

// rmsnormQuant fuses RMSNorm → activation-quantize: it reads the hidden vector
// once, computes xn = src·inv·weight(+1) in registers, finds its max-abs by tree
// reduce, and emits the packed int8 activation + scale directly — replacing the
// rms(→xn global)→quant(xn→q,s) pair with ONE dispatch and no xn round-trip.
// One workgroup (64 lanes); xn is recomputed in the quantize pass rather than
// staged, so there is no width-bounded workgroup array (works for any hidden).
const rmsnormQuantWGSL = `
struct P { h: u32, eps: f32, addone: u32, np: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>;  // [h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [h]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // 1) sum of squares → rmsnorm inverse scale
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[i]; ss = ss + v*v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps);
    workgroupBarrier();
    // 2) max-abs of xn = src·inv·weight(+1), parallel tree reduce
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        mx = max(mx, abs(src[i] * inv * w));
    }
    sh[t] = mx;
    workgroupBarrier();
    stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = max(sh[t], sh[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    var s: f32 = sh[0] / 127.0;
    if (s == 0.0) { s = 1.0; }
    if (t == 0u) { scales[0] = s; }
    let invs = 1.0 / s;
    // 3) quantize + pack (recompute xn; one u32 word = 4 int8)
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.h) {
                var w = weight[k];
                if (p.addone == 1u) { w = w + 1.0; }
                q = i32(round(src[k] * inv * w * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

// swigluQuant fuses SwiGLU → activation-quantize: silu(gate)·up is computed,
// max-abs'd, and packed to int8 in ONE dispatch, so the inter-wide (8960, ~36 KB)
// SwiGLU output never materializes in global memory or crosses a barrier — a
// double win under the §5 "per-link drain scales with the bordering kernel's
// data" model. The product is recomputed in the pack pass (cheap arithmetic)
// rather than staged, so no inter-wide workgroup array is needed.
const swigluQuantWGSL = `
struct P { n: u32, np: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;  // [n]
@group(0) @binding(1) var<storage, read>       up:     array<f32>;  // [n]
@group(0) @binding(2) var<storage, read_write> qout:   array<u32>;  // [np/4] packed int8
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;  // [1]
@group(0) @binding(4) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
fn silu_up(g: f32, u: f32) -> f32 { return (g / (1.0 + exp(-g))) * u; }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // 1) max-abs of mid = silu(gate)·up, parallel tree reduce
    var mx: f32 = 0.0;
    for (var i: u32 = t; i < p.n; i = i + 64u) { mx = max(mx, abs(silu_up(gate[i], up[i]))); }
    sh[t] = mx;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = max(sh[t], sh[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    var s: f32 = sh[0] / 127.0;
    if (s == 0.0) { s = 1.0; }
    if (t == 0u) { scales[0] = s; }
    let invs = 1.0 / s;
    // 2) quantize + pack
    let nw = p.np / 4u;
    for (var wi: u32 = t; wi < nw; wi = wi + 64u) {
        var word: u32 = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let k = wi * 4u + j;
            var q: i32 = 0;
            if (k < p.n) {
                q = i32(round(silu_up(gate[k], up[k]) * invs));
                if (q > 127) { q = 127; } else if (q < -127) { q = -127; }
            }
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        qout[wi] = word;
    }
}
`

func (c *Context) ensureFuse() error {
	if c.rmsQuantPipeline != nil {
		return nil
	}
	mk := func(label, code string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
		sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: label, WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code}})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("gpu: compile %s: %w", label, err)
		}
		pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: label, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"}})
		if err != nil {
			sh.Release()
			return nil, nil, nil, fmt.Errorf("gpu: pipeline %s: %w", label, err)
		}
		return sh, pl, pl.GetBindGroupLayout(0), nil
	}
	var err error
	if c.rmsQuantShader, c.rmsQuantPipeline, c.rmsQuantLayout, err = mk("rmsnormQuant", rmsnormQuantWGSL); err != nil {
		return err
	}
	if c.swigluQuantShader, c.swigluQuantPipeline, c.swigluQuantLayout, err = mk("swigluQuant", swigluQuantWGSL); err != nil {
		return err
	}
	return nil
}
