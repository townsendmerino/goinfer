//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// W8A16 GEMV: int8 WEIGHT × f32 ACTIVATION (no activation quantization). Control/fix for the
// granite-resident gap (docs/ssm-int8-quality.md): Phase A pinned the 93.6%→66% loss on int8
// ACTIVATION re-quantization compounding across the 40-layer stack (the mamba kernels and
// in_proj are bit-correct). Keeping the activation f32 stops that cascade. In a decode GEMV
// (M=1) the activation is [K] vs the weight's [N,K], so reading it f32 is negligible — W8A16
// should stay ~W8A8 speed. The weight is still the packed int8 (w.wbuf/w.sbuf); only the
// activation binding changes from packed int8 to a plain f32 array (kp-padded, zero tail).

const gemvW8A16ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, addResidual: u32 };
@group(0) @binding(0) var<storage, read>       act:     array<f32>;        // [kp] f32 activation (zero-padded)
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16] int8 weight
@group(0) @binding(2) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(3) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(4) var<uniform>             dims:    Dims;
fn unpack_i8x4w(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotwf(wpacked: u32, base: u32) -> f32 {
    let wv = unpack_i8x4w(wpacked);
    return f32(wv.x) * act[base] + f32(wv.y) * act[base + 1u] + f32(wv.z) * act[base + 2u] + f32(wv.w) * act[base + 3u];
}
var<workgroup> partf: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kv = dims.kp / 16u;
    let bBase = n * kv;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < kv; v = v + 64u) {
        let b4 = bq[bBase + v];
        let base = v * 16u;
        acc = acc + dotwf(b4.x, base) + dotwf(b4.y, base + 4u) + dotwf(b4.z, base + 8u) + dotwf(b4.w, base + 12u);
    }
    partf[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partf[t] = partf[t] + partf[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partf[0] * bScales[n];
        if (dims.addResidual == 1u) { dst[n] = dst[n] + r; } else { dst[n] = r; }
    }
}
`

// moeExpertW8A16: the indexed stacked-expert GEMV (moe.go's moeExpertGEMVWGSL) but with f32
// activation. mode 1 folds the router weight (dst[n] += wgt[slot]·r) for the down combine.
const moeExpertW8A16ShaderWGSL = `
struct Dims { kp: u32, n: u32, slot: u32, mode: u32 };
@group(0) @binding(0) var<storage, read>       act:     array<f32>;        // [kp] f32 activation
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/16] stacked int8
@group(0) @binding(2) var<storage, read>       bScales: array<f32>;        // [nE*N]
@group(0) @binding(3) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(4) var<storage, read>       idx:     array<u32>;        // [k] chosen experts
@group(0) @binding(5) var<storage, read>       wgt:     array<f32>;        // [k] chosen weights
@group(0) @binding(6) var<uniform>             dims:    Dims;
fn unpack_i8x4e2(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotef(wpacked: u32, base: u32) -> f32 {
    let wv = unpack_i8x4e2(wpacked);
    return f32(wv.x) * act[base] + f32(wv.y) * act[base + 1u] + f32(wv.z) * act[base + 2u] + f32(wv.w) * act[base + 3u];
}
var<workgroup> partef: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let kv = dims.kp / 16u;
    let row = e * dims.n + n;
    let bBase = row * kv;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < kv; v = v + 64u) {
        let b4 = bq[bBase + v];
        let base = v * 16u;
        acc = acc + dotef(b4.x, base) + dotef(b4.y, base + 4u) + dotef(b4.z, base + 8u) + dotef(b4.w, base + 12u);
    }
    partef[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partef[t] = partef[t] + partef[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partef[0] * bScales[row];
        if (dims.mode == 1u) { dst[n] = dst[n] + wgt[dims.slot] * r; } else { dst[n] = r; }
    }
}
`

func (c *Context) ensureGEMVW8A16() error {
	if c.gemvW8A16Pipeline == nil {
		sh, pl, lay, err := c.buildCompute("gemvW8A16", gemvW8A16ShaderWGSL)
		if err != nil {
			return err
		}
		c.gemvW8A16Shader, c.gemvW8A16Pipeline, c.gemvW8A16Layout = sh, pl, lay
	}
	if c.moeExpertW8A16Pipeline == nil {
		sh, pl, lay, err := c.buildCompute("moeExpertW8A16", moeExpertW8A16ShaderWGSL)
		if err != nil {
			return err
		}
		c.moeExpertW8A16Shader, c.moeExpertW8A16Pipeline, c.moeExpertW8A16Layout = sh, pl, lay
	}
	return nil
}

// buildCompute compiles a WGSL compute shader → (module, pipeline, group-0 layout).
func (c *Context) buildCompute(label, code string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: label, WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("gpu: compile %s: %w", label, err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: label, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return nil, nil, nil, fmt.Errorf("gpu: pipeline %s: %w", label, err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	return sh, pl, c.bgl(pl), nil
}
