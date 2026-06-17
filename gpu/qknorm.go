//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Per-head QK-norm (Lever C — extend resident DecodeRunner eligibility). Qwen3 / GLM /
// Mellum RMS-normalize each query and key HEAD over headDim (with its own learned
// weight[headDim]) BEFORE RoPE — the one feature that otherwise made these archs
// residency-ineligible. This kernel does that in place: one workgroup per head, 64 lanes
// tree-reduce the sum-of-squares over headDim, then rescale. Same f32 RMS math as the
// CPU rmsNorm(qi, lw.QNorm, nH, hd, …) (decoder/attention.go), so the resident forward
// stays cosine ~1.0 vs CPU (TestResidentQKNorm_parity). addOne carries the Gemma-style
// (1+w) offset (false for Qwen3).
const qkNormWGSL = `
struct P { heads: u32, hd: u32, eps: f32, addone: u32 };
@group(0) @binding(0) var<storage, read_write> vec:    array<f32>;  // [heads*hd], normalized in place
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [hd]
@group(0) @binding(2) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    if (h >= p.heads) { return; }
    let t = lid.x;
    let base = h * p.hd;
    var ss: f32 = 0.0;
    for (var i: u32 = t; i < p.hd; i = i + 64u) { let v = vec[base + i]; ss = ss + v * v; }
    sh[t] = ss;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.hd) + p.eps);
    for (var i: u32 = t; i < p.hd; i = i + 64u) {
        var w = weight[i];
        if (p.addone == 1u) { w = w + 1.0; }
        vec[base + i] = vec[base + i] * inv * w;
    }
}
`

func (c *Context) ensureQKNorm() error {
	if c.qkNormPipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "qkNorm", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: qkNormWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile qkNorm: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "qkNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline qkNorm: %w", err)
	}
	c.qkNormShader, c.qkNormPipeline, c.qkNormLayout = sh, pl, pl.GetBindGroupLayout(0)
	return nil
}
