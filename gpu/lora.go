//go:build gpu

package gpu

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Compute-time LoRA on the resident path (G3, docs/task-gpu-paths-2026-09.md) — the WebGPU
// backend, following metal/lora.go's design (same math, same "reuse the base projection's own
// quantized activation" input, same before-any-subsequent-op ordering) but adapted to this
// backend's architecture: newDecodeRunner builds a FIXED dispatch-step list (r.steps) ONCE, and
// every Run() just walks that Go-side slice fresh each token (record() re-encodes it into a new
// command buffer every call — nothing about it is baked GPU-side). So unlike Metal (which
// re-encodes its whole trunk per token and can just skip a Go-side `if`), a bound/unbound
// adapter here is a matter of REBUILDING r.steps from a saved pristine r.baseSteps plus this
// layer's LoRA dispatches spliced in at recorded hook points — see SetAdapter below.

const loraDeltaDownWGSL = `
struct PDown { k: u32, r: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       aq: array<vec4<u32>>;   // [kp/16] int8 act (kp >= k)
@group(0) @binding(1) var<storage, read>       ascale: array<f32>;     // [1]
@group(0) @binding(2) var<storage, read>       amat: array<f32>;       // [r, k] row-major
@group(0) @binding(3) var<storage, read_write> tout: array<f32>;       // [r]
@group(0) @binding(4) var<uniform>             p: PDown;

fn get_aq_i8(k: u32) -> i32 {
    let v = aq[k >> 4u];
    var word: u32;
    let wi = (k >> 2u) & 3u;
    if (wi == 0u) { word = v.x; } else if (wi == 1u) { word = v.y; } else if (wi == 2u) { word = v.z; } else { word = v.w; }
    let shifted = word << (24u - (k & 3u) * 8u);
    return i32(shifted) >> 24u;
}

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let sc = ascale[0];
    for (var r: u32 = 0u; r < p.r; r = r + 1u) {
        let base = r * p.k;
        var part: f32 = 0.0;
        for (var k: u32 = t; k < p.k; k = k + 64u) {
            part = part + amat[base + k] * (f32(get_aq_i8(k)) * sc);
        }
        red[t] = part;
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { red[t] = red[t] + red[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) { tout[r] = red[0]; }
        workgroupBarrier();
    }
}
`

const loraDeltaUpWGSL = `
struct PUp { r: u32, outn: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       bmat: array<f32>;  // [outn, r] row-major
@group(0) @binding(1) var<storage, read>       tin:  array<f32>;  // [r]
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;  // [outn], accumulated into
@group(0) @binding(3) var<uniform>             p:    PUp;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.outn) { return; }
    let base = row * p.r;
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < p.r; r = r + 1u) {
        acc = acc + bmat[base + r] * tin[r];
    }
    dst[row] = dst[row] + p.scale * acc;
}
`

func (c *Context) ensureLora() error {
	if c.loraDownPipeline != nil {
		return nil
	}
	mk := func(label, src string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, error) {
		sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
			Label:      label,
			WGSLSource: &wgpu.ShaderSourceWGSL{Code: src},
		})
		if err != nil {
			return nil, nil, fmt.Errorf("gpu: compile %s shader: %w", label, err)
		}
		pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
			Label:   label,
			Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
		})
		if err != nil {
			sh.Release()
			return nil, nil, fmt.Errorf("gpu: create %s pipeline: %w", label, err)
		}
		return sh, pl, nil
	}
	dsh, dpl, err := mk("loraDeltaDown", loraDeltaDownWGSL)
	if err != nil {
		return err
	}
	ush, upl, err := mk("loraDeltaUp", loraDeltaUpWGSL)
	if err != nil {
		dsh.Release()
		dpl.Release()
		return err
	}
	c.track(dsh.Release, dpl.Release, ush.Release, upl.Release)
	c.loraDownShader, c.loraDownPipeline = dsh, dpl
	c.loraDownLayout = c.bgl(dpl)
	c.loraUpShader, c.loraUpPipeline = ush, upl
	c.loraUpLayout = c.bgl(upl)
	return nil
}
