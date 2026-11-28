//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Compute-time LoRA on the resident path — the WebGPU
// backend, following metal/lora.go's design (same math, same "reuse the base projection's own
// quantized activation" input, same before-any-subsequent-op ordering) but adapted to this
// backend's architecture: newDecodeRunner builds a FIXED dispatch-step list (r.steps) ONCE, and
// every Run() just walks that Go-side slice fresh each token (record() re-encodes it into a new
// command buffer every call — nothing about it is baked GPU-side). So unlike Metal (which
// re-encodes its whole trunk per token and can just skip a Go-side `if`), a bound/unbound
// adapter here is a matter of REBUILDING r.steps from a saved pristine r.baseSteps plus this
// layer's LoRA dispatches spliced in at recorded hook points — see SetAdapter below.

// loraPipes holds the Context's compute-time LoRA down/up pipelines (ensureLora).
type loraPipes struct {
	loraDownShader   *wgpu.ShaderModule
	loraDownPipeline *wgpu.ComputePipeline
	loraDownLayout   *wgpu.BindGroupLayout
	loraUpShader     *wgpu.ShaderModule
	loraUpPipeline   *wgpu.ComputePipeline
	loraUpLayout     *wgpu.BindGroupLayout
}

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
	dsh, dpl, err := mk("loraDeltaDown", wgsl.LoraDeltaDownWGSL)
	if err != nil {
		return err
	}
	ush, upl, err := mk("loraDeltaUp", wgsl.LoraDeltaUpWGSL)
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
