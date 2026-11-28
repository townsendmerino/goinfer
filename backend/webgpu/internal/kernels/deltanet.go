//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// deltaPipes holds the Context's Gated-DeltaNet recurrence, q/k norm, gates, gated norm, and qGate pipelines.
type deltaPipes struct {
	deltaRuleShader       *wgpu.ShaderModule
	deltaRulePipeline     *wgpu.ComputePipeline
	deltaRuleLayout       *wgpu.BindGroupLayout
	deltaNormShader       *wgpu.ShaderModule
	deltaNormPipeline     *wgpu.ComputePipeline
	deltaNormLayout       *wgpu.BindGroupLayout
	deltaGatesShader      *wgpu.ShaderModule
	deltaGatesPipeline    *wgpu.ComputePipeline
	deltaGatesLayout      *wgpu.BindGroupLayout
	deltaGNormShader      *wgpu.ShaderModule
	deltaGNormPipeline    *wgpu.ComputePipeline
	deltaGNormLayout      *wgpu.BindGroupLayout
	deltaQSplitShader     *wgpu.ShaderModule
	deltaQSplitPipeline   *wgpu.ComputePipeline
	deltaQSplitLayout     *wgpu.BindGroupLayout
	deltaAttnGateShader   *wgpu.ShaderModule
	deltaAttnGatePipeline *wgpu.ComputePipeline
	deltaAttnGateLayout   *wgpu.BindGroupLayout
}

func (c *Context) ensureDeltaRule() error {
	if c.deltaRulePipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaRule", wgsl.DeltaRuleShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaRuleShader, c.deltaRulePipeline, c.deltaRuleLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaNorm() error {
	if c.deltaNormPipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaNorm", wgsl.DeltaNormShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaNormShader, c.deltaNormPipeline, c.deltaNormLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaGates() error {
	if c.deltaGatesPipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaGates", wgsl.DeltaGatesShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaGatesShader, c.deltaGatesPipeline, c.deltaGatesLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaGNorm() error {
	if c.deltaGNormPipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaGNorm", wgsl.DeltaGNormShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaGNormShader, c.deltaGNormPipeline, c.deltaGNormLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaQSplit() error {
	if c.deltaQSplitPipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaQSplit", wgsl.DeltaQSplitShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaQSplitShader, c.deltaQSplitPipeline, c.deltaQSplitLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureDeltaAttnGate() error {
	if c.deltaAttnGatePipeline != nil {
		return nil
	}
	sh, pl, err := c.compute("deltaAttnGate", wgsl.DeltaAttnGateShaderWGSL)
	if err != nil {
		return err
	}
	c.deltaAttnGateShader, c.deltaAttnGatePipeline, c.deltaAttnGateLayout = sh, pl, c.bgl(pl)
	return nil
}

// compute is the ensure* boilerplate these four kernels share: compile, pipeline, register the releases at creation, and wrap
// both errors with the kernel's name.
func (c *Context) compute(name, wgsl string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, error) {
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: name, WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("gpu: compile %s: %w", name, err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: name, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return nil, nil, fmt.Errorf("gpu: pipeline %s: %w", name, err)
	}
	c.track(sh.Release, pl.Release)
	return sh, pl, nil
}
