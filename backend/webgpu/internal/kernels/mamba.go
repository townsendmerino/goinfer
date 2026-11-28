//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// mambaPipes holds the Context's resident Mamba-2 SSM, causal-conv, and gated-norm pipelines.
type mambaPipes struct {
	mambaSSMShader     *wgpu.ShaderModule
	mambaSSMPipeline   *wgpu.ComputePipeline
	mambaSSMLayout     *wgpu.BindGroupLayout
	mambaConvShader    *wgpu.ShaderModule
	mambaConvPipeline  *wgpu.ComputePipeline
	mambaConvLayout    *wgpu.BindGroupLayout
	mambaGNormShader   *wgpu.ShaderModule
	mambaGNormPipeline *wgpu.ComputePipeline
	mambaGNormLayout   *wgpu.BindGroupLayout
}

func (c *Context) ensureMambaConv() error {
	if c.mambaConvPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaConv", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MambaConvShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaConv: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaConv", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaConv: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mambaConvShader, c.mambaConvPipeline, c.mambaConvLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMambaGNorm() error {
	if c.mambaGNormPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaGNorm", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MambaGNormShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaGNorm: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaGNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaGNorm: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mambaGNormShader, c.mambaGNormPipeline, c.mambaGNormLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMambaSSM() error {
	if c.mambaSSMPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaSSM", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MambaSSMShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaSSM: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaSSM", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaSSM: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.mambaSSMShader, c.mambaSSMPipeline, c.mambaSSMLayout = sh, pl, c.bgl(pl)
	return nil
}
