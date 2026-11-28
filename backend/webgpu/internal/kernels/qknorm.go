//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

func (c *Context) ensureQKNorm() error {
	if c.qkNormPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "qkNorm", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.QkNormWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile qkNorm: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "qkNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline qkNorm: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.qkNormShader, c.qkNormPipeline, c.qkNormLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureVNorm() error {
	if c.vNormPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "vNorm", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.VNormWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile vNorm: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "vNorm", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline vNorm: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.vNormShader, c.vNormPipeline, c.vNormLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureScaleVec() error {
	if c.scaleVecPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "scaleVec", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.ScaleVecWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile scaleVec: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "scaleVec", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline scaleVec: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.scaleVecShader, c.scaleVecPipeline, c.scaleVecLayout = sh, pl, c.bgl(pl)
	return nil
}
