//go:build gpu

package kernels

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// gemmTileDefault picks the prefill GEMM tile: 64 (the register-blocked kernel) unless
// GOINFER_WEBGPU_GEMM=tiled16 asks for the 16×16 kernel it replaced.
func gemmTileDefault() int {
	if os.Getenv("GOINFER_WEBGPU_GEMM") == "tiled16" {
		return 16
	}
	return 64
}

// tiledPipes is one compiled tiled-GEMM variant (plain or bias) for one tile size.
type tiledPipes struct {
	sh     *wgpu.ShaderModule
	Pl     *wgpu.ComputePipeline
	Layout *wgpu.BindGroupLayout
}

// tiledSource returns the shader for (tile, bias) at this context's dot4 flavour.
func (c *Context) tiledSource(tile int, bias bool) (code, label string) {
	switch {
	case tile == 64 && bias:
		code, label = wgsl.MatmulRB64W8A8BiasShaderWGSL, "matmulRB64W8A8Bias"
		if c.hasDP4A {
			code, label = wgsl.MatmulRB64W8A8DP4ABiasShaderWGSL, "matmulRB64W8A8DP4ABias"
		}
	case tile == 64:
		code, label = wgsl.MatmulRB64W8A8ShaderWGSL, "matmulRB64W8A8"
		if c.hasDP4A {
			code, label = wgsl.MatmulRB64W8A8DP4AShaderWGSL, "matmulRB64W8A8DP4A"
		}
	case bias:
		code, label = wgsl.MatmulTiledW8A8BiasShaderWGSL, "matmulTiledW8A8Bias"
		if c.hasDP4A {
			code, label = wgsl.MatmulTiledW8A8DP4ABiasShaderWGSL, "matmulTiledW8A8DP4ABias"
		}
	default:
		code, label = wgsl.MatmulTiledW8A8ShaderWGSL, "matmulTiledW8A8"
		if c.hasDP4A {
			code, label = wgsl.MatmulTiledW8A8DP4AShaderWGSL, "matmulTiledW8A8DP4A"
		}
	}
	return code, label
}

// compileTiled builds one variant and registers its releases.
func (c *Context) compileTiled(tile int, bias bool) (tiledPipes, error) {
	code, label := c.tiledSource(tile, bias)
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      label,
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return tiledPipes{}, fmt.Errorf("gpu: compile %s: %w", label, err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   label,
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return tiledPipes{}, fmt.Errorf("gpu: create %s pipeline: %w", label, err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	return tiledPipes{sh: sh, Pl: pl, Layout: c.bgl(pl)}, nil
}

// gemmGrid is the tiled GEMM's workgroup grid for an [M, N] output at the active tile.
func (c *Context) gemmGrid(N, M int) (gx, gy uint32) {
	t := uint32(c.gemmTile)
	return (uint32(N) + t - 1) / t, (uint32(M) + t - 1) / t
}

// SetGEMMTileForTest switches the active prefill GEMM kernel between the 16×16 (the
// pre-R10 kernel) and the 64×64 register-blocked one, for an A/B on one loaded model.
// Both variants stay compiled; only the active pointers and the grid change.
func (c *Context) SetGEMMTileForTest(tile int) error {
	if tile != 16 && tile != 64 {
		return fmt.Errorf("gpu: GEMM tile %d: want 16 or 64", tile)
	}
	c.gemmTile = tile
	if c.tiledPipeline != nil {
		c.tiledPipeline = nil
		if err := c.ensureTiled(); err != nil {
			return err
		}
	}
	if c.tiledBiasPipeline != nil {
		c.tiledBiasPipeline = nil
		if err := c.ensureTiledBias(); err != nil {
			return err
		}
	}
	return nil
}

// GEMMTile reports the active prefill GEMM tile (16 or 64).
func (c *Context) GEMMTile() int { return c.gemmTile }
