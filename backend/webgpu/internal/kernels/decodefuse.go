//go:build gpu

package kernels

import (
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
)

// §2 fused decode kernels. Decode is glue-serialization-bound: the per-token cost is a deep RAW dependency chain of small
// dispatches, each forcing a barrier the GPU can't hide, so the lever is critical-PATH length. Every standalone glue op that
// can be folded into the kernel it borders removes a link from the serialized spine; these kernels do that folding, and each
// is bit-exact with the unfused pair it replaces (same f32 math, same int8 pack).
// History: docs/code-notes/gpu.md#§2 fused decode kernels.

func (c *Context) ensureFuse() error {
	// Per-pipeline guards: a mid-build failure must not leave the first-field guard satisfied with a later pipeline nil.
	// Shared tracked constructor (gpu.go) registers for release.
	mk := func(label, src string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
		code, err := c.quantSrc(label, src)
		if err != nil {
			return nil, nil, nil, err
		}
		return c.mkPipeline(label, code)
	}
	var err error
	if c.rmsQuantPipeline == nil {
		if c.rmsQuantShader, c.rmsQuantPipeline, c.rmsQuantLayout, err = mk("rmsnormQuant", wgsl.RmsnormQuantWGSL); err != nil {
			return err
		}
	}
	if c.swigluQuantPipeline == nil {
		if c.swigluQuantShader, c.swigluQuantPipeline, c.swigluQuantLayout, err = mk("swigluQuant", wgsl.SwigluQuantWGSL); err != nil {
			return err
		}
	}
	if c.gegluQuantPipeline == nil {
		if c.gegluQuantShader, c.gegluQuantPipeline, c.gegluQuantLayout, err = mk("gegluQuant", wgsl.GegluQuantWGSL); err != nil {
			return err
		}
	}
	return nil
}
