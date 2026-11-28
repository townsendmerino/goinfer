//go:build gpu

package kernels

import (
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
)

// ensureLayerNorm compiles LayerNormQuantWGSL once per Context.
func (c *Context) ensureLayerNorm() error {
	if c.layerNormQuant.pl != nil {
		return nil
	}
	_, pl, ly, err := c.mkPipeline("layerNormQuant", wgsl.LayerNormQuantWGSL)
	if err != nil {
		return err
	}
	c.layerNormQuant = layerNormPipe{pl, ly}
	return nil
}

// layerNormPipe is the compiled LayerNormQuantWGSL and its bind-group layout.
type layerNormPipe struct {
	pl *wgpu.ComputePipeline
	ly *wgpu.BindGroupLayout
}
