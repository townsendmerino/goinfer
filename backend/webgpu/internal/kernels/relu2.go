//go:build gpu

package kernels

import (
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
)

func (c *Context) ensureRelu2() error {
	if c.relu2Pipeline != nil {
		return nil
	}
	code, err := c.quantSrc("relu2Quant", wgsl.Relu2QuantWGSL)
	if err != nil {
		return err
	}
	sh, pl, lay, err := c.buildCompute("relu2Quant", code)
	if err != nil {
		return err
	}
	c.relu2Shader, c.relu2Pipeline, c.relu2Layout = sh, pl, lay
	return nil
}

// relu2QuantOp records the fused relu²(up)→int8 dispatch into the resident plan, returning the
// packed int8 activation + its scale (for the down-projection W8A8 GEMV).
func (c *Context) relu2QuantBind(up, qout, scales, dims *wgpu.Buffer) (*wgpu.BindGroup, error) {
	return c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.relu2Layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: up, Size: up.GetSize()}, {Binding: 1, Buffer: qout, Size: qout.GetSize()},
		{Binding: 2, Buffer: scales, Size: scales.GetSize()}, {Binding: 3, Buffer: dims, Size: dims.GetSize()},
	}})
}
