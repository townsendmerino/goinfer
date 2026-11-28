//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Multi-row int4 projections: M quantized activation rows against one ResidentW4A8, the weight streamed once for all of
// them. Two kernels compute the same product, each bit for bit the single-row kernel's. Both read the activations in the layout quantizeShaderWGSL writes with
// np = the weight's kPad() ([M, kp/4] packed int8 words, one scale a row) and write dst [M, N] row-major.

// w4MultiMaxRows is the most rows one dispatch of the row-shaped kernel takes.
const w4MultiMaxRows = 16

// W4MultiKind names one of the two multi-row int4 kernels.
type W4MultiKind int

const (
	W4MultiRows  W4MultiKind = iota // gemmRowsW4A8ShaderWGSL
	W4MultiTiled                    // matmulTiledW4A8KernelWGSL
)

func (k W4MultiKind) String() string {
	if k == W4MultiTiled {
		return "tiled"
	}
	return "rows"
}

// W4MultiVerified lists the WebGPU backends on which the multi-row int4 kernels have been checked bit for bit against
// the single-row kernel and the batched pass against the row-by-row walk (TestW4Multi_matchesSingleRow,
// TestPrefillInt4_matchesWalk). Each kernel's output is a float sum in a fixed order, and a backend's shader compiler
// may evaluate the same expression differently in two kernels, so the identity is a property of a backend, not of the
// source. The batched pass takes an int4 model only on a backend listed here; add one after those tests pass on it.
var W4MultiVerified = map[string]bool{"metal": true, "vulkan": true}

// w4MultiFor is the kernel for a projection over M rows: the row-shaped one while a single dispatch of it holds the
// rows, the tiled one beyond. The tiled kernel computes whole 64-row tiles however few rows there are, so below a
// block of rows it does more work than the row-shaped one. The choice changes speed only: both give the same bits.
func (c *Context) w4MultiFor(M int) W4MultiKind {
	if c.w4MultiPin != 0 {
		return W4MultiKind(c.w4MultiPin - 1)
	}
	if M <= w4MultiMaxRows {
		return W4MultiRows
	}
	return W4MultiTiled
}

// w4MultiSource is kind's shader at this context's dot4 flavour.
func (c *Context) w4MultiSource(kind W4MultiKind) (code, label string) {
	if kind == W4MultiRows {
		return wgsl.GemmRowsW4A8ShaderWGSL, "gemmRowsW4A8"
	}
	if c.hasDP4A {
		return fmt.Sprintf(wgsl.MatmulTiledW4A8KernelWGSL, wgsl.Dot4PackedWGSL), "matmulTiledW4A8DP4A"
	}
	return fmt.Sprintf(wgsl.MatmulTiledW4A8KernelWGSL, wgsl.Dot4UnpackedWGSL), "matmulTiledW4A8"
}

// ensureW4Multi compiles both multi-row int4 kernels.
func (c *Context) ensureW4Multi() error {
	for kind := range c.w4Multi {
		if c.w4Multi[kind].Pl != nil {
			continue
		}
		code, label := c.w4MultiSource(W4MultiKind(kind))
		sh, pl, layout, err := c.mkPipeline(label, code)
		if err != nil {
			return err
		}
		c.w4Multi[kind] = tiledPipes{sh: sh, Pl: pl, Layout: layout}
	}
	return nil
}

// encodeW4Multi encodes dst[M, N] = aq[M, kp/4] · rmᵀ with kind's kernel into disp. p is kind's pipeline and layout
// (c.w4Multi[kind], or a test's own build of the source). uni makes a uniform buffer and bind a bind group, as the
// caller's encoder does; aq and aScales are the quantized rows, dst is at least M*N floats.
func encodeW4Multi(kind W4MultiKind, p tiledPipes, rm *ResidentW4A8, M int, aq, aScales, dst *wgpu.Buffer,
	uni func([]uint32) *wgpu.Buffer, bind func(*wgpu.BindGroupLayout, ...*wgpu.Buffer) *wgpu.BindGroup,
	disp func(*wgpu.ComputePipeline, *wgpu.BindGroup, uint32, uint32)) {
	N := rm.rows
	if kind == W4MultiTiled {
		d := uni([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0})
		disp(p.Pl, bind(p.Layout, aq, rm.bq, aScales, rm.bScales, dst, d), (uint32(N)+63)/64, (uint32(M)+63)/64)
		return
	}
	gx, gy := GemvGrid(N)
	for row0 := 0; row0 < M; row0 += w4MultiMaxRows {
		d := uni([]uint32{uint32(min(M-row0, w4MultiMaxRows)), uint32(rm.kp), uint32(N), uint32(row0)})
		disp(p.Pl, bind(p.Layout, aq, rm.bq, aScales, rm.bScales, dst, d), gx, gy)
	}
}
