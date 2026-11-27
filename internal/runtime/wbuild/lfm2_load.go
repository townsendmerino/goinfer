package wbuild

import (
	"github.com/townsendmerino/aikit/embed"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
)

// loadLFM2Conv reads one LFM2 gated short-convolution mixer from a safetensors checkpoint.
//
// Shapes are checked against the ARCHITECTURE (convDim/hidden from the config), not merely
// against each other: in_proj [3*convDim, hidden] and conv [convDim, 1, K] can be mutually
// consistent and still be the wrong tensors, and a silently mis-shaped mixer produces fluent
// output rather than an error. Same discipline gptoss_safetensors.go states for its blocks.
//
// Stored f32, matching the qwen3_5_moe linear layers: this is a parity-first forward, and the
// conv is ~0.6% of the model's bytes (three tensors per layer against a 10752-wide FFN), so
// quantizing it would buy little and add a numeric variable to a new family's first oracle.
func loadLFM2Conv(st *embed.SafetensorsFile, i int, l *darch.LayerWeights, arch *darch.Architecture, hidden int,
	tn func(int, string) string,
) error {
	g := arch.LFM2
	nm := func(suf string) string { return tn(i, suf) }
	c := &archparts.ShortConvWeights{}
	var err error
	if c.InProj, err = giwcodec.STIsland(st, nm("conv.in_proj.weight"), 3*g.ConvDim, hidden); err != nil {
		return err
	}
	// [convDim, 1, K] on disk — the middle dim is the depthwise groups=channels marker and is
	// always 1, so it flattens to [convDim, K].
	if c.ConvW, err = st.TensorF32(nm("conv.conv.weight"), g.ConvDim, 1, g.ConvLCache); err != nil {
		return err
	}
	if c.OutProj, err = giwcodec.STIsland(st, nm("conv.out_proj.weight"), hidden, g.ConvDim); err != nil {
		return err
	}
	l.ShortConv = c
	return nil
}
