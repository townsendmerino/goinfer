package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"

	"slices"

	"github.com/townsendmerino/aikit/embed"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
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

// shortConvState is one LFM2 conv layer's rolling window: the last K-1 Bx vectors, oldest
// first, where K is conv_L_cache. The K-th tap is the current token, so it is never stored.
//
// This is a strict subset of mamba2State — that one carries convWin PLUS the SSM state, and
// LFM2 has no SSM. Kept as its own type rather than reusing mamba2State so a conv layer
// cannot silently acquire an unused ssm field that some future code path starts reading.
type shortConvState struct {
	convWin [][]float32 // up to K-1 prior Bx vectors, oldest first
}

func newShortConvState() *shortConvState { return &shortConvState{} }

// push appends this token's Bx and keeps at most K-1 entries, dropping the oldest.
//
// Storing K-1 rather than K is the invariant the step function depends on: the current
// token is applied directly from its own tap, so a window of K would double-count it.
func (s *shortConvState) push(bx []float32, K int) {
	if K <= 1 {
		return // kernel of 1 has no history
	}
	s.convWin = append(s.convWin, slices.Clone(bx))
	if len(s.convWin) > K-1 {
		s.convWin = s.convWin[len(s.convWin)-(K-1):]
	}
}
