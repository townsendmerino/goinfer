package memprice

import (
	"fmt"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// LayerShapeOf returns layer l's widths for its operations' buffers (program.Buffers).
func LayerShapeOf(a *darch.Architecture, l int, ops []program.Op) program.LayerShape {
	s := program.LayerShape{KVHeads: a.KVHeadsAt(l), HeadDim: a.HeadDimAt(l), Hidden: a.HiddenDim, Heads: a.HeadsAt(l), FFN: a.FFNAt(l)}
	for _, o := range ops {
		switch o.Kind {
		case program.OpMLA:
			p := a.MLA
			s.MLALatent = p.KVLoRARank + p.QKRopeHeadDim
			s.MLA.QLoRA, s.MLA.Rank, s.MLA.QKHead, s.MLA.VHead = p.QLoRARank, p.KVLoRARank, p.QKNopeHeadDim+p.QKRopeHeadDim, p.VHeadDim
		case program.OpDeltaNet:
			p, d := a.Qwen35, &s.DeltaNet
			d.KeyHeads, d.KeyHeadDim, d.ValueHeads, d.ValueHeadDim, d.ConvKernel = p.NumKeyHeads, p.KeyHeadDim, p.NumValueHeads, p.ValueHeadDim, p.ConvKernel
		case program.OpMamba2:
			p, d := a.Mamba2Shapes(o.Mamba.NormGroups), &s.Mamba2
			d.Heads, d.HeadDim, d.DState, d.ConvDim, d.ConvKernel = p.NHeads, p.HeadDim, p.DState, p.Mixer().ConvDim(), p.DConv
		case program.OpKDA:
			p, d := a.KDA, &s.KDA
			d.Heads, d.HeadDim, d.ConvKernel = p.NumHeads, p.HeadDim, p.ConvKernel
		case program.OpShortConv:
			s.ShortConv.Dim, s.ShortConv.LCache = a.LFM2.ConvDim, a.LFM2.ConvLCache
		case program.OpPLE:
			s.PerLayerEmbed = a.Gemma4.HiddenSizePerLayerInput
		case program.OpGemma4MoE:
			s.Gemma4MoE.Experts, s.Gemma4MoE.TopK, s.Gemma4MoE.ExpertFFN = a.MoE.NumExperts, a.MoE.TopK, a.MoE.IntermediateDim
		}
	}
	return s
}

// LayerBuffers returns what layer l's operations declare beyond their weights.
func LayerBuffers(a *darch.Architecture, l int) []program.Buf {
	ops := a.LayerOps(l, false)
	return program.LayerBuffers(ops, LayerShapeOf(a, l, ops))
}

// LayerBuffersOf returns LayerBuffers bound to a.
func LayerBuffersOf(a *darch.Architecture) func(l int) []program.Buf {
	return func(l int) []program.Buf { return LayerBuffers(a, l) }
}

// StateSpec is one buffer of a recurrent or conv mixer's state for one KV slot, before allocator rounding.
type StateSpec struct {
	Name   string
	Floats int
}

func stateFloats(b program.Buf) int {
	if b.Prec != program.PrecF32 {
		panic(fmt.Sprintf("decoder: state buffer %q is declared at precision %d; recurrent state is f32", b.Name, b.Prec))
	}
	return b.Elems
}

// StateSpecs returns layer l's per-slot recurrent or conv state buffers, or nil when the layer has none.
func StateSpecs(a *darch.Architecture, l int) []StateSpec {
	var out []StateSpec
	for _, b := range LayerBuffers(a, l) {
		if b.Unit == program.PerSlot {
			out = append(out, StateSpec{b.Name, stateFloats(b)})
		}
	}
	return out
}

// StateBuffers returns one KV slot's recurrent and conv state for a, in bytes per buffer.
func StateBuffers(a *darch.Architecture) []int64 {
	var out []int64
	for l := range a.NumLayers {
		for _, s := range StateSpecs(a, l) {
			if s.Floats > 0 {
				out = append(out, int64(s.Floats)*4)
			}
		}
	}
	return out
}

// RecurrentCheckpointBuffers returns the per-slot buffers marked checkpointed, in bytes per buffer.
func RecurrentCheckpointBuffers(a *darch.Architecture) []int64 {
	var out []int64
	for l := range a.NumLayers {
		for _, b := range LayerBuffers(a, l) {
			if b.Unit == program.PerSlot && b.Checkpointed && b.Elems > 0 {
				out = append(out, int64(stateFloats(b))*4)
			}
		}
	}
	return out
}

// LayerRopeInvFreq returns layer i's resident RoPE inverse-frequency table as float32, matching BackendModelView.Layer(i).RopeInvFreq.
func LayerRopeInvFreq(a *darch.Architecture, i int) []float32 {
	if a.Gemma4 != nil {
		hd, rot, base := a.HeadDim, a.HeadDim, a.RoPELocalBase
		if a.IsGlobalLayer(i) {
			hd, rot, base = a.Gemma4.GlobalHeadDim, a.Gemma4.GlobalRotaryDim, a.RoPEGlobalBase
		}
		f := archparts.Gemma4InvFreq(hd, rot, base)
		out := make([]float32, len(f))
		for j, val := range f {
			out[j] = float32(val)
		}
		return out
	}
	inv := a.RoPEInvFreq(i)
	width := len(inv)
	if width == 0 {
		width = a.EffectiveRotaryDim() / 2
	}
	out := make([]float32, width)
	if a.IsNoPELayer(i) {
		return out
	}
	for j, v := range inv {
		out[j] = float32(v)
	}
	return out
}

// LayerRotaryDim returns layer i's rotary dimension on resident backends.
func LayerRotaryDim(a *darch.Architecture, i int) int {
	if a.Gemma4 != nil {
		return a.HeadDimAt(i)
	}
	if a.RotaryDim != 0 && a.RotaryDim < a.HeadDim {
		return a.RotaryDim
	}
	return a.HeadDim
}

// MRoPEAxis returns the per-pair axis map of width half when a uses 3-section m-RoPE, else nil.
func MRoPEAxis(a *darch.Architecture, half int) []int {
	sec := a.MRopeSection
	if len(sec) != 3 || half <= 0 {
		return nil
	}
	comp := rope.MRoPEComponent
	if a.MRopeInterleaved {
		comp = rope.MRoPEComponentInterleaved
	}
	axis := make([]int, half)
	for d := range axis {
		axis[d] = comp(d, sec)
	}
	return axis
}

// VFromK reports whether layer i derives V from K without a separate V projection.
func VFromK(w *darch.Weights, i int) bool {
	if w == nil || w.Arch == nil || w.Arch.Gemma4 == nil {
		return false
	}
	if w.Arch.Gemma4.KVShared && w.Arch.IsGlobalLayer(i) {
		return true
	}
	return i >= 0 && i < len(w.Layers) && w.Layers[i].VFromK
}

// LayerOps classifies layer i's resident mixer and whether it has an FFN sublayer.
func LayerOps(a *darch.Architecture, i int) (mixer program.Mixer, ffn bool) {
	switch {
	case a.Nemotron != nil:
		switch a.Nemotron.BlockKind[i] {
		case archparts.NemoMamba:
			return program.MixerMamba, false
		case archparts.NemoAttn:
			return program.MixerAttention, false
		default:
			return program.MixerNone, true
		}
	case a.Qwen35 != nil && a.IsLinearLayer(i):
		return program.MixerDeltaNet, true
	case a.Granite != nil && a.IsMambaLayer(i):
		return program.MixerMamba, true
	case a.MLA != nil:
		return program.MixerMLA, true
	}
	return program.MixerAttention, true
}
