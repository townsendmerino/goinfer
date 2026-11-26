package decoder

import (
	"sync/atomic"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// buildProgramCalls counts the programs buildProgram builds, so a test can see that a fit search builds none per probe.
var buildProgramCalls atomic.Int64

// buildProgram builds a's program. experts reports whether layer l carries MoE experts: a MoE family's leading dense layers
// (first_k_dense_replace) do not, and only the weights say so.
func buildProgram(a *darch.Architecture, experts func(l int) bool) *layerProgram {
	buildProgramCalls.Add(1)
	p := &layerProgram{
		Embed: embedParams{Scale: a.EffectiveEmbedScale(), LearnedPos: a.LearnedPosEmbed, PLE: a.Gemma4 != nil && a.Gemma4.HiddenSizePerLayerInput > 0},
		Head: headParams{
			Norm: normParams{Slot: slotFinal, Form: a.NormForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true},
			Tied: a.TiedLMHead, Softcap: a.FinalLogitSoftcap, LogitScale: a.LogitScale,
		},
	}
	p.Layers = make([][]op, a.NumLayers)
	for l := range a.NumLayers {
		p.Layers[l] = a.LayerOps(l, experts != nil && experts(l))
	}
	return p
}
