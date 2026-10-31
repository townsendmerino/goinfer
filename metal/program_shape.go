package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/program"
)

// shapePlantForTest plants one defect in denseShapeOf, so B1-1 can show it goes red. Zero in production.
var shapePlantForTest shapePlant

type shapePlant uint8

const (
	shapePlantNone         shapePlant = iota
	shapePlantSandwichPost            // any post-attention norm reads as sandwich, a post-only layer's too
	shapePlantNoWhole                 // the QK-norm's Whole is read as false
)

// denseShape is the layer shape Metal's dense decode dispatches on (OP4m-b1, docs/tasks/task-forward-architecture-2026-10.md
// §4.9): the norm placement, the parallel block, the FFN's form, the norm's form, the QK-norm and its reach, the output
// projection's bias and learned positions. Each is a model-level flag because the encoder takes them at that level.
type denseShape struct {
	sandwich, postOnly, parallel bool
	nonGated, layerNorm          bool
	qkNorm, qkNormWhole          bool
	outBias, learnedPos          bool
}

// agreed is one part of the shape, which every layer that carries it must give the same value.
type agreed[T comparable] struct {
	set   bool
	v     T
	layer int
}

// see records layer l's value, or reports that it differs from an earlier layer's.
func (a *agreed[T]) see(part string, l int, v T) error {
	if !a.set {
		a.set, a.v, a.layer = true, v, l
		return nil
	}
	if a.v != v {
		return fmt.Errorf("layer %d's %s (%v) differs from layer %d's (%v), and the encoder takes one for the model", l, part, v, a.layer, a.v)
	}
	return nil
}

// denseShapeOf derives the shape from the model's program, operation by operation:
//   - a layer's placement: post-only when its input norm is a copy (program.NormCopy), parallel when a residual adds two
//     sublayers' outputs, sandwich when a post-attention norm follows a non-copy input norm, and pre-norm otherwise;
//   - a QK-norm is a weighted RMS norm of q and k per head: a qk-norm operation of form program.QKRMS (Llama 4's parameter-free
//     L2 form is not one), the QK-norm inside qwen3.5's gated attention, or Gemma 4's attention, which always norms q and k;
//   - a non-gated FFN is a dense one of form program.FfnNonGated, or Nemotron's MoE, whose experts are non-gated;
//   - the rest are the parameters of the norm and output-projection operations, and of the embedding stage.
//
// The encoder takes each part for the whole model, so a program is accepted only when its layers agree: every layer's placement
// is the same, or the program is the one mix the encoder handles per layer, Olmo Hybrid's post-only attention layers beside
// pre-norm recurrent layers; every layer with attention agrees on the QK-norm and its reach; and the layers that carry the norm,
// FFN and output projection agree on their parameters. A layer that asks for two placements at once, or a disagreement, is an
// error naming the layers.
func denseShapeOf(p *program.Program) (denseShape, error) {
	s := denseShape{learnedPos: p.Embed.LearnedPos}
	var layerNorm, nonGated, qkNorm, qkWhole, outBias agreed[string]
	yes := func(b bool) string {
		if b {
			return "yes"
		}
		return "no"
	}
	layers := make([]layerFacts, len(p.Layers))
	for l, ops := range p.Layers {
		copied := len(ops) > 0 && ops[0].Kind == program.OpNorm && ops[0].Norm.Form == program.NormCopy
		recurrent, attention, post, two, qk, whole := false, false, false, false, false, false
		for i := range ops {
			o := &ops[i]
			if o.Kind.Recurrent() {
				recurrent = true
			}
			switch o.Kind {
			case program.OpNorm:
				if o.Norm.Form != program.NormCopy {
					if err := layerNorm.see("norm form", l, yes(o.Norm.Form == program.NormLayer)); err != nil {
						return denseShape{}, err
					}
				}
				if o.Norm.Slot == program.SlotPostAttn && (!copied || shapePlantForTest == shapePlantSandwichPost) {
					post = true
				}
			case program.OpAttention, program.OpMLA:
				attention = true
			case program.OpQKNorm:
				attention = true
				if o.QKNorm.Form == program.QKRMS {
					qk, whole = true, o.QKNorm.Whole
				}
			case program.OpGatedAttn:
				attention = true
				if o.Gated.QKNorm.Form == program.QKRMS {
					qk, whole = true, o.Gated.QKNorm.Whole
				}
			case program.OpGemma4Attn:
				attention, qk = true, true
			case program.OpProjO:
				if err := outBias.see("output bias", l, yes(o.Proj.Bias)); err != nil {
					return denseShape{}, err
				}
			case program.OpFFN:
				if err := nonGated.see("FFN form (non-gated)", l, yes(o.FFN.Form == program.FfnNonGated || o.FFN.Form == program.FfnNemotronMoE)); err != nil {
					return denseShape{}, err
				}
			case program.OpResidual:
				two = two || o.Res.Two
			}
		}
		if btoi(copied)+btoi(two)+btoi(post) > 1 && shapePlantForTest == shapePlantNone {
			return denseShape{}, fmt.Errorf("layer %d asks for two placements at once (post-only %v, parallel %v, sandwich %v), and the encoder takes one",
				l, copied, two, post)
		}
		placement := "pre-norm"
		switch {
		case post && copied && shapePlantForTest == shapePlantSandwichPost:
			placement = "sandwich"
		case copied:
			placement = "post-only"
		case two:
			placement = "parallel"
		case post:
			placement = "sandwich"
		}
		layers[l] = layerFacts{placement, attention, recurrent}
		if attention {
			if err := qkNorm.see("QK-norm", l, yes(qk)); err != nil {
				return denseShape{}, err
			}
			if qk && shapePlantForTest != shapePlantNoWhole {
				if err := qkWhole.see("QK-norm reach (whole)", l, yes(whole)); err != nil {
					return denseShape{}, err
				}
			}
		}
	}
	model := ""
	if len(layers) > 0 {
		model = layers[0].placement
	}
	for l, f := range layers {
		if f.placement == model {
			continue
		}
		if !olmoHybridMix(layers) {
			return denseShape{}, fmt.Errorf("layer %d's placement (%s) differs from layer 0's (%s), and the encoder takes one for the model; the one mix it handles per layer is post-only attention beside pre-norm recurrent layers",
				l, f.placement, model)
		}
		model = "post-only"
		break
	}
	s.sandwich, s.postOnly, s.parallel = model == "sandwich", model == "post-only", model == "parallel"
	s.layerNorm, s.nonGated, s.outBias = layerNorm.v == "yes", nonGated.v == "yes", outBias.v == "yes"
	s.qkNorm, s.qkNormWhole = qkNorm.v == "yes", qkWhole.v == "yes"
	return s, nil
}

// layerFacts is what denseShapeOf records of one layer: its placement and whether it has an attention or a recurrent mixer.
type layerFacts struct {
	placement            string
	attention, recurrent bool
}

// olmoHybridMix reports whether layers are Olmo Hybrid's mix: every layer either an attention layer, post-only, or a recurrent
// layer, pre-norm, with at least one of each. The encoder applies post-only to the attention layers alone (resident.postOnly
// with the layer's geometry), so this is the one mixed placement it can run.
func olmoHybridMix(layers []layerFacts) bool {
	attn, rec := 0, 0
	for _, f := range layers {
		switch {
		case f.attention && !f.recurrent && f.placement == "post-only":
			attn++
		case f.recurrent && !f.attention && f.placement == "pre-norm":
			rec++
		default:
			return false
		}
	}
	return attn > 0 && rec > 0
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
