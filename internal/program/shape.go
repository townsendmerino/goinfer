package program

import (
	"fmt"
)

// ShapePlantForTest plants one defect in ShapeOf or MixerOf, so a backend's shape test can show it goes red. It is a test hook:
// zero in production, set and reset only by tests.
var ShapePlantForTest ShapePlant

// ShapePlant names one planted defect.
type ShapePlant uint8

const (
	ShapePlantNone            ShapePlant = iota
	ShapePlantSandwichPost               // any post-attention norm reads as sandwich, a post-only layer's too
	ShapePlantNoWhole                    // the QK-norm's Whole is read as false
	ShapePlantFFNOnlyAsAttend            // MixerOf reads a layer with no mixer as an attention layer
)

// Shape is the layer shape a resident's dense layer body dispatches on (OP4m-b1 and OP4cu-b1,
// docs/tasks/task-forward-architecture-2026-10.md §4.9 and §4.9a): the norm placement, the parallel block, the FFN's form, the
// norm's form, the QK-norm and its reach, the output projection's bias and learned positions. Each is a model-level flag because
// the encoders take them at that level.
type Shape struct {
	Sandwich, PostOnly, Parallel bool
	NonGated, LayerNorm          bool
	QKNorm, QKNormWhole          bool
	OutBias, LearnedPos          bool
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

// ShapeOf derives the shape from the model's program, operation by operation:
//   - a layer's placement: post-only when its input norm is a copy (NormCopy), parallel when a residual adds two
//     sublayers' outputs, sandwich when a post-attention norm follows a non-copy input norm, and pre-norm otherwise;
//   - a QK-norm is a weighted RMS norm of q and k per head: a qk-norm operation of form QKRMS (Llama 4's parameter-free
//     L2 form is not one), the QK-norm inside qwen3.5's gated attention, or Gemma 4's attention, which always norms q and k;
//   - a non-gated FFN is a dense one of form FfnNonGated, or Nemotron's MoE, whose experts are non-gated;
//   - the rest are the parameters of the norm and output-projection operations, and of the embedding stage.
//
// The encoder takes each part for the whole model, so a program is accepted only when its layers agree: every layer's placement
// is the same, or the program is the one mix the encoder handles per layer, Olmo Hybrid's post-only attention layers beside
// pre-norm recurrent layers, and a placement other than pre-norm is only ever an attention layer's; every layer with attention
// agrees on the QK-norm and its reach; and the layers that carry the norm, FFN and output projection agree on their parameters.
// A layer that asks for two placements at once or carries both an attention and a recurrent mixer, or a disagreement, is an
// error naming the layers.
func ShapeOf(p *Program) (Shape, error) {
	s := Shape{LearnedPos: p.Embed.LearnedPos}
	var layerNorm, nonGated, qkNorm, qkWhole, outBias agreed[string]
	yes := func(b bool) string {
		if b {
			return "yes"
		}
		return "no"
	}
	layers := make([]layerFacts, len(p.Layers))
	for l, ops := range p.Layers {
		copied := len(ops) > 0 && ops[0].Kind == OpNorm && ops[0].Norm.Form == NormCopy
		recurrent, attention, post, two, qk, whole := false, false, false, false, false, false
		for i := range ops {
			o := &ops[i]
			if o.Kind.Recurrent() {
				recurrent = true
			}
			switch o.Kind {
			case OpNorm:
				if o.Norm.Form != NormCopy {
					if err := layerNorm.see("norm form", l, yes(o.Norm.Form == NormLayer)); err != nil {
						return Shape{}, err
					}
				}
				if o.Norm.Slot == SlotPostAttn && (!copied || ShapePlantForTest == ShapePlantSandwichPost) {
					post = true
				}
			case OpAttention, OpMLA:
				attention = true
			case OpQKNorm:
				attention = true
				if o.QKNorm.Form == QKRMS {
					qk, whole = true, o.QKNorm.Whole
				}
			case OpGatedAttn:
				attention = true
				if o.Gated.QKNorm.Form == QKRMS {
					qk, whole = true, o.Gated.QKNorm.Whole
				}
			case OpGemma4Attn:
				attention, qk = true, true
			case OpProjO:
				if err := outBias.see("output bias", l, yes(o.Proj.Bias)); err != nil {
					return Shape{}, err
				}
			case OpFFN:
				if err := nonGated.see("FFN form (non-gated)", l, yes(o.FFN.Form == FfnNonGated || o.FFN.Form == FfnNemotronMoE)); err != nil {
					return Shape{}, err
				}
			case OpResidual:
				two = two || o.Res.Two
			}
		}
		if attention && recurrent {
			return Shape{}, fmt.Errorf("layer %d has both an attention and a recurrent mixer, and the encoder runs one mixer a layer", l)
		}
		if btoi(copied)+btoi(two)+btoi(post) > 1 && ShapePlantForTest == ShapePlantNone {
			return Shape{}, fmt.Errorf("layer %d asks for two placements at once (post-only %v, parallel %v, sandwich %v), and the encoder takes one",
				l, copied, two, post)
		}
		placement := "pre-norm"
		switch {
		case post && copied && ShapePlantForTest == ShapePlantSandwichPost:
			placement = "sandwich"
		case copied:
			placement = "post-only"
		case two:
			placement = "parallel"
		case post:
			placement = "sandwich"
		}
		if placement != "pre-norm" && !attention {
			return Shape{}, fmt.Errorf("layer %d is %s without an attention mixer, and the encoder applies that placement to attention layers only",
				l, placement)
		}
		layers[l] = layerFacts{placement, attention, recurrent}
		if attention {
			if err := qkNorm.see("QK-norm", l, yes(qk)); err != nil {
				return Shape{}, err
			}
			if qk && ShapePlantForTest != ShapePlantNoWhole {
				if err := qkWhole.see("QK-norm reach (whole)", l, yes(whole)); err != nil {
					return Shape{}, err
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
		if bad, ok := olmoHybridMix(layers); !ok {
			return Shape{}, fmt.Errorf("layer %d's placement (%s) differs from layer 0's (%s), and the encoder takes one for the model; the one mix it handles per layer is post-only attention beside pre-norm recurrent layers, which %s",
				l, f.placement, model, bad)
		}
		model = "post-only"
		break
	}
	s.Sandwich, s.PostOnly, s.Parallel = model == "sandwich", model == "post-only", model == "parallel"
	s.LayerNorm, s.NonGated, s.OutBias = layerNorm.v == "yes", nonGated.v == "yes", outBias.v == "yes"
	s.QKNorm, s.QKNormWhole = qkNorm.v == "yes", qkWhole.v == "yes"
	return s, nil
}

// layerFacts is what ShapeOf records of one layer: its placement and whether it has an attention or a recurrent mixer.
type layerFacts struct {
	placement            string
	attention, recurrent bool
}

// olmoHybridMix reports whether layers, whose placements differ (its one caller's condition), are Olmo Hybrid's mix: every layer
// either an attention layer, post-only, or a recurrent layer, pre-norm. Two placements guarantee one of each. The encoder
// applies post-only to the attention layers alone (the backend's post-only flag with the layer's geometry), so this is the one mixed
// placement it can run. When they are not, bad names the first layer that fits neither half.
func olmoHybridMix(layers []layerFacts) (bad string, ok bool) {
	for l, f := range layers {
		switch {
		case f.attention && !f.recurrent && f.placement == "post-only":
		case f.recurrent && !f.attention && f.placement == "pre-norm":
		default:
			kind := "an FFN-only"
			switch {
			case f.attention:
				kind = "an attention"
			case f.recurrent:
				kind = "a recurrent"
			}
			return fmt.Sprintf("layer %d (%s layer, %s) is not", l, kind, f.placement), false
		}
	}
	return "", true
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
