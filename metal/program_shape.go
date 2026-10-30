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
// The encoder takes each part for the whole model, so every layer that carries a part must agree on it, and a program whose layers
// disagree is an error naming both. Attention layers must share one placement; a layer with a recurrent mixer may instead be
// pre-norm (Olmo Hybrid's DeltaNet layers beside its post-only attention layers), which the encoder handles per layer.
func denseShapeOf(p *program.Program) (denseShape, error) {
	s := denseShape{learnedPos: p.Embed.LearnedPos}
	var attnPlacement, recPlacement, layerNorm, nonGated, qkNorm, qkWhole, outBias agreed[string]
	yes := func(b bool) string {
		if b {
			return "yes"
		}
		return "no"
	}
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
		switch {
		case attention && !recurrent:
			if err := attnPlacement.see("placement", l, placement); err != nil {
				return denseShape{}, err
			}
			if err := qkNorm.see("QK-norm", l, yes(qk)); err != nil {
				return denseShape{}, err
			}
			if qk && shapePlantForTest != shapePlantNoWhole {
				if err := qkWhole.see("QK-norm reach (whole)", l, yes(whole)); err != nil {
					return denseShape{}, err
				}
			}
		case placement != "pre-norm":
			if err := recPlacement.see("placement", l, placement); err != nil {
				return denseShape{}, err
			}
		}
	}
	model := attnPlacement.v
	if !attnPlacement.set {
		model = recPlacement.v
	} else if recPlacement.set && recPlacement.v != model {
		return denseShape{}, fmt.Errorf("layer %d's placement (%s) differs from layer %d's (%s), and the encoder takes one for the model",
			recPlacement.layer, recPlacement.v, attnPlacement.layer, model)
	}
	s.sandwich, s.postOnly, s.parallel = model == "sandwich", model == "post-only", model == "parallel"
	s.layerNorm, s.nonGated, s.outBias = layerNorm.v == "yes", nonGated.v == "yes", outBias.v == "yes"
	s.qkNorm, s.qkNormWhole = qkNorm.v == "yes", qkWhole.v == "yes"
	return s, nil
}
