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

// denseShapeOf derives the shape from the model's program, operation by operation:
//   - post-only placement is a layer whose input norm is a copy (program.NormCopy);
//   - sandwich placement is a post-attention norm after an input norm that is not a copy;
//   - a parallel block is a residual that adds two sublayers' outputs;
//   - a QK-norm is a weighted RMS norm of q and k per head: a qk-norm operation of form program.QKRMS (Llama 4's parameter-free
//     L2 form is not one), the QK-norm inside qwen3.5's gated attention, or Gemma 4's attention, which always norms q and k;
//   - a non-gated FFN is a dense one of form program.FfnNonGated, or Nemotron's MoE, whose experts are non-gated;
//   - the rest are the parameters of the norm and output-projection operations, and of the embedding stage.
//
// A model whose layers ask for two placements the encoder cannot hold at once is an error naming the layer.
func denseShapeOf(p *program.Program) (denseShape, error) {
	s := denseShape{learnedPos: p.Embed.LearnedPos}
	for l, ops := range p.Layers {
		copied := len(ops) > 0 && ops[0].Kind == program.OpNorm && ops[0].Norm.Form == program.NormCopy
		for i := range ops {
			o := &ops[i]
			switch o.Kind {
			case program.OpNorm:
				if o.Norm.Form == program.NormLayer {
					s.layerNorm = true
				}
				if o.Norm.Slot == program.SlotPostAttn && (!copied || shapePlantForTest == shapePlantSandwichPost) {
					s.sandwich = true
				}
			case program.OpQKNorm:
				if o.QKNorm.Form == program.QKRMS {
					s.qkNorm = true
					s.qkNormWhole = s.qkNormWhole || (o.QKNorm.Whole && shapePlantForTest != shapePlantNoWhole)
				}
			case program.OpGatedAttn:
				if o.Gated.QKNorm.Form == program.QKRMS {
					s.qkNorm = true
					s.qkNormWhole = s.qkNormWhole || (o.Gated.QKNorm.Whole && shapePlantForTest != shapePlantNoWhole)
				}
			case program.OpGemma4Attn:
				s.qkNorm = true
			case program.OpProjO:
				s.outBias = s.outBias || o.Proj.Bias
			case program.OpFFN:
				s.nonGated = s.nonGated || o.FFN.Form == program.FfnNonGated || o.FFN.Form == program.FfnNemotronMoE
			case program.OpResidual:
				s.parallel = s.parallel || o.Res.Two
			}
		}
		if copied {
			s.postOnly = true
		}
		if n := btoi(s.sandwich) + btoi(s.postOnly) + btoi(s.parallel); n > 1 && shapePlantForTest == shapePlantNone {
			return denseShape{}, fmt.Errorf("layer %d: the program mixes placements (sandwich %v, post-only %v, parallel %v), and the encoder takes one",
				l, s.sandwich, s.postOnly, s.parallel)
		}
	}
	return s, nil
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
