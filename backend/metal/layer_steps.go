package metal

import (
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// Metal's names for internal/program's sublayers (OP4cu-b2 moved the derivation there, from this file): a layer's steps are its
// sublayers, each one of Metal's fused blocks.
type (
	layerStep = program.Sublayer
	stepSpan  = program.SublayerSpan
)

const (
	stepAttention   = program.SublayerAttention
	stepDeltaNet    = program.SublayerDeltaNet
	stepMamba       = program.SublayerMamba
	stepMLA         = program.SublayerMLA
	stepGemma4MoE   = program.SublayerGemma4MoE
	stepMoE         = program.SublayerMoE
	stepNonGatedFFN = program.SublayerNonGatedFFN
	stepDenseFFN    = program.SublayerDenseFFN
	stepPLE         = program.SublayerPLE
	stepLayerScale  = program.SublayerLayerScale
)

// layerStepsOf and checkStepClaims are program.SublayersOf and program.CheckSublayerClaims, whose declines name Metal.
func layerStepsOf(ops []program.Op) ([]stepSpan, error) { return program.SublayersOf(ops, "Metal") }

func checkStepClaims(ops []program.Op, spans []stepSpan) error {
	return program.CheckSublayerClaims(ops, spans)
}

// buildLayerSteps sets every layer's steps from p, and declines when a layer has an operation or parameter outside Metal's table
// (program.Metal), its operations match no step, a step consumes an operation its block does not emit, the steps differ from
// the blocks the layer's build loaded (its mixer, FFN form, PLE and layer scale), or the q4k lane would run a step its FFN does
// not cover. A layer scale step on a layer whose scalar is 1 or absent stays: its block then emits nothing.
func (r *resident) buildLayerSteps(p *program.Program) error {
	if len(p.Layers) != len(r.layers) {
		return fmt.Errorf("the program has %d layers and the resident %d", len(p.Layers), len(r.layers))
	}
	for l, ops := range p.Layers {
		err := program.Metal.CheckLayer(ops)
		var spans []stepSpan
		if err == nil {
			spans, err = layerStepsOf(ops)
		}
		if err == nil {
			err = checkStepClaims(ops, spans)
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
		L := &r.layers[l]
		steps := make([]layerStep, len(spans))
		for i, s := range spans {
			steps[i] = s.Kind
		}
		if want := r.loadedSteps(L); !slices.Equal(steps, want) {
			return fmt.Errorf("layer %d: the program's steps are %v and the layer's build loaded %v", l, steps, want)
		}
		if r.q4kLane && (!slices.Equal(steps, []layerStep{stepAttention, stepDenseFFN}) || spans[0].Parallel) {
			return fmt.Errorf("layer %d: the q4k lane runs a sequential attention and dense FFN step only, and the layer's steps are %v (a parallel block: %v)",
				l, steps, spans[0].Parallel)
		}
		L.steps = steps
	}
	return nil
}

// loadedSteps is the steps layer L's build loaded the weights for, in the order the layer body runs them.
func (r *resident) loadedSteps(L *residLayer) []layerStep {
	var s []layerStep
	switch {
	case L.delta != nil:
		s = append(s, stepDeltaNet)
	case L.mamba != nil:
		s = append(s, stepMamba)
	case L.mla != nil:
		s = append(s, stepMLA)
	case L.geom != nil:
		s = append(s, stepAttention)
	}
	if !L.ffn {
		return s
	}
	switch {
	case L.g4moe != nil:
		s = append(s, stepGemma4MoE)
	case L.moe != nil:
		s = append(s, stepMoE)
	case r.nonGatedMLP:
		s = append(s, stepNonGatedFFN)
	default:
		s = append(s, stepDenseFFN)
	}
	if L.pleGW != (Buffer{}) {
		s = append(s, stepPLE)
	}
	if L.layerScaleRead {
		s = append(s, stepLayerScale)
	}
	return s
}

// prefillBlocks is the steps Metal's batched prefill has a block for (resident.prefillLastImgSrc).
var prefillBlocks = []layerStep{stepAttention, stepDeltaNet, stepMamba, stepMoE, stepDenseFFN, stepNonGatedFFN, stepPLE, stepLayerScale}

// prefillStepsDecline is why the batched prefill cannot run p, or "": a layer whose operations match no step, or a step the
// pass has no block for (MLA's latent attention, Gemma 4's MoE block), named with its layer.
func prefillStepsDecline(p *program.Program) string {
	for l, ops := range p.Layers {
		spans, err := layerStepsOf(ops)
		if err != nil {
			return fmt.Sprintf("layer %d: %v", l, err)
		}
		for _, s := range spans {
			if !slices.Contains(prefillBlocks, s.Kind) {
				return fmt.Sprintf("layer %d: the batched prefill has no block for the %v step", l, s.Kind)
			}
		}
	}
	return ""
}
