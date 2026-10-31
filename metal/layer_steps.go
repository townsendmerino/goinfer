package metal

import (
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/internal/program"
)

// layerStep is one fused block of Metal's per-token layer body (OP4m-b2, docs/tasks/task-forward-architecture-2026-10.md §4.9).
// Each emits exactly the dispatches its block always has and reads its own parameters from the layer; the steps of a layer run
// in its program's order.
type layerStep uint8

const (
	stepAttention   layerStep = iota + 1 // input norm through the output projection, any post-attention norm, the residual add
	stepDeltaNet                         // the Gated DeltaNet mixer sublayer
	stepMamba                            // the Mamba-2 mixer sublayer
	stepMLA                              // the Multi-head Latent Attention sublayer
	stepGemma4MoE                        // Gemma 4's parallel dense and MoE block
	stepMoE                              // a MoE FFN sublayer
	stepNonGatedFFN                      // a non-gated dense FFN sublayer
	stepDenseFFN                         // a gated dense FFN sublayer, or a parallel block's FFN half
	stepPLE                              // Gemma 4's per-layer-embedding branch
	stepLayerScale                       // Gemma 4's per-layer output scalar
)

var stepNames = [...]string{stepAttention: "attention", stepDeltaNet: "DeltaNet", stepMamba: "Mamba-2", stepMLA: "MLA",
	stepGemma4MoE: "Gemma 4 MoE block", stepMoE: "MoE FFN", stepNonGatedFFN: "non-gated FFN", stepDenseFFN: "dense FFN",
	stepPLE: "PLE", stepLayerScale: "layer scale"}

func (s layerStep) String() string {
	if int(s) < len(stepNames) && stepNames[s] != "" {
		return stepNames[s]
	}
	return fmt.Sprintf("layerStep(%d)", uint8(s))
}

// stepSpan is a step and the operations of its layer it consumes, ops[from:to].
type stepSpan struct {
	step     layerStep
	from, to int
}

// stepPlantForTest makes layerStepsOf's attention step also consume the operation after its residual add (the FFN sublayer's
// input norm), so OP4m-b2's M2 can show the claim check goes red. False in production.
var stepPlantForTest bool

// layerStepsOf matches one layer's operations against the fusion table. A sublayer runs to its residual add; one with a mixer is
// that mixer's step, one with an FFN the FFN form's step, and a parallel block's single sublayer is the attention step up to its
// FFN and the dense FFN step from there. Gemma 4's MoE block, PLE and layer scale are steps of one operation. An operation no
// entry matches is an error naming it.
func layerStepsOf(ops []program.Op) ([]stepSpan, error) {
	var spans []stepSpan
	for i := 0; i < len(ops); {
		switch ops[i].Kind {
		case program.OpGemma4MoE, program.OpPLE, program.OpLayerScale:
			spans = append(spans, stepSpan{map[program.OpKind]layerStep{program.OpGemma4MoE: stepGemma4MoE, program.OpPLE: stepPLE,
				program.OpLayerScale: stepLayerScale}[ops[i].Kind], i, i + 1})
			i++
			continue
		}
		end := i
		for end < len(ops) && ops[end].Kind != program.OpResidual {
			end++
		}
		if end == len(ops) {
			return nil, fmt.Errorf("operations %d to %d end without a residual add, and no step ends there", i, end-1)
		}
		end++
		mixer, ffn := layerStep(0), -1
		for j := i; j < end; j++ {
			switch k := ops[j].Kind; k {
			case program.OpAttention, program.OpGatedAttn, program.OpGemma4Attn:
				mixer = stepAttention
			case program.OpDeltaNet:
				mixer = stepDeltaNet
			case program.OpMamba2:
				mixer = stepMamba
			case program.OpMLA:
				mixer = stepMLA
			case program.OpFFN:
				ffn = j
			case program.OpNorm, program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpOutGate,
				program.OpProjO, program.OpResidual:
			default:
				return nil, fmt.Errorf("operation %d (%v) has no step on Metal", j, k)
			}
		}
		switch {
		case mixer != 0 && ffn >= 0:
			spans = append(spans, stepSpan{mixer, i, ffn}, stepSpan{ffnStep(ops[ffn].FFN.Form), ffn, end})
		case mixer != 0:
			if stepPlantForTest && mixer == stepAttention && end < len(ops) {
				end++
			}
			spans = append(spans, stepSpan{mixer, i, end})
		case ffn >= 0:
			spans = append(spans, stepSpan{ffnStep(ops[ffn].FFN.Form), i, end})
		default:
			return nil, fmt.Errorf("operations %d to %d have neither a mixer nor an FFN", i, end-1)
		}
		i = end
	}
	return spans, nil
}

// ffnStep is the step an FFN operation of form f is.
func ffnStep(f program.FFNForm) layerStep {
	switch f {
	case program.FfnNonGated:
		return stepNonGatedFFN
	case program.FfnMoE, program.FfnGptOss, program.FfnLlama4MoE, program.FfnNemotronMoE:
		return stepMoE
	}
	return stepDenseFFN
}

// checkStepClaims reports an error unless spans consume ops exactly once, in order, and each span holds exactly one of its step's
// core operations and otherwise only operations its block emits: a mixer step its input norm, its mixer's operations, a
// post-attention norm and its residual add; an FFN step its input norm (the layer's first, on an FFN-only layer), the FFN, a
// post-MLP norm and the residual add.
func checkStepClaims(ops []program.Op, spans []stepSpan) error {
	at := 0
	for _, s := range spans {
		if s.from != at || s.to <= s.from || s.to > len(ops) {
			return fmt.Errorf("the %v step consumes operations %d to %d, want from %d", s.step, s.from, s.to-1, at)
		}
		at = s.to
		core := 0
		for j := s.from; j < s.to; j++ {
			o := &ops[j]
			isCore, claimed := stepClaims(s, o)
			if !claimed {
				return fmt.Errorf("the %v step consumes operation %d (%v), which its block does not emit", s.step, j, o.Kind)
			}
			if isCore {
				core++
			}
		}
		if core != 1 {
			return fmt.Errorf("the %v step over operations %d to %d holds %d of its core operations, want 1", s.step, s.from, s.to-1, core)
		}
	}
	if at != len(ops) {
		return fmt.Errorf("operations %d to %d are consumed by no step", at, len(ops)-1)
	}
	return nil
}

// stepClaims reports whether operation o of a layer is the core of span s's step, and whether the step's block emits it at all.
func stepClaims(s stepSpan, o *program.Op) (core, claimed bool) {
	slot := func(slots ...program.NormSlot) bool {
		return o.Kind == program.OpNorm && slices.Contains(slots, o.Norm.Slot)
	}
	residual := o.Kind == program.OpResidual && !o.Res.Two
	switch s.step {
	case stepAttention:
		switch o.Kind {
		case program.OpAttention, program.OpGatedAttn, program.OpGemma4Attn:
			return true, true
		case program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpOutGate, program.OpProjO:
			return false, true
		}
		return false, slot(program.SlotPreAttn, program.SlotPostAttn) || residual
	case stepDeltaNet, stepMamba, stepMLA:
		core := map[layerStep]program.OpKind{stepDeltaNet: program.OpDeltaNet, stepMamba: program.OpMamba2, stepMLA: program.OpMLA}[s.step]
		if o.Kind == core {
			return true, true
		}
		return false, slot(program.SlotPreAttn) || residual
	case stepMoE, stepNonGatedFFN, stepDenseFFN:
		if o.Kind == program.OpFFN {
			return true, ffnStep(o.FFN.Form) == s.step
		}
		return false, slot(program.SlotPreMLP, program.SlotPostMLP) || (s.from == 0 && slot(program.SlotPreAttn)) || o.Kind == program.OpResidual
	case stepGemma4MoE:
		return o.Kind == program.OpGemma4MoE, o.Kind == program.OpGemma4MoE
	case stepPLE:
		return o.Kind == program.OpPLE, o.Kind == program.OpPLE
	case stepLayerScale:
		return o.Kind == program.OpLayerScale, o.Kind == program.OpLayerScale
	}
	return false, false
}

// buildLayerSteps sets every layer's steps from p, and declines when a layer's operations match no step, a step consumes an
// operation its block does not emit, or the steps differ from the blocks the layer's build loaded (its mixer, FFN form, PLE and
// layer scale). A layer scale step on a layer whose scalar is 1 or absent stays: its block then emits nothing.
func (r *resident) buildLayerSteps(p *program.Program) error {
	if len(p.Layers) != len(r.layers) {
		return fmt.Errorf("the program has %d layers and the resident %d", len(p.Layers), len(r.layers))
	}
	for l, ops := range p.Layers {
		spans, err := layerStepsOf(ops)
		if err == nil {
			err = checkStepClaims(ops, spans)
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
		L := &r.layers[l]
		steps := make([]layerStep, len(spans))
		for i, s := range spans {
			steps[i] = s.step
		}
		if want := r.loadedSteps(L); !slices.Equal(slices.DeleteFunc(slices.Clone(steps), func(s layerStep) bool {
			return s == stepLayerScale && L.uLayerScalar == (Buffer{})
		}), want) {
			return fmt.Errorf("layer %d: the program's steps are %v and the layer's build loaded %v", l, steps, want)
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
	if L.uLayerScalar != (Buffer{}) {
		s = append(s, stepLayerScale)
	}
	return s
}
