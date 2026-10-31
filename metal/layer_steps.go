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

// stepSpan is a step and the operations of its layer it consumes, ops[from:to]. parallel marks the two halves of a parallel
// block's one sublayer: the attention step up to its FFN, and the dense FFN step through the residual add of both outputs.
type stepSpan struct {
	step     layerStep
	from, to int
	parallel bool
}

// stepPlantForTest makes layerStepsOf's attention step also consume the operation after its residual add (the FFN sublayer's
// input norm), so OP4m-b2's M2 can show the claim check goes red. False in production.
var stepPlantForTest bool

// layerStepsOf matches one layer's operations against the fusion table. A sublayer runs to its residual add; one with a mixer is
// that mixer's step, one with an FFN the FFN form's step, and a parallel block's single sublayer (attention and a gated dense
// FFN, whose residual adds both outputs) is the attention step up to its FFN and the dense FFN step from there. Gemma 4's MoE
// block, PLE and layer scale are steps of one operation. An operation or FFN form no entry matches is an error naming it.
func layerStepsOf(ops []program.Op) ([]stepSpan, error) {
	var spans []stepSpan
	for i := 0; i < len(ops); {
		switch ops[i].Kind {
		case program.OpGemma4MoE, program.OpPLE, program.OpLayerScale:
			spans = append(spans, stepSpan{step: map[program.OpKind]layerStep{program.OpGemma4MoE: stepGemma4MoE, program.OpPLE: stepPLE,
				program.OpLayerScale: stepLayerScale}[ops[i].Kind], from: i, to: i + 1})
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
			case program.OpNorm, program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpProjO,
				program.OpResidual:
			default:
				return nil, fmt.Errorf("operation %d (%v) has no step on Metal", j, k)
			}
		}
		var fs layerStep
		if ffn >= 0 {
			var ok bool
			if fs, ok = ffnStep(ops[ffn].FFN.Form); !ok {
				return nil, fmt.Errorf("operation %d (ffn) is the FFN form %v, which has no step on Metal", ffn, ops[ffn].FFN.Form)
			}
		}
		two := ops[end-1].Res.Two
		switch {
		case mixer != 0 && ffn >= 0:
			if mixer != stepAttention || fs != stepDenseFFN || !two {
				return nil, fmt.Errorf("operations %d to %d hold a %v mixer and a %v step in one sublayer (a residual of two outputs: %v); only a parallel block (attention, a dense FFN, a residual of both) is one sublayer",
					i, end-1, mixer, fs, two)
			}
			spans = append(spans, stepSpan{mixer, i, ffn, true}, stepSpan{fs, ffn, end, true})
		case two:
			return nil, fmt.Errorf("operation %d adds two outputs to the residual, and operations %d to %d are not a parallel block", end-1, i, end-1)
		case mixer != 0:
			if stepPlantForTest && mixer == stepAttention && end < len(ops) {
				end++
			}
			spans = append(spans, stepSpan{step: mixer, from: i, to: end})
		case ffn >= 0:
			spans = append(spans, stepSpan{step: fs, from: i, to: end})
		default:
			return nil, fmt.Errorf("operations %d to %d have neither a mixer nor an FFN", i, end-1)
		}
		i = end
	}
	return spans, nil
}

// ffnStep is the step an FFN operation of form f is, and false for a form with no Metal block.
func ffnStep(f program.FFNForm) (layerStep, bool) {
	switch f {
	case program.FfnNonGated:
		return stepNonGatedFFN, true
	case program.FfnMoE, program.FfnGptOss:
		return stepMoE, true
	case program.FfnGated, program.FfnGemma4Dense:
		return stepDenseFFN, true
	}
	return 0, false
}

// stepForms is the FFN forms each FFN step's block computes. checkStepClaims reads it rather than ffnStep, so a form ffnStep
// sends to the wrong step goes red.
var stepForms = map[layerStep][]program.FFNForm{
	stepMoE:         {program.FfnMoE, program.FfnGptOss},
	stepNonGatedFFN: {program.FfnNonGated},
	stepDenseFFN:    {program.FfnGated, program.FfnGemma4Dense},
}

// checkStepClaims reports an error unless spans consume ops exactly once, in order, and each span holds exactly one of its step's
// core operations and otherwise only operations its block emits: a mixer step its input norm, its mixer's operations, a
// post-attention norm and its residual add; an FFN step its input norm (the layer's first, on an FFN-only layer), the FFN, a
// post-MLP norm and the residual add. A sublayer's span ends with its residual add, of one output, or of two for a parallel
// block, whose attention half ends before its FFN and is followed by the dense FFN half.
func checkStepClaims(ops []program.Op, spans []stepSpan) error {
	at := 0
	for k, s := range spans {
		if s.from != at || s.to <= s.from || s.to > len(ops) {
			return fmt.Errorf("the %v step consumes operations %d to %d, want from %d", s.step, s.from, s.to-1, at)
		}
		at = s.to
		switch last := &ops[s.to-1]; {
		case s.parallel && s.step != stepAttention && s.step != stepDenseFFN:
			return fmt.Errorf("the %v step over operations %d to %d is marked as a parallel block's half", s.step, s.from, s.to-1)
		case s.step == stepGemma4MoE || s.step == stepPLE || s.step == stepLayerScale:
		case s.parallel && s.step == stepAttention:
			if k+1 == len(spans) || !spans[k+1].parallel || spans[k+1].step != stepDenseFFN {
				return fmt.Errorf("the parallel block's attention step over operations %d to %d is not followed by its dense FFN step", s.from, s.to-1)
			}
		case s.parallel && (k == 0 || !spans[k-1].parallel || spans[k-1].step != stepAttention):
			return fmt.Errorf("the parallel block's dense FFN step over operations %d to %d does not follow its attention step", s.from, s.to-1)
		case last.Kind != program.OpResidual || last.Res.Two != s.parallel:
			return fmt.Errorf("the %v step over operations %d to %d ends with operation %d (%v), not its sublayer's residual add (of two outputs: %v)",
				s.step, s.from, s.to-1, s.to-1, last.Kind, s.parallel)
		}
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
	residual := o.Kind == program.OpResidual && o.Res.Two == s.parallel && !(s.parallel && s.step == stepAttention)
	switch s.step {
	case stepAttention:
		switch o.Kind {
		case program.OpAttention, program.OpGatedAttn, program.OpGemma4Attn:
			return true, true
		case program.OpProjQKV, program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpProjO:
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
			return true, slices.Contains(stepForms[s.step], o.FFN.Form)
		}
		return false, slot(program.SlotPreMLP, program.SlotPostMLP) || (s.from == 0 && slot(program.SlotPreAttn)) || residual
	case stepGemma4MoE:
		return o.Kind == program.OpGemma4MoE, o.Kind == program.OpGemma4MoE
	case stepPLE:
		return o.Kind == program.OpPLE, o.Kind == program.OpPLE
	case stepLayerScale:
		return o.Kind == program.OpLayerScale, o.Kind == program.OpLayerScale
	}
	return false, false
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
			steps[i] = s.step
		}
		if want := r.loadedSteps(L); !slices.Equal(steps, want) {
			return fmt.Errorf("layer %d: the program's steps are %v and the layer's build loaded %v", l, steps, want)
		}
		if r.q4kLane && (!slices.Equal(steps, []layerStep{stepAttention, stepDenseFFN}) || spans[0].parallel) {
			return fmt.Errorf("layer %d: the q4k lane runs a sequential attention and dense FFN step only, and the layer's steps are %v (a parallel block: %v)",
				l, steps, spans[0].parallel)
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
			if !slices.Contains(prefillBlocks, s.step) {
				return fmt.Sprintf("layer %d: the batched prefill has no block for the %v step", l, s.step)
			}
		}
	}
	return ""
}
