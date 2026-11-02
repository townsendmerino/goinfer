package program

import (
	"fmt"
	"slices"
)

// Sublayer is one sublayer kind of a layer's operations, as a resident's walker runs it (OP4m-b2 and OP4cu-b2,
// docs/tasks/task-forward-architecture-2026-10.md §4.9 and §4.9a). A layer's sublayers run in its program's order; each backend
// maps them to its own blocks.
type Sublayer uint8

const (
	SublayerAttention   Sublayer = iota + 1 // input norm through the output projection, any post-attention norm, the residual add
	SublayerDeltaNet                         // the Gated DeltaNet mixer sublayer
	SublayerMamba                            // the Mamba-2 mixer sublayer
	SublayerMLA                              // the Multi-head Latent Attention sublayer
	SublayerGemma4MoE                        // Gemma 4's parallel dense and MoE block
	SublayerMoE                              // a MoE FFN sublayer
	SublayerNonGatedFFN                      // a non-gated dense FFN sublayer
	SublayerDenseFFN                         // a gated dense FFN sublayer, or a parallel block's FFN half
	SublayerPLE                              // Gemma 4's per-layer-embedding branch
	SublayerLayerScale                       // Gemma 4's per-layer output scalar
)

var sublayerNames = [...]string{SublayerAttention: "attention", SublayerDeltaNet: "DeltaNet", SublayerMamba: "Mamba-2", SublayerMLA: "MLA",
	SublayerGemma4MoE: "Gemma 4 MoE block", SublayerMoE: "MoE FFN", SublayerNonGatedFFN: "non-gated FFN", SublayerDenseFFN: "dense FFN",
	SublayerPLE: "PLE", SublayerLayerScale: "layer scale"}

func (s Sublayer) String() string {
	if int(s) < len(sublayerNames) && sublayerNames[s] != "" {
		return sublayerNames[s]
	}
	return fmt.Sprintf("Sublayer(%d)", uint8(s))
}

// SublayerSpan is a step and the operations of its layer it consumes, ops[from:to]. parallel marks the two halves of a parallel
// block's one sublayer: the attention step up to its FFN, and the dense FFN step through the residual add of both outputs.
type SublayerSpan struct {
	Kind     Sublayer
	From, To int
	Parallel bool
}

// SublayersOf matches one layer's operations against the fusion table. A sublayer runs to its residual add; one with a mixer is
// that mixer's step, one with an FFN the FFN form's step, and a parallel block's single sublayer (attention and a gated dense
// FFN, whose residual adds both outputs) is the attention step up to its FFN and the dense FFN step from there. Gemma 4's MoE
// block, PLE and layer scale are steps of one operation. An operation or FFN form no entry matches is an error naming it.
func SublayersOf(ops []Op, backend string) ([]SublayerSpan, error) {
	var spans []SublayerSpan
	for i := 0; i < len(ops); {
		switch ops[i].Kind {
		case OpGemma4MoE, OpPLE, OpLayerScale:
			spans = append(spans, SublayerSpan{Kind: map[OpKind]Sublayer{OpGemma4MoE: SublayerGemma4MoE, OpPLE: SublayerPLE,
				OpLayerScale: SublayerLayerScale}[ops[i].Kind], From: i, To: i + 1})
			i++
			continue
		}
		end := i
		for end < len(ops) && ops[end].Kind != OpResidual {
			end++
		}
		if end == len(ops) {
			return nil, fmt.Errorf("operations %d to %d end without a residual add, and no step ends there", i, end-1)
		}
		end++
		mixer, ffn := Sublayer(0), -1
		for j := i; j < end; j++ {
			switch k := ops[j].Kind; k {
			case OpAttention, OpGatedAttn, OpGemma4Attn:
				mixer = SublayerAttention
			case OpDeltaNet:
				mixer = SublayerDeltaNet
			case OpMamba2:
				mixer = SublayerMamba
			case OpMLA:
				mixer = SublayerMLA
			case OpFFN:
				ffn = j
			case OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpProjO,
				OpResidual:
			default:
				return nil, fmt.Errorf("operation %d (%v) has no step on %s", j, k, backend)
			}
		}
		var fs Sublayer
		if ffn >= 0 {
			var ok bool
			if fs, ok = ffnSublayer(ops[ffn].FFN.Form); !ok {
				return nil, fmt.Errorf("operation %d (ffn) is the FFN form %v, which has no step on %s", ffn, ops[ffn].FFN.Form, backend)
			}
		}
		two := ops[end-1].Res.Two
		switch {
		case mixer != 0 && ffn >= 0:
			if mixer != SublayerAttention || fs != SublayerDenseFFN || !two {
				return nil, fmt.Errorf("operations %d to %d hold a %v mixer and a %v step in one sublayer (a residual of two outputs: %v); only a parallel block (attention, a dense FFN, a residual of both) is one sublayer",
					i, end-1, mixer, fs, two)
			}
			spans = append(spans, SublayerSpan{Kind: mixer, From: i, To: ffn, Parallel: true}, SublayerSpan{Kind: fs, From: ffn, To: end, Parallel: true})
		case two:
			return nil, fmt.Errorf("operation %d adds two outputs to the residual, and operations %d to %d are not a parallel block", end-1, i, end-1)
		case mixer != 0:
			if shapePlant == ShapePlantClaimOneMore && mixer == SublayerAttention && end < len(ops) {
				end++
			}
			spans = append(spans, SublayerSpan{Kind: mixer, From: i, To: end})
		case ffn >= 0:
			spans = append(spans, SublayerSpan{Kind: fs, From: i, To: end})
		default:
			return nil, fmt.Errorf("operations %d to %d have neither a mixer nor an FFN", i, end-1)
		}
		i = end
	}
	return spans, nil
}

// ffnSublayer is the sublayer an FFN operation of form f is, and false for a form no sublayer kind computes.
func ffnSublayer(f FFNForm) (Sublayer, bool) {
	switch f {
	case FfnNonGated:
		return SublayerNonGatedFFN, true
	case FfnMoE, FfnGptOss:
		return SublayerMoE, true
	case FfnGated, FfnGemma4Dense:
		return SublayerDenseFFN, true
	}
	return 0, false
}

// sublayerForms is the FFN forms each FFN step's block computes. CheckSublayerClaims reads it rather than ffnSublayer, so a form ffnSublayer
// sends to the wrong step goes red.
var sublayerForms = map[Sublayer][]FFNForm{
	SublayerMoE:         {FfnMoE, FfnGptOss},
	SublayerNonGatedFFN: {FfnNonGated},
	SublayerDenseFFN:    {FfnGated, FfnGemma4Dense},
}

// CheckSublayerClaims reports an error unless spans consume ops exactly once, in order, and each span holds exactly one of its step's
// core operations and otherwise only operations its block emits: a mixer step its input norm, its mixer's operations, a
// post-attention norm and its residual add; an FFN step its input norm (the layer's first, on an FFN-only layer), the FFN, a
// post-MLP norm and the residual add. A sublayer's span ends with its residual add, of one output, or of two for a parallel
// block, whose attention half ends before its FFN and is followed by the dense FFN half.
func CheckSublayerClaims(ops []Op, spans []SublayerSpan) error {
	at := 0
	for k, s := range spans {
		if s.From != at || s.To <= s.From || s.To > len(ops) {
			return fmt.Errorf("the %v step consumes operations %d to %d, want from %d", s.Kind, s.From, s.To-1, at)
		}
		at = s.To
		switch last := &ops[s.To-1]; {
		case s.Parallel && s.Kind != SublayerAttention && s.Kind != SublayerDenseFFN:
			return fmt.Errorf("the %v step over operations %d to %d is marked as a parallel block's half", s.Kind, s.From, s.To-1)
		case s.Kind == SublayerGemma4MoE || s.Kind == SublayerPLE || s.Kind == SublayerLayerScale:
		case s.Parallel && s.Kind == SublayerAttention:
			if k+1 == len(spans) || !spans[k+1].Parallel || spans[k+1].Kind != SublayerDenseFFN {
				return fmt.Errorf("the parallel block's attention step over operations %d to %d is not followed by its dense FFN step", s.From, s.To-1)
			}
		case s.Parallel && (k == 0 || !spans[k-1].Parallel || spans[k-1].Kind != SublayerAttention):
			return fmt.Errorf("the parallel block's dense FFN step over operations %d to %d does not follow its attention step", s.From, s.To-1)
		case last.Kind != OpResidual || last.Res.Two != s.Parallel:
			return fmt.Errorf("the %v step over operations %d to %d ends with operation %d (%v), not its sublayer's residual add (of two outputs: %v)",
				s.Kind, s.From, s.To-1, s.To-1, last.Kind, s.Parallel)
		}
		core := 0
		for j := s.From; j < s.To; j++ {
			o := &ops[j]
			isCore, claimed := sublayerClaims(s, o)
			if !claimed {
				return fmt.Errorf("the %v step consumes operation %d (%v), which its block does not emit", s.Kind, j, o.Kind)
			}
			if isCore {
				core++
			}
		}
		if core != 1 {
			return fmt.Errorf("the %v step over operations %d to %d holds %d of its core operations, want 1", s.Kind, s.From, s.To-1, core)
		}
	}
	if at != len(ops) {
		return fmt.Errorf("operations %d to %d are consumed by no step", at, len(ops)-1)
	}
	return nil
}

// sublayerClaims reports whether operation o of a layer is the core of span s's step, and whether the step's block emits it at all.
func sublayerClaims(s SublayerSpan, o *Op) (core, claimed bool) {
	slot := func(slots ...NormSlot) bool {
		return o.Kind == OpNorm && slices.Contains(slots, o.Norm.Slot)
	}
	residual := o.Kind == OpResidual && o.Res.Two == s.Parallel && !(s.Parallel && s.Kind == SublayerAttention)
	switch s.Kind {
	case SublayerAttention:
		switch o.Kind {
		case OpAttention, OpGatedAttn, OpGemma4Attn:
			return true, true
		case OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpProjO:
			return false, true
		}
		return false, slot(SlotPreAttn, SlotPostAttn) || residual
	case SublayerDeltaNet, SublayerMamba, SublayerMLA:
		core := map[Sublayer]OpKind{SublayerDeltaNet: OpDeltaNet, SublayerMamba: OpMamba2, SublayerMLA: OpMLA}[s.Kind]
		if o.Kind == core {
			return true, true
		}
		return false, slot(SlotPreAttn) || residual
	case SublayerMoE, SublayerNonGatedFFN, SublayerDenseFFN:
		if o.Kind == OpFFN {
			return true, slices.Contains(sublayerForms[s.Kind], o.FFN.Form)
		}
		return false, slot(SlotPreMLP, SlotPostMLP) || (s.From == 0 && slot(SlotPreAttn)) || residual
	case SublayerGemma4MoE:
		return o.Kind == OpGemma4MoE, o.Kind == OpGemma4MoE
	case SublayerPLE:
		return o.Kind == OpPLE, o.Kind == OpPLE
	case SublayerLayerScale:
		return o.Kind == OpLayerScale, o.Kind == OpLayerScale
	}
	return false, false
}
