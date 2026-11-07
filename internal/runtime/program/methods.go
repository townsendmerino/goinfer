package program

import (
	"fmt"
	"math"
	"slices"
)

// The methods and variables of program.go's types. They execute, and no program fingerprint sees their bodies, so this file
// stays in every parity key while program.go, which only declares, is a builder (paritykey.ProgramTypesFile).

// Recurrent reports whether the operation mutates per-layer state in place, which no positional rewind restores.
func (k OpKind) Recurrent() bool {
	return k == OpKDA || k == OpShortConv || k == OpMamba2 || k == OpDeltaNet
}

// GenericKinds are the operations of the generic layer: the ones the batched and multi-sequence walkers implement, LoRA's
// deltas reach (q/k/v, o and the gated FFN), and the layer pager's spans cover.
var GenericKinds = []OpKind{OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpAttention, OpOutGate, OpProjO, OpFFN, OpResidual}

func (k OpKind) String() string {
	names := [...]string{OpNorm: "norm", OpProjQKV: "proj-qkv", OpQKNorm: "qk-norm", OpRope: "rope", OpAttnTemp: "attn-temp",
		OpAttention: "attention", OpOutGate: "out-gate", OpProjO: "proj-o", OpFFN: "ffn", OpResidual: "residual", OpMLA: "mla", OpKDA: "kda", OpShortConv: "short-conv", OpMamba2: "mamba2", OpDeltaNet: "deltanet", OpGatedAttn: "gated-attn", OpGemma4Attn: "gemma4-attn", OpGemma4MoE: "gemma4-moe", OpPLE: "ple",
		OpLayerScale: "layer-scale"}
	if int(k) < len(names) && names[k] != "" {
		return names[k]
	}
	return fmt.Sprintf("op(%d)", uint8(k))
}

// SubCaptureUncovered reports whether a layer of p has an operation that changes the residual in place without a captured
// contribution: Gemma 4's PLE branch (E-models) and its parallel dense ‖ MoE block (26B). Decoder's ForwardSubCapture refuses
// such a model, since its contributions do not account for the layer (docs/tasks/task-forward-architecture-2026-10.md §4.8, the
// seams PR).
func (p *Program) SubCaptureUncovered() bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpPLE || ops[i].Kind == OpGemma4MoE {
				return true
			}
		}
	}
	return false
}

// Start is the first key position a query at pos attends: past the window, and inside the query's chunk.
func (p AttentionParams) Start(pos int) int {
	s := 0
	if p.Window > 0 {
		s = max(pos-p.Window+1, 0)
	}
	if p.Chunk > 0 {
		s = max(s, (pos/p.Chunk)*p.Chunk)
	}
	return s
}

// Scale is the query's temperature at pos.
func (p *AttnTempParams) Scale(pos int) float32 {
	return float32(1 + p.Beta*math.Log1p(math.Floor(float64(pos+p.PosOffset)/p.OrigMaxPos)))
}

// AnySoftcap reports whether any attention operation caps its scores.
func (p *Program) AnySoftcap() bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpAttention && ops[i].Attn.Softcap != 0 {
				return true
			}
		}
	}
	return false
}

// MultiStepEligible is the program-level scope of the multi-sequence walker: the batched attention kernel on every layer, every
// layer pre-norm placed (no post-sublayer norm, no copy for post-only placement, no parallel block), a gated dense FFN, no output
// gate, and no learned positions.
func (p *Program) MultiStepEligible() error {
	if p.Layers == nil {
		return fmt.Errorf("no layer program (an own loop)")
	}
	if !p.AttnOrderIs(AttnBatchedAcc64) || !p.KindsIn(GenericKinds...) {
		return fmt.Errorf("an attention kernel or an operation this walker does not run")
	}
	if p.Embed.LearnedPos {
		return fmt.Errorf("learned positions")
	}
	for l, ops := range p.Layers {
		for i := range ops {
			o := &ops[i]
			switch {
			case o.Kind == OpNorm && (o.Norm.Slot == SlotPostAttn || o.Norm.Slot == SlotPostMLP || o.Norm.Form == NormCopy):
				return fmt.Errorf("layer %d: a post-sublayer or post-only norm", l)
			case o.Kind == OpResidual && o.Res.Two:
				return fmt.Errorf("layer %d: a parallel block", l)
			case o.Kind == OpFFN && o.FFN.Form != FfnGated:
				return fmt.Errorf("layer %d: not a gated dense FFN", l)
			case o.Kind == OpOutGate:
				return fmt.Errorf("layer %d: an attention output gate", l)
			}
		}
	}
	return nil
}

// BatchEligible reports whether the batched walker runs prog: a layer program with no learned positions, the batched attention
// kernel on every layer, and a gated or MoE FFN.
func (p *Program) BatchEligible() bool {
	if p.Layers == nil || p.Embed.LearnedPos || !p.AttnOrderIs(AttnBatchedAcc64) || !p.KindsIn(GenericKinds...) {
		return false
	}
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpFFN && ops[i].FFN.Form != FfnGated && ops[i].FFN.Form != FfnMoE {
				return false
			}
		}
	}
	return true
}

// AnyOp reports whether any operation in p satisfies f.
func (p *Program) AnyOp(f func(o *Op) bool) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if f(&ops[i]) {
				return true
			}
		}
	}
	return false
}

// KindsIn reports whether every operation in p is one of kinds.
func (p *Program) KindsIn(kinds ...OpKind) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if !slices.Contains(kinds, ops[i].Kind) {
				return false
			}
		}
	}
	return true
}

// AttnOrderIs reports whether every attention operation in p computes in order o.
func (p *Program) AttnOrderIs(o AttnOrder) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpAttention && ops[i].Attn.Order != o {
				return false
			}
		}
	}
	return true
}

// String renders the activation for the capability matrix / logs. The gated vs
// non-gated distinction is rendered by the matrix from Architecture.NonGatedMLP.
func (a ActKind) String() string {
	switch a {
	case ActGeluTanh:
		return "GELU-tanh"
	case ActSiLU:
		return "SiLU"
	case ActReLU2:
		return "ReLU²"
	case ActGelu:
		return "GELU"
	default:
		return "unknown"
	}
}
