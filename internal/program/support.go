package program

import (
	"fmt"
	"slices"
)

// Support is one backend's operation-and-parameter table: the operation kinds it has a step for, and the parameter values its
// blocks implement. The backend's walker checks a program against it at build, and decoder's admission reads the same table, so
// a decline names the operation and parameter. A parameter the table does not name is implemented at every value. A tier-0 test
// per backend holds the table to its walker (docs/tasks/task-forward-architecture-2026-10.md §4.9, OP4m-c).
type Support struct {
	Backend string
	Kinds   []OpKind
	FFN     map[FFNForm][]ActKind // each FFN form with a block, and the activations that block computes
	QKNorms []QKNormForm          // an opQKNorm's, and the one inside opGatedAttn
	Ropes   []RopeKernel          // an opRope's, and the one inside opGatedAttn

	PairwiseRope   bool // GPT-J pairwise rotation (dims 2d, 2d+1), m-RoPE sections included, beside the NeoX half-split
	AttnChunk      bool // an attention with Chunk set
	AttnWindow     bool // an attention with Window set
	AttnSoftcap    bool // an attention with Softcap set
	QKNormWhole    bool // a QK-norm with one statistic over all heads
	ResidualScale  bool // a residual add with a Scale other than 0 or 1
	DeltaNegEigval bool // DeltaNet's write gate widened to [0, 2)
	DeltaONormEps  bool // DeltaNet's output norm at an epsilon of its own

	// The placements and forms of the generic layer, and the embedding stage.
	SandwichNorm  bool // a norm on a sublayer's output (the post-attention or post-MLP slot)
	PostOnlyNorm  bool // an input norm of form NormCopy: the sublayer reads the raw residual
	ParallelBlock bool // a residual add of two sublayers' outputs
	LayerNorm     bool // a norm of form NormLayer, the head's included
	OutBias       bool // an output projection with a bias
	LearnedPos    bool // learned position embeddings
}

// Check returns an error naming the first layer, operation and parameter of p outside s, or nil.
func (s *Support) Check(p *Program) error {
	if g := s.stageGaps(p); len(g) > 0 {
		return fmt.Errorf("%s", g[0])
	}
	for l, ops := range p.Layers {
		if err := s.CheckLayer(ops); err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
	}
	return nil
}

// stageGaps is every way p's embedding and head stages fall outside s.
func (s *Support) stageGaps(p *Program) []string {
	var out []string
	if p.Embed.LearnedPos && !s.LearnedPos {
		out = append(out, "the embedding stage: "+s.Backend+" does not implement learned position embeddings")
	}
	if p.Head.Norm.Form == NormLayer && !s.LayerNorm {
		out = append(out, "the head: "+s.Backend+" does not implement LayerNorm")
	}
	return out
}

// Gaps is every distinct way p falls outside s, each named at its first layer and operation, several on one operation
// included: an admission that reports only the first would hide the rest, and lifting one leaves the others.
func (s *Support) Gaps(p *Program) []string {
	out := s.stageGaps(p)
	seen := map[string]bool{}
	for l, ops := range p.Layers {
		for i := range ops {
			for _, why := range s.gaps(&ops[i]) {
				if key := ops[i].Kind.String() + why; !seen[key] {
					seen[key] = true
					out = append(out, fmt.Sprintf("layer %d: operation %d (%v): %s %s", l, i, ops[i].Kind, s.Backend, why))
				}
			}
		}
	}
	return out
}

// CheckLayer is Check for one layer's operations.
func (s *Support) CheckLayer(ops []Op) error {
	for i := range ops {
		if g := s.gaps(&ops[i]); len(g) > 0 {
			return fmt.Errorf("operation %d (%v): %s %s", i, ops[i].Kind, s.Backend, g[0])
		}
	}
	return nil
}

// HasStep reports whether s has a step for o: its kind, and for an FFN its form. The parameters Check also reads are not a
// step's choice.
func (s *Support) HasStep(o *Op) bool {
	if !slices.Contains(s.Kinds, o.Kind) {
		return false
	}
	_, ok := s.FFN[o.FFN.Form]
	return o.Kind != OpFFN || ok
}

// gaps is every way s does not implement o, in a fixed order; none when it does.
func (s *Support) gaps(o *Op) []string {
	if !s.HasStep(o) {
		if o.Kind == OpFFN {
			return []string{fmt.Sprintf("has no block for the FFN form %v", o.FFN.Form)}
		}
		return []string{"has no step for it"}
	}
	qk, rope, attn := o.QKNorm.Form, &o.Rope, &o.Attn
	if o.Kind == OpGatedAttn {
		qk, rope, attn = o.Gated.QKNorm.Form, &o.Gated.Rope, &o.Gated.Attn
	}
	qkOp, ropeOp, attnOp := o.Kind == OpQKNorm || o.Kind == OpGatedAttn, o.Kind == OpRope || o.Kind == OpGatedAttn,
		o.Kind == OpAttention || o.Kind == OpGatedAttn
	var out []string
	add := func(gap bool, format string, args ...any) {
		if gap {
			out = append(out, fmt.Sprintf(format, args...))
		}
	}
	add(o.Kind == OpFFN && !slices.Contains(s.FFN[o.FFN.Form], o.FFN.Act), "has no %v activation in its %v block", o.FFN.Act, o.FFN.Form)
	add(qkOp && !slices.Contains(s.QKNorms, qk), "does not implement the %s QK-norm",
		map[QKNormForm]string{QKRMS: "weighted RMS", QKL2: "parameter-free L2"}[qk])
	add(ropeOp && !slices.Contains(s.Ropes, rope.Kernel), "does not implement the %s rotation",
		map[RopeKernel]string{RopeAtKernel: "ropeAt", RopeMLAKernel: "de-interleaving (mlaRope)"}[rope.Kernel])
	add(ropeOp && rope.Interleave && !s.PairwiseRope, "does not implement GPT-J pairwise rotation")
	add(attnOp && attn.Chunk != 0 && !s.AttnChunk, "does not implement chunked attention")
	add(attnOp && attn.Window != 0 && !s.AttnWindow, "does not implement a windowed attention (%d)", attn.Window)
	add(attnOp && attn.Softcap != 0 && !s.AttnSoftcap, "does not implement an attention softcap (%g)", attn.Softcap)
	add(qkOp && (o.QKNorm.Whole || o.Gated.QKNorm.Whole) && !s.QKNormWhole, "does not implement a whole-vector QK-norm")
	add(o.Kind == OpNorm && (o.Norm.Slot == SlotPostAttn || o.Norm.Slot == SlotPostMLP) && !s.SandwichNorm,
		"does not implement a norm on a sublayer's output")
	add(o.Kind == OpNorm && o.Norm.Form == NormCopy && !s.PostOnlyNorm, "does not implement post-only placement (no input norm)")
	add(o.Kind == OpNorm && o.Norm.Form == NormLayer && !s.LayerNorm, "does not implement LayerNorm")
	add(o.Kind == OpResidual && o.Res.Two && !s.ParallelBlock, "does not implement a parallel block's residual of two outputs")
	add(o.Kind == OpProjO && o.Proj.Bias && !s.OutBias, "does not implement an output projection bias")
	add(o.Kind == OpResidual && o.Res.Scale != 0 && o.Res.Scale != 1 && !s.ResidualScale, "does not implement a residual scale (%.4g)", o.Res.Scale)
	add(o.Kind == OpDeltaNet && o.Delta.NegEigval && !s.DeltaNegEigval, "does not implement DeltaNet's widened write gate (NegEigval)")
	add(o.Kind == OpDeltaNet && o.Delta.ONormEps != 0 && o.Delta.ONormEps != o.Delta.Eps && !s.DeltaONormEps,
		"does not implement DeltaNet's own output-norm epsilon (%g, beside %g)", o.Delta.ONormEps, o.Delta.Eps)
	return out
}

// Metal is the Metal resident's table. Its walker is metal's layerStepsOf.
var Metal = &Support{
	Backend: "metal",
	Kinds: []OpKind{OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpAttention, OpProjO, OpFFN, OpResidual, OpMLA, OpMamba2,
		OpDeltaNet, OpGatedAttn, OpGemma4Attn, OpGemma4MoE, OpPLE, OpLayerScale},
	FFN: map[FFNForm][]ActKind{
		FfnGated:       {ActSiLU, ActGeluTanh},
		FfnNonGated:    {ActGeluTanh, ActSiLU, ActReLU2},
		FfnMoE:         {ActSiLU, ActGeluTanh},
		FfnGptOss:      {ActSiLU, ActGeluTanh},
		FfnGemma4Dense: {ActSiLU, ActGeluTanh},
	},
	QKNorms:      []QKNormForm{QKRMS},
	Ropes:        []RopeKernel{RopeAtKernel},
	PairwiseRope: true,
	AttnWindow:   true, AttnSoftcap: true, QKNormWhole: true,
	SandwichNorm: true, PostOnlyNorm: true, ParallelBlock: true, LayerNorm: true, OutBias: true, LearnedPos: true,
}

// MetalStep is the table of Metal's multi-sequence step (metal/batch.go, MC3's batched decode): one layer shape, pre-norm
// attention and a gated dense FFN, with the per-head QK-norm, the query temperature and every rotation but pairwise. Its
// other limits (one KV allocation per layer, uniform attention geometry, tiled widths) are checked by the step, not here.
var MetalStep = &Support{
	Backend: "metal's batched step",
	Kinds:   []OpKind{OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpAttention, OpProjO, OpFFN, OpResidual},
	FFN:     map[FFNForm][]ActKind{FfnGated: {ActSiLU, ActGeluTanh}},
	QKNorms: []QKNormForm{QKRMS},
	Ropes:   []RopeKernel{RopeAtKernel},
}

func (f FFNForm) String() string {
	names := [...]string{FfnGated: "gated", FfnNonGated: "non-gated", FfnMoE: "MoE", FfnGptOss: "gpt-oss MoE", FfnLlama4MoE: "Llama 4 MoE",
		FfnNemotronMoE: "Nemotron MoE", FfnGemma4Dense: "Gemma 4 dense"}
	if int(f) < len(names) && names[f] != "" {
		return names[f]
	}
	return fmt.Sprintf("ffn-form(%d)", uint8(f))
}
