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
	ResidualScale  bool // a residual add with a Scale other than 0 or 1
	DeltaNegEigval bool // DeltaNet's write gate widened to [0, 2)
	DeltaONormEps  bool // DeltaNet's output norm at an epsilon of its own
}

// Check returns an error naming the first layer, operation and parameter of p outside s, or nil.
func (s *Support) Check(p *Program) error {
	for l, ops := range p.Layers {
		if err := s.CheckLayer(ops); err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
	}
	return nil
}

// Gaps is every distinct way p falls outside s, each named at its first layer and operation, several on one operation
// included: an admission that reports only the first would hide the rest, and lifting one leaves the others.
func (s *Support) Gaps(p *Program) []string {
	var out []string
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
}

func (f FFNForm) String() string {
	names := [...]string{FfnGated: "gated", FfnNonGated: "non-gated", FfnMoE: "MoE", FfnGptOss: "gpt-oss MoE", FfnLlama4MoE: "Llama 4 MoE",
		FfnNemotronMoE: "Nemotron MoE", FfnGemma4Dense: "Gemma 4 dense"}
	if int(f) < len(names) && names[f] != "" {
		return names[f]
	}
	return fmt.Sprintf("ffn-form(%d)", uint8(f))
}
