package program

import (
	"slices"
	"testing"
)

// The pure tests of SublayersOf and CheckSublayerClaims, moved from metal's resident_memguard_test.go with the derivation
// (OP4cu-b2, docs/tasks/task-forward-architecture-2026-10.md §4.9a); their assertions are unchanged.

// TestSublayersOf_parallelAndForms (Gemini on #544, points 1 and 2): an FFN form with no Metal block declines instead of mapping
// to a step; one sublayer holding a mixer and an FFN is a parallel block only with attention, a dense FFN and a residual of two
// outputs; and a residual of two outputs outside a parallel block declines.
func TestSublayersOf_parallelAndForms(t *testing.T) {
	norm := Op{Kind: OpNorm, Norm: NormParams{Slot: SlotPreAttn, Form: NormRMS}}
	ffn := func(f FFNForm) Op {
		return Op{Kind: OpFFN, FFN: FFNParams{Form: f}}
	}
	res, two := Op{Kind: OpResidual}, Op{Kind: OpResidual, Res: ResidualParams{Two: true}}
	attn := []Op{norm, {Kind: OpProjQKV}, {Kind: OpAttention}, {Kind: OpProjO}}
	cat := func(parts ...[]Op) []Op { return slices.Concat(parts...) }
	for _, c := range []struct {
		name string
		ops  []Op
	}{
		{"Llama 4's MoE", []Op{norm, ffn(FfnLlama4MoE), res}},
		{"Nemotron's MoE", []Op{norm, ffn(FfnNemotronMoE), res}},
		{"an unknown FFN form", []Op{norm, ffn(FfnGemma4Dense + 1), res}},
		{"a DeltaNet mixer and an FFN, two outputs", []Op{norm, {Kind: OpDeltaNet}, ffn(FfnGated), two}},
		{"attention and a MoE FFN, two outputs", cat(attn, []Op{ffn(FfnMoE), two})},
		{"attention and a non-gated FFN, two outputs", cat(attn, []Op{ffn(FfnNonGated), two})},
		{"a post-only layer missing its attention residual", cat(attn, []Op{{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: NormRMS}},
			{Kind: OpNorm, Norm: NormParams{Slot: SlotPreMLP, Form: NormRMS}}, ffn(FfnGated), res})},
		{"attention alone, two outputs", cat(attn, []Op{two})},
		{"an FFN alone, two outputs", []Op{norm, ffn(FfnGated), two}},
	} {
		spans, err := SublayersOf(c.ops, "Metal")
		if err == nil {
			err = CheckSublayerClaims(c.ops, spans)
		}
		if err == nil {
			t.Errorf("%s: steps %v accepted, want a decline", c.name, spans)
		}
	}
	parallel := cat(attn, []Op{ffn(FfnGated), two})
	if spans, err := SublayersOf(parallel, "Metal"); err != nil || CheckSublayerClaims(parallel, spans) != nil {
		t.Errorf("a parallel block (attention, a gated FFN, a residual of both): %v", err)
	}
}

// TestCheckSublayerClaims_spans (Gemini on #544, point 4): CheckSublayerClaims declines a gap, an overlap, an empty span, a span past the
// layer, operations no span consumes, a span with no core operation or two, an FFN step over another step's form, a span not
// ending with its residual add, and a parallel half without its other half.
func TestCheckSublayerClaims_spans(t *testing.T) {
	norm := func(slot NormSlot) Op {
		return Op{Kind: OpNorm, Norm: NormParams{Slot: slot, Form: NormRMS}}
	}
	res := Op{Kind: OpResidual}
	ffn := func(f FFNForm) Op {
		return Op{Kind: OpFFN, FFN: FFNParams{Form: f}}
	}
	// 0 norm, 1 qkv, 2 attention, 3 o, 4 residual, 5 norm, 6 ffn, 7 residual
	ops := []Op{norm(SlotPreAttn), {Kind: OpProjQKV}, {Kind: OpAttention}, {Kind: OpProjO}, res,
		norm(SlotPreMLP), ffn(FfnGated), res}
	good := []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 5}, {Kind: SublayerDenseFFN, From: 5, To: 8}}
	if err := CheckSublayerClaims(ops, good); err != nil {
		t.Fatalf("the control: %v", err)
	}
	twoAttn, noFFN := slices.Clone(ops), slices.Clone(ops)
	twoAttn[3] = Op{Kind: OpAttention}
	noFFN[6] = norm(SlotPostMLP)
	ffnTwo := []Op{norm(SlotPreAttn), ffn(FfnGated), {Kind: OpResidual, Res: ResidualParams{Two: true}}}
	for _, c := range []struct {
		name  string
		ops   []Op
		spans []SublayerSpan
	}{
		{"a gap", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 4}, {Kind: SublayerDenseFFN, From: 5, To: 8}}},
		{"an overlap", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 5}, {Kind: SublayerDenseFFN, From: 4, To: 8}}},
		{"an empty span", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 5}, {Kind: SublayerPLE, From: 5, To: 5}, {Kind: SublayerDenseFFN, From: 5, To: 8}}},
		{"a span past the layer", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 5}, {Kind: SublayerDenseFFN, From: 5, To: 9}}},
		{"operations no span consumes", ops, good[:1]},
		{"no core operation", noFFN, good},
		{"two core operations", twoAttn, good},
		{"an FFN step over another step's form", ops, []SublayerSpan{good[0], {Kind: SublayerMoE, From: 5, To: 8}}},
		{"a span not ending with its residual", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 4}, {Kind: SublayerDenseFFN, From: 4, To: 8}}},
		{"a parallel FFN half over a one-output residual", ops, []SublayerSpan{good[0], {Kind: SublayerDenseFFN, From: 5, To: 8, Parallel: true}}},
		{"a parallel attention half alone", ops, []SublayerSpan{{Kind: SublayerAttention, From: 0, To: 5, Parallel: true}, good[1]}},
		// Gemini on #547: a parallel FFN half with no attention half before it, over the two-output residual it claims, and a
		// step of one operation marked as a parallel half.
		{"a parallel FFN half alone", ffnTwo, []SublayerSpan{{Kind: SublayerDenseFFN, From: 0, To: 3, Parallel: true}}},
		{"a parallel PLE step", []Op{{Kind: OpPLE}}, []SublayerSpan{{Kind: SublayerPLE, From: 0, To: 1, Parallel: true}}},
	} {
		if err := CheckSublayerClaims(c.ops, c.spans); err == nil {
			t.Errorf("%s: accepted, want a decline", c.name)
		}
	}
}
