package program

import (
	"reflect"
	"testing"
)

// TestBuffers_stateIsF32: every per-slot buffer any operation declares is f32, the precision every backend holds and prices state in
// (decoder.stateFloats refuses any other), and every per-position one says which precision it takes.
func TestBuffers_stateIsF32(t *testing.T) {
	var s LayerShape
	s.KVHeads, s.HeadDim, s.MLALatent = 2, 8, 24
	s.DeltaNet.KeyHeads, s.DeltaNet.KeyHeadDim, s.DeltaNet.ValueHeads, s.DeltaNet.ValueHeadDim, s.DeltaNet.ConvKernel = 2, 4, 3, 5, 4
	s.Mamba2.Heads, s.Mamba2.HeadDim, s.Mamba2.DState, s.Mamba2.ConvDim, s.Mamba2.ConvKernel = 2, 3, 4, 20, 4
	s.KDA.Heads, s.KDA.HeadDim, s.KDA.ConvKernel = 2, 4, 3
	s.ShortConv.Dim, s.ShortConv.LCache = 16, 3
	slots := 0
	for k := OpNorm; k <= OpLayerScale; k++ {
		for _, b := range Buffers(Op{Kind: k, G4: Gemma4AttnParams{OwnsKV: true}}, s) {
			switch {
			case b.Unit != PerSlot && b.Unit != PerPosition && b.Unit != PerRow:
				t.Errorf("operation %d declares %q with no unit, which no consumer reads", k, b.Name)
			case b.Unit == PerRow && b.Prec != PrecF32:
				t.Errorf("operation %d declares row %q at precision %d, not f32", k, b.Name, b.Prec)
			case b.Unit == PerRow:
			case b.Unit == PerSlot && b.Prec != PrecF32:
				t.Errorf("operation %d declares state %q at precision %d, not f32", k, b.Name, b.Prec)
			case b.Unit == PerSlot:
				slots++
			case b.Prec != PrecKV && b.Prec != PrecF32:
				t.Errorf("operation %d declares row %q with no precision", k, b.Name)
			}
		}
	}
	if slots != 9 { // DeltaNet's window and state, Mamba-2's, KDA's three windows and state, short conv's window
		t.Errorf("the operation kinds declare %d state buffers, want 9", slots)
	}
}

// eachOperationShape is a layer shape with every width set, each distinct enough that a row sized from the wrong one shows.
func eachOperationShape() LayerShape {
	var s LayerShape
	s.KVHeads, s.HeadDim, s.MLALatent = 2, 8, 24
	s.DeltaNet.KeyHeads, s.DeltaNet.KeyHeadDim, s.DeltaNet.ValueHeads, s.DeltaNet.ValueHeadDim, s.DeltaNet.ConvKernel = 2, 4, 3, 5, 4
	s.Mamba2.Heads, s.Mamba2.HeadDim, s.Mamba2.DState, s.Mamba2.ConvDim, s.Mamba2.ConvKernel = 2, 3, 4, 20, 4
	s.KDA.Heads, s.KDA.HeadDim, s.KDA.ConvKernel = 2, 4, 3
	s.ShortConv.Dim, s.ShortConv.LCache = 16, 3
	s.Hidden, s.Heads, s.FFN = 36, 4, 48
	s.MLA.QLoRA, s.MLA.Rank, s.MLA.QKHead, s.MLA.VHead = 12, 16, 10, 6
	s.PLE = 5
	s.Gemma4MoE.Experts, s.Gemma4MoE.TopK, s.Gemma4MoE.ExpertFFN = 8, 2, 40
	return s
}

// TestBuffers_eachOperation: what each operation declares beyond its weights, at a shape with every width set, and that the rest declare
// nothing. A KV-shared Gemma 4 layer stores no rows and projects no K or V; a windowed one carries its window. Each row is the f32
// activation the operation writes per row of a step: attention's q, k, v and context (gated attention's [query ‖ gate] and gate too), the
// FFN's gate and up (up alone where it is not gated), and each whole-sublayer operation's own.
func TestBuffers_eachOperation(t *testing.T) {
	s := eachOperationShape()
	kv := func(w int) []Buf {
		return []Buf{{"k", PerPosition, PrecKV, 16, 2, w, false}, {"v", PerPosition, PrecKV, 16, 2, w, false}}
	}
	st := func(name string, n int, ck bool) Buf { return Buf{name, PerSlot, PrecF32, n, 0, 0, ck} }
	rows := func(nameElems ...any) []Buf {
		var out []Buf
		for i := 0; i < len(nameElems); i += 2 {
			out = append(out, Buf{Name: nameElems[i].(string), Unit: PerRow, Prec: PrecF32, Elems: nameElems[i+1].(int)})
		}
		return out
	}
	cat := func(parts ...[]Buf) []Buf {
		var out []Buf
		for _, p := range parts {
			out = append(out, p...)
		}
		return out
	}
	// q is 4 heads of 8, k and v 2 of 8; DeltaNet's keys are 2×4, its values 3×5, its conv 2·8+15; Mamba-2's inner width is 2×3.
	cases := []struct {
		name string
		op   Op
		want []Buf
	}{
		{"q, k, v", Op{Kind: OpProjQKV}, rows("q", 32, "k", 16, "v", 16)},
		{"attention", Op{Kind: OpAttention}, cat(kv(0), rows("ctx", 32))},
		{"windowed attention", Op{Kind: OpAttention, Attn: AttentionParams{Window: 512}}, cat(kv(512), rows("ctx", 32))},
		{"gated attention", Op{Kind: OpGatedAttn, Gated: GatedAttnParams{Attn: AttentionParams{Window: 7}}},
			cat(kv(7), rows("q-gate", 64, "q", 32, "k", 16, "v", 16, "attn-gate", 32, "ctx", 32))},
		{"gemma 4 owning", Op{Kind: OpGemma4Attn, G4: Gemma4AttnParams{OwnsKV: true, Window: 9}}, cat(kv(9), rows("q", 32, "k", 16, "v", 16, "ctx", 32))},
		{"gemma 4 shared", Op{Kind: OpGemma4Attn, G4: Gemma4AttnParams{KVSrc: 1}}, rows("q", 32, "ctx", 32)},
		{"mla", Op{Kind: OpMLA}, cat([]Buf{{"latent", PerPosition, PrecF32, 24, 0, 0, false}},
			rows("mla-qa", 12, "mla-q", 40, "mla-kv", 24, "mla-q-abs", 96, "mla-wsum", 64, "mla-ctx", 24))},
		{"deltanet", Op{Kind: OpDeltaNet}, cat([]Buf{st("window", 3*(2*8+15), true), st("state", 3*4*5, true)},
			rows("dn-mixed", 31, "dn-conv", 31, "dn-beta", 3, "dn-decay", 3, "dn-ba", 6, "dn-q", 8, "dn-k", 8, "dn-core", 15, "dn-z", 15, "dn-gated", 15))},
		{"mamba2", Op{Kind: OpMamba2}, cat([]Buf{st("window", 3*20, true), st("state", 2*3*4, true)},
			rows("mb-proj", 6+20+2, "mb-conv", 20, "mb-y", 6, "mb-gated", 6))},
		{"kda", Op{Kind: OpKDA}, []Buf{st("window-q", 2*8, false), st("window-k", 2*8, false), st("window-v", 2*8, false), st("state", 2*4*4, false)}},
		{"short conv", Op{Kind: OpShortConv}, []Buf{st("window", 2*16, false)}},
		{"gated ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGated}}, rows("gate", 48, "up", 48)},
		{"gemma 4 dense ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGemma4Dense}}, rows("gate", 48, "up", 48)},
		{"non-gated ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnNonGated}}, rows("up", 48)},
		{"moe ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnMoE, MoE: MoEConfig{NumExperts: 8, TopK: 2, IntermediateDim: 20, SharedIntermediateDim: 24}}},
			rows("gate", 20, "up", 20, "shared-gate", 24, "shared-up", 24, "moe-router", 8, "moe-experts", 2, "moe-weights", 2)},
		{"gpt-oss ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnGptOss, MoE: MoEConfig{NumExperts: 4, TopK: 2, IntermediateDim: 20}}},
			rows("gate", 20, "up", 20, "gptoss-router", 4, "gptoss-experts", 2, "gptoss-weights", 2)},
		{"llama 4 moe ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnLlama4MoE, MoE: MoEConfig{NumExperts: 4, TopK: 1, IntermediateDim: 20, SharedIntermediateDim: 20}}},
			rows("gate", 20, "up", 20, "shared-gate", 20, "shared-up", 20, "llama4moe-router", 4, "llama4moe-experts", 1, "llama4moe-weights", 1)},
		{"nemotron moe ffn", Op{Kind: OpFFN, FFN: FFNParams{Form: FfnNemotronMoE, MoE: MoEConfig{NumExperts: 4, TopK: 2, IntermediateDim: 20, SharedIntermediateDim: 28}}},
			rows("up", 20, "shared-up", 28, "nemotronmoe-router", 4, "nemotronmoe-experts", 2, "nemotronmoe-weights", 2)},
		{"gemma 4 moe", Op{Kind: OpGemma4MoE}, rows("gate", 48, "up", 48, "g4moe-router", 8, "g4moe-experts", 2, "g4moe-weights", 2,
			"g4moe-dense-out", 36, "g4moe-moe-out", 36, "g4moe-router-in", 36)},
		{"ple", Op{Kind: OpPLE}, rows("ple-gate", 5)},
	}
	for _, c := range cases {
		if got := Buffers(c.op, s); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: %+v, want %+v", c.name, got, c.want)
		}
	}
	for _, k := range []OpKind{OpNorm, OpQKNorm, OpRope, OpAttnTemp, OpOutGate, OpProjO, OpResidual, OpLayerScale} {
		if got := Buffers(Op{Kind: k}, s); got != nil {
			t.Errorf("operation %d declares %+v; it holds nothing beyond its weights", k, got)
		}
	}
	layer := []Op{{Kind: OpNorm}, {Kind: OpAttention}, {Kind: OpFFN, FFN: FFNParams{Form: FfnGated}}, {Kind: OpDeltaNet}}
	want := cat(Buffers(layer[1], s), Buffers(layer[2], s), Buffers(Op{Kind: OpDeltaNet}, s))
	if got := LayerBuffers(layer, s); !reflect.DeepEqual(got, want) {
		t.Errorf("LayerBuffers: %+v, want %+v", got, want)
	}
}

// TestHeadBuffers: the rows no layer owns are the residual stream, a sublayer's input and output, and the logits; the parallel block's
// second output joins them where a layer writes it, and the per-layer-embedding inputs, every layer's, where the embedding has them.
func TestHeadBuffers(t *testing.T) {
	row := func(name string, n int) Buf { return Buf{Name: name, Unit: PerRow, Prec: PrecF32, Elems: n} }
	s := HeadShape{Hidden: 36, Vocab: 100, PLE: 5}
	serial := []Op{{Kind: OpNorm, Out: RegX}, {Kind: OpAttention}, {Kind: OpResidual}}
	parallel := []Op{{Kind: OpNorm, Out: RegX}, {Kind: OpAttention}, {Kind: OpFFN, Out: RegB}, {Kind: OpResidual, Res: ResidualParams{Two: true}}}
	base := []Buf{row("h", 36), row("x", 36), row("a", 36)}
	cases := []struct {
		name string
		p    Program
		want []Buf
	}{
		{"serial", Program{Layers: [][]Op{serial, serial}}, append(append([]Buf{}, base...), row("logits", 100))},
		{"parallel", Program{Layers: [][]Op{serial, parallel}}, append(append([]Buf{}, base...), row("b", 36), row("logits", 100))},
		{"per-layer embedding", Program{Embed: EmbedParams{PLE: true}, Layers: [][]Op{serial, serial, serial}},
			append(append([]Buf{}, base...), row("logits", 100), row("ple-in", 3*5))},
	}
	for _, c := range cases {
		if got := HeadBuffers(&c.p, s); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: %+v, want %+v", c.name, got, c.want)
		}
	}
}
