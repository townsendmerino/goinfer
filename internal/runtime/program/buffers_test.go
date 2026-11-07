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
			case b.Unit != PerSlot && b.Unit != PerPosition:
				t.Errorf("operation %d declares %q with no unit, which no consumer reads", k, b.Name)
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

// TestBuffers_eachOperation: what each operation declares beyond its weights, at a shape with every width set, and that the rest declare
// nothing. A KV-shared Gemma 4 layer declares no rows; a windowed one carries its window.
func TestBuffers_eachOperation(t *testing.T) {
	var s LayerShape
	s.KVHeads, s.HeadDim, s.MLALatent = 2, 8, 24
	s.DeltaNet.KeyHeads, s.DeltaNet.KeyHeadDim, s.DeltaNet.ValueHeads, s.DeltaNet.ValueHeadDim, s.DeltaNet.ConvKernel = 2, 4, 3, 5, 4
	s.Mamba2.Heads, s.Mamba2.HeadDim, s.Mamba2.DState, s.Mamba2.ConvDim, s.Mamba2.ConvKernel = 2, 3, 4, 20, 4
	s.KDA.Heads, s.KDA.HeadDim, s.KDA.ConvKernel = 2, 4, 3
	s.ShortConv.Dim, s.ShortConv.LCache = 16, 3
	kv := func(w int) []Buf {
		return []Buf{{"k", PerPosition, PrecKV, 16, 2, w, false}, {"v", PerPosition, PrecKV, 16, 2, w, false}}
	}
	st := func(name string, n int, ck bool) Buf { return Buf{name, PerSlot, PrecF32, n, 0, 0, ck} }
	cases := []struct {
		name string
		op   Op
		want []Buf
	}{
		{"attention", Op{Kind: OpAttention}, kv(0)},
		{"windowed attention", Op{Kind: OpAttention, Attn: AttentionParams{Window: 512}}, kv(512)},
		{"gated attention", Op{Kind: OpGatedAttn, Gated: GatedAttnParams{Attn: AttentionParams{Window: 7}}}, kv(7)},
		{"gemma 4 owning", Op{Kind: OpGemma4Attn, G4: Gemma4AttnParams{OwnsKV: true, Window: 9}}, kv(9)},
		{"gemma 4 shared", Op{Kind: OpGemma4Attn, G4: Gemma4AttnParams{KVSrc: 1}}, nil},
		{"mla", Op{Kind: OpMLA}, []Buf{{"latent", PerPosition, PrecF32, 24, 0, 0, false}}},
		{"deltanet", Op{Kind: OpDeltaNet}, []Buf{st("window", 3*(2*8+15), true), st("state", 3*4*5, true)}},
		{"mamba2", Op{Kind: OpMamba2}, []Buf{st("window", 3*20, true), st("state", 2*3*4, true)}},
		{"kda", Op{Kind: OpKDA}, []Buf{st("window-q", 2*8, false), st("window-k", 2*8, false), st("window-v", 2*8, false), st("state", 2*4*4, false)}},
		{"short conv", Op{Kind: OpShortConv}, []Buf{st("window", 2*16, false)}},
	}
	for _, c := range cases {
		if got := Buffers(c.op, s); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: %+v, want %+v", c.name, got, c.want)
		}
	}
	for _, k := range []OpKind{OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpOutGate, OpProjO, OpFFN, OpResidual, OpGemma4MoE, OpPLE, OpLayerScale} {
		if got := Buffers(Op{Kind: k}, s); got != nil {
			t.Errorf("operation %d declares %+v; it holds nothing beyond its weights", k, got)
		}
	}
	layer := []Op{{Kind: OpNorm}, {Kind: OpAttention}, {Kind: OpFFN}, {Kind: OpDeltaNet}}
	if got, want := LayerBuffers(layer, s), append(kv(0), Buffers(Op{Kind: OpDeltaNet}, s)...); !reflect.DeepEqual(got, want) {
		t.Errorf("LayerBuffers: %+v, want %+v", got, want)
	}
}
