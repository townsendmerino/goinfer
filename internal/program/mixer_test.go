package program

import "testing"

// TestMixerOf_eachKind is OP4cu-b1's direct check of MixerOf (docs/tasks/task-forward-architecture-2026-10.md §4.9a): each mixer
// operation maps to its own mixer, a layer with none is MixerNone, and the FFN is an OpFFN or Gemma 4's MoE block.
func TestMixerOf_eachKind(t *testing.T) {
	norm, ffn := Op{Kind: OpNorm}, Op{Kind: OpFFN}
	for _, c := range []struct {
		name  string
		ops   []Op
		mixer Mixer
		ffn   bool
	}{
		{"attention", []Op{norm, {Kind: OpProjQKV}, {Kind: OpAttention}, {Kind: OpProjO}, ffn}, MixerAttention, true},
		{"gated attention", []Op{norm, {Kind: OpGatedAttn}, ffn}, MixerAttention, true},
		{"Gemma 4 attention and MoE", []Op{norm, {Kind: OpGemma4Attn}, {Kind: OpGemma4MoE}}, MixerAttention, true},
		{"Nemotron-H attention block", []Op{norm, {Kind: OpProjQKV}, {Kind: OpAttention}, {Kind: OpProjO}}, MixerAttention, false},
		{"DeltaNet", []Op{norm, {Kind: OpDeltaNet}, ffn}, MixerDeltaNet, true},
		{"Mamba-2 block", []Op{norm, {Kind: OpMamba2}}, MixerMamba, false},
		{"MLA", []Op{norm, {Kind: OpMLA}, ffn}, MixerMLA, true},
		{"KDA", []Op{norm, {Kind: OpKDA}, ffn}, MixerKDA, true},
		{"short convolution", []Op{norm, {Kind: OpShortConv}, ffn}, MixerShortConv, true},
		{"FFN-only block", []Op{norm, ffn}, MixerNone, true},
	} {
		if m, f := MixerOf(c.ops); m != c.mixer || f != c.ffn {
			t.Errorf("%s: MixerOf (%v, ffn %v), want (%v, ffn %v)", c.name, m, f, c.mixer, c.ffn)
		}
	}
}

// TestMixer_resident: the five mixers that mirror decoder.ResidentMixer are resident, KDA and the short convolution are not.
func TestMixer_resident(t *testing.T) {
	for _, x := range []Mixer{MixerAttention, MixerDeltaNet, MixerMamba, MixerNone, MixerMLA} {
		if !x.Resident() {
			t.Errorf("%v: not resident", x)
		}
	}
	for _, x := range []Mixer{MixerKDA, MixerShortConv} {
		if x.Resident() {
			t.Errorf("%v: resident", x)
		}
	}
}
