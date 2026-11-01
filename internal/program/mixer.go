package program

// Mixer is a layer's sequence mixer, as a resident's layer walk dispatches it. The first five are in the order of
// decoder.ResidentMixer, which they mirror.
type Mixer uint8

const (
	MixerAttention Mixer = iota // softmax attention, of any form (plain, qwen3.5's gated, Gemma 4's)
	MixerDeltaNet               // Gated-DeltaNet
	MixerMamba                  // Mamba-2
	MixerNone                   // no mixer: an FFN-only block (Nemotron-H's MLP and MoE blocks)
	MixerMLA                    // latent-KV attention
	MixerKDA                    // Kimi Delta Attention, which no resident runs
	MixerShortConv              // LFM2's gated short convolution, which no resident runs
)

// MixerOf is a layer's mixer and whether a feed-forward block follows it, read from the layer's operations: the program's form
// of decoder.Model.ResidentLayerOps (OP4cu-b1, docs/tasks/task-forward-architecture-2026-10.md §4.9a). A layer whose
// operations hold no mixer is MixerNone; an FFN is an OpFFN or Gemma 4's parallel dense and MoE block.
func MixerOf(ops []Op) (mixer Mixer, ffn bool) {
	mixer = MixerNone
	for i := range ops {
		switch ops[i].Kind {
		case OpAttention, OpGatedAttn, OpGemma4Attn:
			mixer = MixerAttention
		case OpDeltaNet:
			mixer = MixerDeltaNet
		case OpMamba2:
			mixer = MixerMamba
		case OpMLA:
			mixer = MixerMLA
		case OpKDA:
			mixer = MixerKDA
		case OpShortConv:
			mixer = MixerShortConv
		case OpFFN, OpGemma4MoE:
			ffn = true
		}
	}
	if mixer == MixerNone && ShapePlantForTest == ShapePlantFFNOnlyAsAttend {
		mixer = MixerAttention
	}
	return mixer, ffn
}

// Resident reports whether some resident runs the mixer: false for KDA and the short convolution, whose families every
// backend declines before it reads a layer's mixer.
func (x Mixer) Resident() bool { return x != MixerKDA && x != MixerShortConv }
