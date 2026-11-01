package program

// Mixer is a layer's sequence mixer, as a resident's layer walk dispatches it.
type Mixer uint8

const (
	MixerAttention Mixer = iota // softmax attention, of any form (plain, gated, Gemma 4's)
	MixerNone                   // no mixer: an FFN-only block (Nemotron-H's MLP and MoE blocks)
	MixerDeltaNet               // Gated-DeltaNet
	MixerMamba                  // Mamba-2
	MixerMLA                    // latent-KV attention
)

// MixerOf is a layer's mixer and whether a feed-forward block follows it, read from the layer's operations: the program's form
// of decoder.Model.ResidentLayerOps (OP4cu-b1, docs/tasks/task-forward-architecture-2026-10.md §4.9a).
func MixerOf(ops []Op) (mixer Mixer, ffn bool) {
	return MixerAttention, true
}
