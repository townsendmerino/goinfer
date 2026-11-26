package decoder

import (
	"testing"
)

// validGPT2Config is a tiny ValidateGPT2-clean GPT-2 config (no checkpoint).
func validGPT2Config() *Config {
	return &Config{
		ModelType: "gpt2", VocabSize: 64, NEmbd: 16, NHead: 4, NLayer: 2,
		NPositions: 32, LayerNormEpsilon: 1e-5, ActivationFunction: "gelu_new",
	}
}

func TestResolveArchitecture_gpt2(t *testing.T) {
	a, _, err := resolveArchitecture(validGPT2Config())
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Name", a.Name, "gpt2"},
		{"Norm", a.Norm, NormLayer},
		{"NormPlacement", a.NormPlacement, NormPre2},
		{"NonGatedMLP", a.NonGatedMLP, true},
		{"QKVBias", a.QKVBias, true},
		{"OutBias", a.OutBias, true},
		{"LearnedPosEmbed", a.LearnedPosEmbed, true},
		{"QKNorm", a.QKNorm, false},
		{"TiedLMHead", a.TiedLMHead, true},
		{"HeadDim", a.HeadDim, 4},                  // n_embd 16 / n_head 4
		{"IntermediateDim", a.IntermediateDim, 64}, // 4 * n_embd
		{"NumKVHeads", a.NumKVHeads, 4},            // no GQA
		{"MaxPositions", a.MaxPositions, 32},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}
