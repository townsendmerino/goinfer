package decoder

import (
	"math"
	"testing"
)

// validGPT2Config is a tiny validateGPT2-clean GPT-2 config (no checkpoint).
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

// TestLayerNorm checks the mean-centered, biased normalization against a
// hand-computed reference. For x=[1,2,3,4]: mean=2.5, var=1.25, and with
// weight=1, bias=0 the output is (x-mean)/sqrt(var+eps).
func TestLayerNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	layerNorm(x, w, b, 1, 4, 0)
	mean, variance := 2.5, 1.25
	inv := 1.0 / math.Sqrt(variance)
	for i, raw := range []float64{1, 2, 3, 4} {
		want := float32((raw - mean) * inv)
		if math.Abs(float64(x[i]-want)) > 1e-5 {
			t.Errorf("layerNorm[%d] = %v, want %v", i, x[i], want)
		}
	}
	// Normalized output has ~zero mean and unit variance.
	var s float64
	for _, v := range x {
		s += float64(v)
	}
	if math.Abs(s) > 1e-5 {
		t.Errorf("normalized mean = %v, want ~0", s/4)
	}

	// weight + bias are applied affinely: out = norm*w + b.
	x2 := []float32{1, 2, 3, 4}
	layerNorm(x2, []float32{2, 2, 2, 2}, []float32{1, 1, 1, 1}, 1, 4, 0)
	for i := range x2 {
		want := x[i]*2 + 1
		if math.Abs(float64(x2[i]-want)) > 1e-5 {
			t.Errorf("affine layerNorm[%d] = %v, want %v", i, x2[i], want)
		}
	}
}
