package decoder

import (
	"math"
	"strings"
	"testing"
)

// validGemma3Config is a tiny but ValidateAssumptions-clean gemma3 config for
// descriptor tests (no checkpoint needed).
func validGemma3Config() *Config {
	return &Config{
		ModelType: "gemma3_text", VocabSize: 100, HiddenDim: 8, NumLayers: 2,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 16, RMSNormEps: 1e-6,
		RoPELocalBase: 10000, RoPEGlobalBase: 1000000, SlidingWindow: 512,
		SlidingWindowPattern: 6, QueryPreAttnScalar: 4, HiddenActivation: "gelu_pytorch_tanh",
	}
}

func TestResolveArchitecture_gemma3(t *testing.T) {
	a, _, err := resolveArchitecture(validGemma3Config())
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Name", a.Name, "gemma3"},
		{"Norm", a.Norm, NormRMS},
		{"RMSAddOne", a.RMSAddOne, true},
		{"NormPlacement", a.NormPlacement, NormSandwich4},
		{"Act", a.Act, ActGeluTanh},
		{"QKNorm", a.QKNorm, true},
		{"TiedLMHead", a.TiedLMHead, true},
		{"AttnScale", a.AttnScale, 0.5},                 // 4^-0.5
		{"EmbedScale", a.EmbedScale, math.Sqrt(8)},      // √hidden
		{"FinalLogitSoftcap", a.FinalLogitSoftcap, 0.0}, // Gemma 3 = none
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if b := a.ropeBase(0); b != 10000 { // layer 0 is local under pattern 6
		t.Errorf("ropeBase(0) = %v, want local 10000", b)
	}
}

func TestResolveArchitecture_unknownModelType(t *testing.T) {
	cfg := validGemma3Config()
	cfg.ModelType = "mamba" // not a registered decoder family
	_, _, err := resolveArchitecture(cfg)
	if err == nil {
		t.Fatal("expected error for unknown model_type")
	}
	if !strings.Contains(err.Error(), "unsupported model_type") || !strings.Contains(err.Error(), "mamba") {
		t.Errorf("error = %q, want it to name the unsupported model_type", err)
	}
}

func TestSilu(t *testing.T) {
	sigmoid := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	for _, x := range []float32{-4, -1, -0.3, 0, 0.7, 2, 5} {
		want := float64(x) * sigmoid(float64(x))
		if d := math.Abs(float64(silu(x)) - want); d > 1e-6 {
			t.Errorf("silu(%v) = %v, want %v (Δ%v)", x, silu(x), want, d)
		}
	}
	if silu(0) != 0 {
		t.Errorf("silu(0) = %v, want 0", silu(0))
	}
}

// TestRMSNorm_addOne checks the (1+w) vs w branch: with weight=0, addOne scales
// by 1 (identity up to normalization) while !addOne scales by 0 (zeroes out).
func TestRMSNorm_addOne(t *testing.T) {
	mk := func() []float32 { return []float32{1, 2, 3, 4} }
	w := []float32{0, 0, 0, 0}

	withOne := mk()
	rmsNorm(withOne, w, 1, 4, 1e-6, true)
	var ss float64
	for _, v := range withOne {
		ss += float64(v) * float64(v)
	}
	if ss == 0 {
		t.Error("addOne=true zeroed the row; expected normalized values")
	}

	without := mk()
	rmsNorm(without, w, 1, 4, 1e-6, false)
	for i, v := range without {
		if v != 0 {
			t.Errorf("addOne=false with weight 0: out[%d]=%v, want 0", i, v)
		}
	}
}
