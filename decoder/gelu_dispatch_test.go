package decoder

import (
	"math"
	"testing"
)

// GPT-2's activation_function must select the function it NAMES. validateGPT2 accepts both "gelu_new" and "gelu", so
// gpt2Architecture must not hardcode ActGeluTanh: a checkpoint declaring HF's exact "gelu" would silently run the tanh
// approximation. In HF these are different entries (ACT2FN["gelu"] is the exact erf, ACT2FN["gelu_new"] the tanh form),
// not two spellings of one function. The next GPT-2-family checkpoint that declares "gelu" must get the function it
// asked for.
func TestGPT2_activationSelectsTheNamedFunction(t *testing.T) {
	for _, c := range []struct {
		name string
		want ActKind
	}{
		{"", ActGeluTanh},         // GPT-2's own default IS gelu_new
		{"gelu_new", ActGeluTanh}, // tanh approximation
		{"gelu", ActGelu},         // exact erf — the case that was wrong
	} {
		if got := gpt2Act(c.name); got != c.want {
			t.Errorf("gpt2Act(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

// The two must actually be different functions, and geluErf must match the real
// definition. Without this, gpt2Act could route correctly to an ActGelu that was
// implemented as another copy of the tanh form and every other test would still pass.
func TestGeluErf_isExactAndDiffersFromTanh(t *testing.T) {
	maxDiff := 0.0
	for i := -10000; i <= 10000; i++ {
		x := float32(i) / 1000
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		got := float64(geluErf(x))
		if d := math.Abs(got - want); d > 1e-6 {
			t.Fatalf("geluErf(%v) = %v, want %v (Δ%g)", x, got, want, d)
		}
		if d := math.Abs(float64(geluErf(x) - geluTanh(x))); d > maxDiff {
			maxDiff = d
		}
	}
	// The measured separation. If this collapses toward zero, geluErf has been
	// reimplemented as the tanh form and the dispatch above is decorative.
	if maxDiff < 4e-4 {
		t.Errorf("max |erf - tanh| = %g, expected ~4.73e-4 — are these the same function?", maxDiff)
	}
	t.Logf("max |geluErf - geluTanh| = %.3e (worst near x = -2.7)", maxDiff)
}
