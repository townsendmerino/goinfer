package normact

import (
	"math"
	"testing"
)

// The two must actually be different functions, and GELUErf must match the real definition. Without this, decoder's
// gpt2Act could route correctly (TestGPT2_activationSelectsTheNamedFunction) to an ActGelu implemented as another copy
// of the tanh form, and every other test would still pass.
func TestGeluErf_isExactAndDiffersFromTanh(t *testing.T) {
	maxDiff := 0.0
	for i := -10000; i <= 10000; i++ {
		x := float32(i) / 1000
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		got := float64(GELUErf(x))
		if d := math.Abs(got - want); d > 1e-6 {
			t.Fatalf("geluErf(%v) = %v, want %v (Δ%g)", x, got, want, d)
		}
		if d := math.Abs(float64(GELUErf(x) - GELUTanh(x))); d > maxDiff {
			maxDiff = d
		}
	}
	// The measured separation. If this collapses toward zero, GELUErf has been
	// reimplemented as the tanh form and decoder's dispatch test is decorative.
	if maxDiff < 4e-4 {
		t.Errorf("max |erf - tanh| = %g, expected ~4.73e-4 — are these the same function?", maxDiff)
	}
	t.Logf("max |geluErf - geluTanh| = %.3e (worst near x = -2.7)", maxDiff)
}

// TestLayerNorm checks the mean-centered, biased normalization against a
// hand-computed reference. For x=[1,2,3,4]: mean=2.5, var=1.25, and with
// weight=1, bias=0 the output is (x-mean)/sqrt(var+eps).
func TestLayerNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	Layer(x, w, b, 1, 4, 0)
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
	Layer(x2, []float32{2, 2, 2, 2}, []float32{1, 1, 1, 1}, 1, 4, 0)
	for i := range x2 {
		want := x[i]*2 + 1
		if math.Abs(float64(x2[i]-want)) > 1e-5 {
			t.Errorf("affine layerNorm[%d] = %v, want %v", i, x2[i], want)
		}
	}
}
