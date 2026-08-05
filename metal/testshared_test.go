//go:build darwin

package metal

import (
	"math"
	"testing"
)

// Shared metal test helpers that must remain available to tests NOT gated on the
// goinfer_testhooks build tag (B-08). mustFinite moved here out of model_test.go
// (which is tagged) so untagged tests like attention_test.go still see it.

// mustFinite fails the test if metric is NaN/Inf — the degenerate-output signal a
// bare threshold check cannot catch (NaN compares false to every bound).
func mustFinite(t *testing.T, label string, metric float64) {
	t.Helper()
	if math.IsNaN(metric) || math.IsInf(metric, 0) {
		t.Fatalf("%s is %v — degenerate (NaN/Inf) output; the threshold check below cannot catch this "+
			"(NaN compares false to every bound)", label, metric)
	}
}

func argmaxF(v []float32) int {
	bi, bv := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func cosF(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

type parityStats struct {
	steps, exact, hard int
	nan                int // positions whose cosine came back NaN/Inf — see the guard in assertParity
	worstTie, minCos   float64
}

func (st *parityStats) observeCos(c float64) {
	if math.IsNaN(c) || math.IsInf(c, 0) {
		st.nan++
	} else if c < st.minCos {
		st.minCos = c
	}
}

func assertParity(t *testing.T, what string, st parityStats, minCosBar float64) {
	t.Helper()
	// A NaN/Inf cosine is the loudest possible failure (degenerate GPU output) and must fail
	// LOUDLY — it cannot reach the floor because `NaN < x` is false, so it is checked first and
	// explicitly. Without this a kernel emitting NaN reads as perfect parity.
	if st.nan > 0 {
		t.Errorf("%s: %d/%d positions had a NaN/Inf logit cosine — degenerate GPU output, the worst "+
			"kind of failure, which the min-cosine floor cannot catch (NaN < x is false)", what, st.nan, st.steps)
	}
	if st.minCos < minCosBar {
		t.Errorf("%s: min logit cosine %.6f < %.2f — gross breakage, not int4 noise", what, st.minCos, minCosBar)
	}
	if st.exact*2 < st.steps { // control: 15/24 = 62% → require >= 50%
		t.Errorf("%s: argmax parity %d/%d < 50%% — the control manages ~62%% on this harness", what, st.exact, st.steps)
	}
}
