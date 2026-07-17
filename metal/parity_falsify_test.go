//go:build darwin

package metal

import (
	"math"
	"testing"
)

// TestParity_NaNCosineFailsTheGate is the break-it-first proof for the NaN<minCos vacuity
// (parity-coverage-policy.md § "The metric must be able to fail"). It forces the failure the
// hole would hide — a degenerate GPU output whose cosine is NaN — and confirms the real gate
// path (cosF → observeCos → assertParity) goes RED, not silently green.
//
// Why this test is necessary and not paranoia: the hole is invisible by construction. A kernel
// emitting NaN makes cosF return NaN; `NaN < minCos` is false so minCos stays at its init 1.0;
// the floor `minCos < 0.95` is then false and the gate PASSES. Every number in the log looks
// perfect. Only an explicit NaN count catches it, and a guard never seen to fire is an
// assumption. This runs on any box (no GPU, no checkpoint) so the guard itself is CI-covered.
func TestParity_NaNCosineFailsTheGate(t *testing.T) {
	// 1. The metric really does produce NaN on degenerate output — the premise of the hole.
	nanVec := []float32{float32(math.NaN()), 1, 2}
	good := []float32{0.1, 1, 2}
	if c := cosF(good, nanVec); !math.IsNaN(c) {
		t.Fatalf("premise broken: cosF over a NaN vector returned %v, not NaN — the hole would not exist", c)
	}

	// 2. observeCos must COUNT that NaN, not drop it into the min reduction.
	var st parityStats
	st.minCos = 1
	st.observeCos(cosF(good, nanVec))
	if st.nan != 1 {
		t.Errorf("observeCos dropped a NaN cosine (nan=%d) — it would sail through the floor", st.nan)
	}
	if st.minCos != 1 {
		t.Errorf("a NaN cosine leaked into minCos (%v) — expected it counted, not reduced", st.minCos)
	}

	// 3. The real assertion path must go RED on nan>0, EVEN with an otherwise-perfect record
	//    (minCos 1.0, full argmax parity) — proving the NaN check is what fails it, nothing else.
	rec := &testing.T{}
	perfectButNaN := parityStats{steps: 24, exact: 24, minCos: 1.0, nan: 1}
	assertParity(rec, "forced-nan", perfectButNaN)
	if !rec.Failed() {
		t.Error("assertParity PASSED a run with a NaN cosine and otherwise-perfect stats — the gate is vacuous")
	}

	// 4. Control: the same perfect record WITHOUT the NaN must pass, so the failure above is the
	//    NaN and not some unrelated assertion.
	recOK := &testing.T{}
	assertParity(recOK, "clean", parityStats{steps: 24, exact: 24, minCos: 1.0})
	if recOK.Failed() {
		t.Error("assertParity failed a clean perfect record — the NaN guard has a false positive")
	}
}
