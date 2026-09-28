package confidence

import (
	"math"
	"testing"
)

func TestLogSumExpFinite(t *testing.T) {
	ninf := float32(math.Inf(-1))
	// legal = {0: ln 1, 2: ln 3}: normalizer ln 4, and p(2) = 3/4.
	logits := []float32{0, ninf, float32(math.Log(3)), ninf}
	lse, n := LogSumExpFinite(logits)
	if n != 2 {
		t.Fatalf("legal = %d, want 2", n)
	}
	if math.Abs(lse-math.Log(4)) > 1e-6 {
		t.Errorf("lse = %v, want ln 4 = %v", lse, math.Log(4))
	}
	if p := math.Exp(float64(logits[2]) - lse); math.Abs(p-0.75) > 1e-6 {
		t.Errorf("p(2) = %v, want 0.75", p)
	}
	// Large logits must not overflow (the max is subtracted first).
	if lse, n := LogSumExpFinite([]float32{1000, 1000}); n != 2 || math.Abs(lse-(1000+math.Log(2))) > 1e-3 {
		t.Errorf("large logits: lse %v legal %d", lse, n)
	}
	if lse, n := LogSumExpFinite([]float32{ninf, ninf}); n != 0 || !math.IsInf(lse, -1) {
		t.Errorf("all masked: lse %v legal %d, want -Inf 0", lse, n)
	}
}

func TestRestrictedLogSoftmax(t *testing.T) {
	ninf := float32(math.Inf(-1))
	logits := []float32{2, 0, float32(math.Log(3)), 5, ninf}
	// Over ids {1, 2}: logits 0 and ln 3, so p = 1/4, 3/4; id 3's large logit is outside the set.
	lp := RestrictedLogSoftmax(logits, []int{1, 2}, 1)
	if math.Abs(math.Exp(lp[0])-0.25) > 1e-6 || math.Abs(math.Exp(lp[1])-0.75) > 1e-6 {
		t.Errorf("p = %v, %v, want 0.25, 0.75", math.Exp(lp[0]), math.Exp(lp[1]))
	}
	// Temperature 2 halves the logits: ln 3 / 2 → p ∝ 1, √3.
	lp = RestrictedLogSoftmax(logits, []int{1, 2}, 2)
	want := math.Sqrt(3) / (1 + math.Sqrt(3))
	if math.Abs(math.Exp(lp[1])-want) > 1e-6 {
		t.Errorf("T=2: p(2) = %v, want %v", math.Exp(lp[1]), want)
	}
	// An out-of-range or masked id takes no mass; the rest still sums to 1.
	lp = RestrictedLogSoftmax(logits, []int{0, 4, 99}, 1)
	if lp[0] != 0 || !math.IsInf(lp[1], -1) || !math.IsInf(lp[2], -1) {
		t.Errorf("masked/out-of-range: %v, want [0 -Inf -Inf]", lp)
	}
	if lp := RestrictedLogSoftmax(logits, []int{4}, 0); !math.IsInf(lp[0], -1) {
		t.Errorf("no finite id: %v, want -Inf", lp)
	}
}
