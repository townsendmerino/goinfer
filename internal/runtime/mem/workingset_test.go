package mem

import (
	"math"
	"strings"
	"testing"
)

// TestMoEHitRatePrior pins the calibration points against benchmarks.md §B4.1 directly plus the monotonic, clamped
// shape between and beyond them.
func TestMoEHitRatePrior(t *testing.T) {
	const eps = 1e-9
	for _, c := range []struct {
		name       string
		residency  float64
		want       float64
		approxOnly bool
	}{
		{"zero residency -> zero hit rate", 0, 0, false},
		{"negative residency clamps to zero", -1, 0, false},
		{"12.5% residency (16/128 slots) matches §B4.1", 0.125, 0.573, true},
		{"23.4% residency (30/128 slots) matches §B4.1", 0.234, 0.761, true},
		{"31.25% residency (40/128 slots) matches §B4.1", 0.3125, 0.822, true},
		{"full residency clamps below 1.0", 1.0, 0.999, false},
		{"over-full residency still clamps below 1.0", 5.0, 0.999, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := MoEHitRatePrior(c.residency)
			if c.approxOnly {
				if math.Abs(got-c.want) > 0.01 {
					t.Errorf("MoEHitRatePrior(%.4f) = %.4f, want ~%.3f (calibration point)", c.residency, got, c.want)
				}
				return
			}
			if math.Abs(got-c.want) > eps {
				t.Errorf("MoEHitRatePrior(%.4f) = %.6f, want %.6f", c.residency, got, c.want)
			}
		})
	}
	t.Run("monotonic non-decreasing across the whole domain", func(t *testing.T) {
		prev := MoEHitRatePrior(0)
		for r := 0.0; r <= 1.5; r += 0.01 {
			got := MoEHitRatePrior(r)
			if got < prev-1e-12 {
				t.Fatalf("MoEHitRatePrior(%.2f) = %.6f is LESS than MoEHitRatePrior(prev) = %.6f — not monotonic", r, got, prev)
			}
			prev = got
		}
	})
}

// TestPredictedMoETokPerSec pins 1 / (missBytes / MoEPreadRateBytesPerSec) against a hand-computed case plus every
// "unknown -> 0" edge.
func TestPredictedMoETokPerSec(t *testing.T) {
	t.Run("hand-computed case", func(t *testing.T) {
		const activeBytes = int64(1) << 30
		const hitRate = 0.5
		missBytes := float64(activeBytes) * (1 - hitRate)
		wantSec := missBytes / MoEPreadRateBytesPerSec
		want := 1 / wantSec
		got := PredictedMoETokPerSec(activeBytes, hitRate)
		if math.Abs(got-want) > want*1e-9 {
			t.Errorf("PredictedMoETokPerSec(1 GiB, 0.5) = %.6f, want %.6f", got, want)
		}
	})
	t.Run("zero/negative activeBytesPerToken is unknown", func(t *testing.T) {
		if got := PredictedMoETokPerSec(0, 0.5); got != 0 {
			t.Errorf("PredictedMoETokPerSec(0, 0.5) = %v, want 0", got)
		}
		if got := PredictedMoETokPerSec(-1, 0.5); got != 0 {
			t.Errorf("PredictedMoETokPerSec(-1, 0.5) = %v, want 0", got)
		}
	})
	t.Run("hitRate clamped to 0.999, never reaches +Inf", func(t *testing.T) {
		got := PredictedMoETokPerSec(1<<30, 1.0)
		if got <= 0 || math.IsInf(got, 1) {
			t.Errorf("PredictedMoETokPerSec(1 GiB, 1.0) = %v, want a finite positive number (clamped hit rate)", got)
		}
	})
	t.Run("negative hitRate clamps to 0, not treated as free", func(t *testing.T) {
		got := PredictedMoETokPerSec(1<<30, -1)
		want := PredictedMoETokPerSec(1<<30, 0)
		if got != want {
			t.Errorf("PredictedMoETokPerSec(1 GiB, -1) = %v, want == hitRate=0 case (%v)", got, want)
		}
	})
	t.Run("more active bytes predicts a slower rate, all else equal", func(t *testing.T) {
		small := PredictedMoETokPerSec(1<<20, 0.5)
		big := PredictedMoETokPerSec(1<<30, 0.5)
		if big >= small {
			t.Errorf("PredictedMoETokPerSec did not decrease with more active bytes: small=%.4f big=%.4f", small, big)
		}
	})
	t.Run("PredictMoEWorkingSet rejects unknown inputs and increases with budget", func(t *testing.T) {
		if _, ok := PredictMoEWorkingSet(0, 8, 16, 2, 1<<26, 1<<28); ok {
			t.Error("PredictMoEWorkingSet(totalBytes=0) reported ok")
		}
		if _, ok := PredictMoEWorkingSet(1<<30, 8, 16, 2, 1<<26, 0); ok {
			t.Error("PredictMoEWorkingSet(budgetBytes=0) reported ok")
		}
		low, ok1 := PredictMoEWorkingSet(1<<30, 8, 16, 2, 1<<26, 1<<27)
		high, ok2 := PredictMoEWorkingSet(1<<30, 8, 16, 2, 1<<26, 1<<29)
		if !ok1 || !ok2 || high <= low {
			t.Errorf("PredictMoEWorkingSet(low=%v/%v, high=%v/%v): want both ok and high > low", low, ok1, high, ok2)
		}
	})
}

// TestMoEWorkingSetRefusal verifies the threshold refusal rule across synthetic predicted rates.
func TestMoEWorkingSetRefusal(t *testing.T) {
	t.Run("below floor without acceptSlow refuses", func(t *testing.T) {
		err := MoEWorkingSetRefusal("my-model", 1.0, MoESlowTokPerSecFloor, false)
		if err == nil {
			t.Fatal("MoEWorkingSetRefusal(1.0 tok/s, acceptSlow=false) = nil, want a refusal")
		}
		if !strings.Contains(err.Error(), "my-model") || !strings.Contains(err.Error(), "1.00 tok/s") {
			t.Errorf("refusal message missing the model name or predicted rate: %v", err)
		}
	})
	t.Run("below floor WITH acceptSlow proceeds", func(t *testing.T) {
		if err := MoEWorkingSetRefusal("my-model", 1.0, MoESlowTokPerSecFloor, true); err != nil {
			t.Errorf("MoEWorkingSetRefusal(1.0 tok/s, acceptSlow=true) = %v, want nil", err)
		}
	})
	t.Run("at or above floor proceeds regardless of acceptSlow", func(t *testing.T) {
		if err := MoEWorkingSetRefusal("my-model", 2.0, MoESlowTokPerSecFloor, false); err != nil {
			t.Errorf("MoEWorkingSetRefusal(2.0 tok/s == floor, acceptSlow=false) = %v, want nil", err)
		}
		if err := MoEWorkingSetRefusal("my-model", 50.0, MoESlowTokPerSecFloor, false); err != nil {
			t.Errorf("MoEWorkingSetRefusal(50.0 tok/s, acceptSlow=false) = %v, want nil", err)
		}
	})
	t.Run("predicted <= 0 (unknown) never refuses", func(t *testing.T) {
		if err := MoEWorkingSetRefusal("my-model", 0, MoESlowTokPerSecFloor, false); err != nil {
			t.Errorf("MoEWorkingSetRefusal(0, acceptSlow=false) = %v, want nil (unknown proceeds)", err)
		}
		if err := MoEWorkingSetRefusal("my-model", -1, MoESlowTokPerSecFloor, false); err != nil {
			t.Errorf("MoEWorkingSetRefusal(-1, acceptSlow=false) = %v, want nil (unknown proceeds)", err)
		}
	})
}
