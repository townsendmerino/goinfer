package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// TestOptFwdGate_hysteresis exercises the EMA/hysteresis on/off logic directly, no model needed:
// starts optimistic (on), a sustained low hit rate must turn it off within a bounded number of
// observations and stay off (no flapping from single draws once off).
func TestOptFwdGate_hysteresis(t *testing.T) {
	g := &optFwdGate{}
	if !g.Should() {
		t.Fatal("gate must start optimistic (on)")
	}

	// A sustained ~73% hit rate (the measured T=1.0 number) must eventually turn the gate off,
	// since it is well below EnableAt (0.90) and even below DisableAt (0.85).
	rng := rand.New(rand.NewSource(1))
	hit := func() bool { return rng.Float64() < 0.73 } // i.i.d., matching the measured T=1.0 rate
	turnedOff := -1
	for i := 0; i < 200; i++ {
		g.Observe(hit())
		if !g.Should() && turnedOff < 0 {
			turnedOff = i
		}
	}
	if turnedOff < 0 {
		t.Fatalf("gate never turned off under a sustained ~75%% hit rate (alpha=%.3f)", g.Alpha())
	}
	// Must STAY off — check a further run of the same rate doesn't flap back on immediately.
	for i := 0; i < 20; i++ {
		g.Observe(hit())
	}
	if g.Should() {
		t.Errorf("gate flapped back on under a sustained sub-threshold hit rate (alpha=%.3f)", g.Alpha())
	}
}

// TestOptFwdGate_highHitRateStaysOn confirms a sustained high hit rate (matching the measured
// T=0.2 number, 99.7%) never turns the gate off.
func TestOptFwdGate_highHitRateStaysOn(t *testing.T) {
	g := &optFwdGate{}
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 200; i++ {
		g.Observe(rng.Float64() < 0.997) // i.i.d., matching the measured T=0.2 rate
		if !g.Should() {
			t.Fatalf("gate turned off under a sustained ~99.7%% hit rate at step %d (alpha=%.3f)", i, g.Alpha())
		}
	}
}

// TestOptFwdGate_marginalRateHoldsSteady checks the T=0.7 measured rate (93.0%, above EnableAt but
// requires the dead band to avoid flapping on noise around it) settles rather than oscillating
// step to step once past the initial transient.
func TestOptFwdGate_marginalRateHoldsSteady(t *testing.T) {
	g := &optFwdGate{}
	rng := rand.New(rand.NewSource(93))
	hit := func() bool { return rng.Float64() < 0.93 } // i.i.d., not a periodic burst pattern
	for i := 0; i < 50; i++ {
		g.Observe(hit()) // warm up past the optimistic seed
	}
	flips := 0
	was := g.Should()
	for i := 0; i < 200; i++ {
		g.Observe(hit())
		if g.Should() != was {
			flips++
			was = g.Should()
		}
	}
	if flips > 2 {
		t.Errorf("gate flapped %d times at a steady 93%% hit rate (alpha settled at %.3f) -- hysteresis band too narrow", flips, g.Alpha())
	}
}

// TestOptFwdEligible mirrors TestSpecRollbackSafetyGuard's shape (spec_ngram_test.go) for the new
// predicate: resident required, Temperature>0 required, LogitProcessor excludes it, and the
// recurrent/sliding-window specRollbackSafe guard is reused (not reimplemented).
func TestOptFwdEligible(t *testing.T) {
	plain := &Model{w: &Weights{arch: &Architecture{}}, resident: stubResident{}}
	if !plain.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("resident + Temperature>0 + no processor + rollback-safe arch must be eligible")
	}
	if plain.optFwdEligible(SamplingParams{Temperature: 0}) {
		t.Error("Temperature<=0 (greedy-equivalent) must NOT be eligible -- fastGreedy already covers it")
	}
	if plain.optFwdEligible(SamplingParams{Temperature: 0.7, LogitProcessor: func([]int, []float32) {}}) {
		t.Error("a LogitProcessor must exclude eligibility -- the argmax guess would be computed on unprocessed logits")
	}
	noResident := &Model{w: &Weights{arch: &Architecture{}}}
	if noResident.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("no resident backend must NOT be eligible")
	}
	recurrent := &Model{w: &Weights{arch: &Architecture{qwen35: &qwen35Params{}}}, resident: stubResident{}}
	if recurrent.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("a recurrent (DeltaNet) arch must NOT be eligible -- specRollbackSafe must refuse it")
	}
	windowed := &Model{w: &Weights{arch: &Architecture{SlidingWindow: 128}}, resident: stubResident{}}
	if windowed.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("a sliding-window arch must NOT be eligible -- specRollbackSafe must refuse it")
	}
}

// TestOptFwdStats_HitRate is a trivial sanity check on the reported telemetry ratio.
func TestOptFwdStats_HitRate(t *testing.T) {
	s := &OptFwdStats{}
	if s.HitRate() != 0 {
		t.Errorf("HitRate on a zero-Guessed stats must be 0, got %v", s.HitRate())
	}
	s.Guessed, s.Hit = 10, 7
	if got := s.HitRate(); math.Abs(got-0.7) > 1e-9 {
		t.Errorf("HitRate = %v, want 0.7", got)
	}
}
