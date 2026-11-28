package decoder

import (
	"testing"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// TestOptFwdEligible mirrors TestSpecRollbackSafetyGuard's shape (spec_ngram_test.go) for the new
// predicate: resident required, Temperature>0 required, LogitProcessor excludes it, and the
// recurrent/sliding-window specRollbackSafe guard is reused (not reimplemented).
func TestOptFwdEligible(t *testing.T) {
	plain := &Model{w: &darch.Weights{Arch: &darch.Architecture{}}, resident: stubResident{}}
	if !plain.optFwdEligible(SamplingParams{Temperature: 0.2}) {
		t.Error("resident + 0 < T <= cap + no processor + rollback-safe arch must be eligible")
	}
	if plain.optFwdEligible(SamplingParams{Temperature: 0}) {
		t.Error("Temperature<=0 (greedy-equivalent) must NOT be eligible -- fastGreedy already covers it")
	}
	// Every NEGATIVE case below uses a temperature INSIDE the cap on purpose: at 0.7 they would be refused for being too hot,
	// and would pass without exercising the property they name.
	// The threshold itself: above optFwdMaxTemp the overlap is a measured loss, so the common chat range must be excluded;
	// this is the whole behaviour change, and 0.7 being ineligible is the point rather than a regression.
	for _, T := range []float64{0.21, 0.4, 0.7, 1.0, 2.0} {
		if plain.optFwdEligible(SamplingParams{Temperature: T}) {
			t.Errorf("T=%g is above the %g cap and must NOT be eligible -- optFwd loses there", T, optFwdMaxTemp)
		}
	}
	for _, T := range []float64{0.01, 0.1, optFwdMaxTemp} {
		if !plain.optFwdEligible(SamplingParams{Temperature: T}) {
			t.Errorf("T=%g is at or below the %g cap and must stay eligible", T, optFwdMaxTemp)
		}
	}
	// The override exists for measurement; a ladder needs to reach temperatures the cap excludes.
	t.Setenv("GOINFER_OPTFWD_MAX_TEMP", "1.0")
	if !plain.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("GOINFER_OPTFWD_MAX_TEMP must raise the cap, so a ladder can measure above it")
	}
	t.Setenv("GOINFER_OPTFWD_MAX_TEMP", "")
	if plain.optFwdEligible(SamplingParams{Temperature: 0.7}) {
		t.Error("an empty override must fall back to the compiled default, not disable the cap")
	}
	if plain.optFwdEligible(SamplingParams{Temperature: 0.2, LogitProcessor: func([]int, []float32) {}}) {
		t.Error("a LogitProcessor must exclude eligibility -- the argmax guess would be computed on unprocessed logits")
	}
	noResident := &Model{w: &darch.Weights{Arch: &darch.Architecture{}}}
	if noResident.optFwdEligible(SamplingParams{Temperature: 0.2}) {
		t.Error("no resident backend must NOT be eligible")
	}
	recurrent := &Model{w: &darch.Weights{Arch: &darch.Architecture{NumLayers: 1, Qwen35: &archparts.Qwen35Params{}, LayerIsLinear: func(int) bool { return true }}}, resident: stubResident{}}
	if recurrent.optFwdEligible(SamplingParams{Temperature: 0.2}) {
		t.Error("a recurrent (DeltaNet) arch must NOT be eligible -- specRollbackSafe must refuse it")
	}
	windowed := &Model{w: &darch.Weights{Arch: &darch.Architecture{SlidingWindow: 128}}, resident: stubResident{}}
	if windowed.optFwdEligible(SamplingParams{Temperature: 0.2}) {
		t.Error("a sliding-window arch must NOT be eligible -- specRollbackSafe must refuse it")
	}
}
