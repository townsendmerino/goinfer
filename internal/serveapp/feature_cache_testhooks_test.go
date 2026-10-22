//go:build goinfer_testhooks

package serveapp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// cacheSizeAt is the budget of the feature cache lm makes on its first image with avail bytes of memory available, and
// the fit plan it read.
func cacheSizeAt(lm *loadedModel, avail int64) (size int, budgetBytes, kvBytes int64, known bool) {
	restore := decoder.SetHostRAMAvailableForTest(avail)
	defer restore()
	_, kvBytes, _, budgetBytes, known = lm.model.FitBudgetSummary()
	return lm.visionFeatureCache().budget, budgetBytes, kvBytes, known
}

// Through the cache's real constructor on a loaded model: ample memory keeps featureCacheBudget; tight memory gives the
// fit plan's leftover after its KV; and a plan with nothing left caches no image, so the tower runs on every resend.
func TestVisionFeatureCache_sizedByTheFitPlan(t *testing.T) {
	t.Run("ample", func(t *testing.T) {
		_, lm := tinyServed(t)
		if got, _, _, known := cacheSizeAt(lm, 1<<40); !known || got != featureCacheBudget {
			t.Errorf("1 TiB available: cache %d (plan priced %v), want featureCacheBudget %d", got, known, featureCacheBudget)
		}
	})
	t.Run("tight", func(t *testing.T) {
		_, lm := tinyServed(t)
		_, kv, _, _, _ := lm.model.FitBudgetSummary()
		got, budget, kvAt, known := cacheSizeAt(lm, 2*kv+64<<20)
		if !known {
			t.Fatal("the tiny model's fit plan is unpriced; the tight case needs a priced one")
		}
		if want := budget - kvAt; want <= 0 || want >= featureCacheBudget || int64(got) != want {
			t.Errorf("cache %d, want the plan's leftover %d (budget %d - KV %d), strictly between 0 and %d", got, want, budget, kvAt, featureCacheBudget)
		}
	})
	t.Run("nothing left", func(t *testing.T) {
		_, lm := tinyServed(t)
		if got, _, _, _ := cacheSizeAt(lm, 1); got != 0 {
			t.Fatalf("1 byte available: cache %d, want 0", got)
		}
		runs := 0
		vi := visionInput{features: func() ([]float32, error) { runs++; return fcFeats(16, 1), nil }}
		for range 2 {
			if _, err := lm.withFeatureCache(vi, []byte("png bytes"), "test").features(); err != nil {
				t.Fatal(err)
			}
		}
		if runs != 2 {
			t.Errorf("the tower ran %d times for a resent image with no cache budget, want 2", runs)
		}
	})
}

// With the fit guard off, or --fit=off, the cache keeps featureCacheBudget however little memory is available.
func TestVisionFeatureCache_fitOffKeepsTheFullBudget(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	for name, opts := range map[string]decoder.Options{
		"GOINFER_NO_FIT_GUARD": {Backend: "cpu", Quant: "int8int8", Knobs: &decoder.Knobs{"GOINFER_NO_FIT_GUARD": "1"}},
		"--fit=off":            {Backend: "cpu", Quant: "int8int8", DisableFit: true},
	} {
		m, err := decoder.Load(p, opts)
		if err != nil {
			t.Fatalf("%s: load: %v", name, err)
		}
		lm := &loadedModel{name: "tiny", model: m}
		got, _, _, _ := cacheSizeAt(lm, 1)
		_ = m.Close()
		if got != featureCacheBudget {
			t.Errorf("%s, 1 byte available: cache %d, want featureCacheBudget %d", name, got, featureCacheBudget)
		}
	}
}
