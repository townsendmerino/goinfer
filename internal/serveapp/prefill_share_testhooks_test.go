//go:build goinfer_testhooks

package serveapp

import (
	"context"
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// loneThreshold is the least available memory at which a lone request of this size is admitted: found by bisection
// through the real admission check, so the test does not restate its pricing.
func loneThreshold(t *testing.T, m *decoder.Model, promptTokens, maxTokens int, residentPath bool) int64 {
	t.Helper()
	lo, hi := int64(1), int64(1)<<40
	for lo < hi {
		mid := lo + (hi-lo)/2
		restore := decoder.SetHostRAMAvailableForTest(mid)
		ok := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, residentPath, 1) == nil
		restore()
		if ok {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return lo
}

// prepareWithAhead runs prepare on lm with `ahead` generations already holding a turn, at concurrency 4, with twice
// the memory a lone request of this size needs.
func prepareWithAhead(t *testing.T, lm *loadedModel, ahead int) error {
	t.Helper()
	const promptTokens, maxTokens = 32, 8
	ids := make([]int, promptTokens)
	for i := range ids {
		ids[i] = 1 + i
	}
	thr := loneThreshold(t, lm.model, promptTokens, maxTokens, lm.residentPath())
	lm.concurrent = 4
	lm.turns.setCap(4)
	for range ahead {
		if !lm.turns.enter(context.Background(), admissionRecord{}) {
			t.Fatal("admission refused a holder")
		}
	}
	defer func() {
		for range ahead {
			lm.turns.release()
		}
	}()
	restore := decoder.SetHostRAMAvailableForTest(2 * thr)
	defer restore()
	mt := maxTokens
	_, err := lm.prepare(sampling{MaxTokens: &mt}, ids, lm.residentPath())
	return err
}

// TestPrepare_prefillShare: with several generations running, a request's prefill-memory margin is split among them
// only where their prefills really run at the same time.
//   - CPU workers (MC3c) prefill concurrently, so with 3 ahead at concurrency 4 a request gets a quarter of the margin.
//     With twice what it needs alone, it is refused (413).
//   - A GPU resident's prefills run one at a time (MC3's exclusive section), and the margin is live memory, which
//     already excludes what the other generations hold. So the same request is admitted.
//
// Before the fix the resident request was split four ways too: on the 7B, MC3 cells lost ~1000-token prompts to 413
// (docs/measurements/spec-vs-batching-metal-2026-09-27.md §5).
func TestPrepare_prefillShare(t *testing.T) {
	t.Run("cpu workers split the margin", func(t *testing.T) {
		_, lm := tinyServed(t)
		if lm.residentPath() {
			t.Fatal("tinyServed is resident; the CPU case needs the CPU path")
		}
		err := prepareWithAhead(t, lm, 3)
		var pm *prefillMemoryError
		if !errors.As(err, &pm) {
			t.Fatalf("3 CPU generations ahead, twice a lone request's need: err = %v, want a prefillMemoryError (413)", err)
		}
		if err := prepareWithAhead(t, lm, 0); err != nil {
			t.Fatalf("a lone CPU request with twice its need: %v", err)
		}
	})
	t.Run("a resident prefills one at a time", func(t *testing.T) {
		lm, _ := residentServed(t)
		if err := prepareWithAhead(t, lm, 3); err != nil {
			t.Fatalf("3 resident generations ahead, twice a lone request's need: %v — the margin was split as if their "+
				"prefills ran at the same time", err)
		}
	})
	t.Run("the resident guard still refuses", func(t *testing.T) {
		lm, _ := residentServed(t)
		const promptTokens, maxTokens = 32, 8
		thr := loneThreshold(t, lm.model, promptTokens, maxTokens, true)
		restore := decoder.SetHostRAMAvailableForTest(thr / 2)
		defer restore()
		ids := make([]int, promptTokens)
		for i := range ids {
			ids[i] = 1 + i
		}
		mt := maxTokens
		_, err := lm.prepare(sampling{MaxTokens: &mt}, ids, true)
		var pm *prefillMemoryError
		if !errors.As(err, &pm) {
			t.Fatalf("a lone resident request with half its need: err = %v, want a prefillMemoryError", err)
		}
	})
}
