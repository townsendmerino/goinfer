package serveapp

import (
	"crypto/sha256"
	"errors"
	"slices"
	"testing"
)

func fcFeats(n int, v float32) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = v + float32(i)
	}
	return f
}

// Through the seam visionPrompt uses: a model whose tower cannot run (this one has none) still answers from a
// pre-seeded cache, keyed by the SHA-256 of the image bytes, and a different image falls through to the tower.
func TestWithFeatureCache_answersBeforeTheTowerRuns(t *testing.T) {
	lm := &loadedModel{}
	raw := []byte("png bytes")
	want := fcFeats(16, 3)
	lm.visionFeatureCache().Put(sha256.Sum256(raw), want)

	towerRan := false
	vi := visionInput{Features: func() ([]float32, error) { towerRan = true; return nil, errors.New("no tower") }}

	got, err := lm.withFeatureCache(vi, raw, "test").Features()
	if err != nil || towerRan || !slices.Equal(got, want) {
		t.Errorf("seeded image: got %v, %v (tower ran: %v), want the cached features", got, err, towerRan)
	}
	if _, err := lm.withFeatureCache(vi, []byte("other bytes"), "test").Features(); err == nil || !towerRan {
		t.Errorf("an unseen image did not reach the tower (err %v, ran %v)", err, towerRan)
	}
}
