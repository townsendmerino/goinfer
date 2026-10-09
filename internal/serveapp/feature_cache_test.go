package serveapp

import (
	"crypto/sha256"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

func fcFeats(n int, v float32) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = v + float32(i)
	}
	return f
}

// The point of the cache: the second request carrying the same bytes does not run the tower again, and gets exactly
// the numbers the first one computed. A different image is a different entry, and an error is not remembered.
func TestFeatureCache_wrapRunsTheTowerOncePerImage(t *testing.T) {
	c := newFeatureCache(1 << 20)
	var calls atomic.Int32
	mk := func(raw []byte, v float32) func() ([]float32, error) {
		return c.wrap(raw, func() ([]float32, error) { calls.Add(1); return fcFeats(64, v), nil })
	}
	a, b := []byte("image-a"), []byte("image-b")

	f1, err := mk(a, 1)()
	if err != nil {
		t.Fatal(err)
	}
	f2, _ := mk(a, 99)() // a resend: the compute func would return different numbers if it ran
	if calls.Load() != 1 {
		t.Errorf("tower ran %d times for the same bytes, want 1", calls.Load())
	}
	if !slices.Equal(f1, f2) {
		t.Errorf("the resend got different features than the first encode")
	}
	if f3, _ := mk(b, 7)(); calls.Load() != 2 || slices.Equal(f3, f1) {
		t.Errorf("a different image reused the first one's entry (calls=%d)", calls.Load())
	}

	// An error is returned, not cached: the next call computes again.
	boom := errors.New("tower failed")
	failing := c.wrap([]byte("image-c"), func() ([]float32, error) { calls.Add(1); return nil, boom })
	if _, err := failing(); !errors.Is(err, boom) {
		t.Fatalf("error not returned: %v", err)
	}
	before := calls.Load()
	if _, err := failing(); !errors.Is(err, boom) || calls.Load() != before+1 {
		t.Errorf("a failed encode was served from the cache (calls %d -> %d)", before, calls.Load())
	}
}

// A hit hands back a copy. The decoder only reads features today; the cache must not depend on that staying true.
func TestFeatureCache_hitIsACopy(t *testing.T) {
	c := newFeatureCache(1 << 20)
	key := sha256.Sum256([]byte("x"))
	c.put(key, fcFeats(8, 1))
	got, ok := c.get(key)
	if !ok {
		t.Fatal("miss after put")
	}
	got[0] = -1000
	again, _ := c.get(key)
	if again[0] != 1 {
		t.Errorf("editing a returned slice changed the cached entry: %v", again[0])
	}
	src := fcFeats(8, 5)
	k2 := sha256.Sum256([]byte("y"))
	c.put(k2, src)
	src[0] = -1000
	if g, _ := c.get(k2); g[0] != 5 {
		t.Errorf("editing the slice that was put changed the cached entry: %v", g[0])
	}
}

// The byte budget holds: the least recently used entry goes first, a get refreshes recency, and one image bigger than
// the whole budget is neither cached nor allowed to flush the others.
func TestFeatureCache_budgetEvictsLeastRecentlyUsed(t *testing.T) {
	const entry = 100 * 4
	c := newFeatureCache(2*entry + 10) // room for two
	k := func(s string) [sha256.Size]byte { return sha256.Sum256([]byte(s)) }
	c.put(k("a"), fcFeats(100, 1))
	c.put(k("b"), fcFeats(100, 2))
	if _, ok := c.get(k("a")); !ok { // a is now newer than b
		t.Fatal("a missing")
	}
	c.put(k("c"), fcFeats(100, 3)) // evicts b
	if _, ok := c.get(k("b")); ok {
		t.Error("b survived: the least recently used entry was not the one evicted")
	}
	for _, s := range []string{"a", "c"} {
		if _, ok := c.get(k(s)); !ok {
			t.Errorf("%s was evicted", s)
		}
	}
	if c.used > c.budget {
		t.Errorf("used %d bytes over a budget of %d", c.used, c.budget)
	}
	c.put(k("huge"), fcFeats(10_000, 9))
	if _, ok := c.get(k("huge")); ok {
		t.Error("an entry larger than the whole budget was cached")
	}
	if _, ok := c.get(k("a")); !ok {
		t.Error("an oversized put flushed an existing entry")
	}
}

// Several requests for the same image at once must not corrupt the list or the byte count (run under -race in CI).
func TestFeatureCache_concurrent(t *testing.T) {
	c := newFeatureCache(10 * 100 * 4)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			for i := range 200 {
				raw := []byte{byte(i % 17), byte(g % 3)}
				f, err := c.wrap(raw, func() ([]float32, error) { return fcFeats(100, float32(raw[0])), nil })()
				if err != nil || len(f) != 100 || f[0] != float32(raw[0]) {
					t.Errorf("goroutine %d: got %v, %v", g, f[:1], err)
					return
				}
			}
		})
	}
	wg.Wait()
	if c.used < 0 || c.used > c.budget || c.order.Len() != len(c.items) {
		t.Errorf("bookkeeping drifted: used %d budget %d list %d map %d", c.used, c.budget, c.order.Len(), len(c.items))
	}
}

// Through the seam visionPrompt uses: a model whose tower cannot run (this one has none) still answers from a
// pre-seeded cache, keyed by the SHA-256 of the image bytes, and a different image falls through to the tower.
func TestWithFeatureCache_answersBeforeTheTowerRuns(t *testing.T) {
	lm := &loadedModel{}
	raw := []byte("png bytes")
	want := fcFeats(16, 3)
	lm.visionFeatureCache().put(sha256.Sum256(raw), want)

	towerRan := false
	vi := visionInput{features: func() ([]float32, error) { towerRan = true; return nil, errors.New("no tower") }}

	got, err := lm.withFeatureCache(vi, raw, "test").features()
	if err != nil || towerRan || !slices.Equal(got, want) {
		t.Errorf("seeded image: got %v, %v (tower ran: %v), want the cached features", got, err, towerRan)
	}
	if _, err := lm.withFeatureCache(vi, []byte("other bytes"), "test").features(); err == nil || !towerRan {
		t.Errorf("an unseen image did not reach the tower (err %v, ran %v)", err, towerRan)
	}
}
