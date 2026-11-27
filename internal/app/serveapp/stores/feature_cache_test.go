package stores

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

func TestFeatureCache_wrapRunsTheTowerOncePerImage(t *testing.T) {
	c := NewFeatureCache(1 << 20)
	var calls atomic.Int32
	mk := func(raw []byte, v float32) func() ([]float32, error) {
		return c.Wrap(raw, func() ([]float32, error) { calls.Add(1); return fcFeats(64, v), nil })
	}
	a, b := []byte("image-a"), []byte("image-b")

	f1, err := mk(a, 1)()
	if err != nil {
		t.Fatal(err)
	}
	f2, _ := mk(a, 99)()
	if calls.Load() != 1 {
		t.Errorf("tower ran %d times for the same bytes, want 1", calls.Load())
	}
	if !slices.Equal(f1, f2) {
		t.Errorf("the resend got different features than the first encode")
	}
	if f3, _ := mk(b, 7)(); calls.Load() != 2 || slices.Equal(f3, f1) {
		t.Errorf("a different image reused the first one's entry (calls=%d)", calls.Load())
	}

	boom := errors.New("tower failed")
	failing := c.Wrap([]byte("image-c"), func() ([]float32, error) { calls.Add(1); return nil, boom })
	if _, err := failing(); !errors.Is(err, boom) {
		t.Fatalf("error not returned: %v", err)
	}
	before := calls.Load()
	if _, err := failing(); !errors.Is(err, boom) || calls.Load() != before+1 {
		t.Errorf("a failed encode was served from the cache (calls %d -> %d)", before, calls.Load())
	}
}

func TestFeatureCache_hitIsACopy(t *testing.T) {
	c := NewFeatureCache(1 << 20)
	key := sha256.Sum256([]byte("x"))
	c.Put(key, fcFeats(8, 1))
	got, ok := c.Get(key)
	if !ok {
		t.Fatal("miss after put")
	}
	got[0] = -1000
	again, _ := c.Get(key)
	if again[0] != 1 {
		t.Errorf("editing a returned slice changed the cached entry: %v", again[0])
	}
	src := fcFeats(8, 5)
	k2 := sha256.Sum256([]byte("y"))
	c.Put(k2, src)
	src[0] = -1000
	if g, _ := c.Get(k2); g[0] != 5 {
		t.Errorf("editing the slice that was put changed the cached entry: %v", g[0])
	}
}

func TestFeatureCache_budgetEvictsLeastRecentlyUsed(t *testing.T) {
	const entry = 100 * 4
	c := NewFeatureCache(2*entry + 10)
	k := func(s string) [sha256.Size]byte { return sha256.Sum256([]byte(s)) }
	c.Put(k("a"), fcFeats(100, 1))
	c.Put(k("b"), fcFeats(100, 2))
	if _, ok := c.Get(k("a")); !ok {
		t.Fatal("a missing")
	}
	c.Put(k("c"), fcFeats(100, 3))
	if _, ok := c.Get(k("b")); ok {
		t.Error("b survived: the least recently used entry was not the one evicted")
	}
	for _, s := range []string{"a", "c"} {
		if _, ok := c.Get(k(s)); !ok {
			t.Errorf("%s was evicted", s)
		}
	}
	if c.used > c.budget {
		t.Errorf("used %d bytes over a budget of %d", c.used, c.budget)
	}
	c.Put(k("huge"), fcFeats(10_000, 9))
	if _, ok := c.Get(k("huge")); ok {
		t.Error("an entry larger than the whole budget was cached")
	}
	if _, ok := c.Get(k("a")); !ok {
		t.Error("an oversized put flushed an existing entry")
	}
}

func TestFeatureCache_concurrent(t *testing.T) {
	c := NewFeatureCache(10 * 100 * 4)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			for i := range 200 {
				raw := []byte{byte(i % 17), byte(g % 3)}
				f, err := c.Wrap(raw, func() ([]float32, error) { return fcFeats(100, float32(raw[0])), nil })()
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

func TestFeatureCacheBudgetFor(t *testing.T) {
	const mib = 1 << 20
	for _, c := range []struct {
		name       string
		known      bool
		budget, kv int64
		want       int
	}{
		{"ample", true, 8 << 30, 2 << 30, FeatureCacheBudget},
		{"exactly the cap left", true, 2<<30 + FeatureCacheBudget, 2 << 30, FeatureCacheBudget},
		{"less left", true, 2<<30 + 100*mib, 2 << 30, 100 * mib},
		{"nothing left", true, 2 << 30, 2 << 30, 0},
		{"KV over budget", true, 1 << 30, 2 << 30, 0},
		{"unpriced", false, 0, 0, FeatureCacheBudget},
	} {
		if got := FeatureCacheBudgetFor(c.known, c.budget, c.kv); got != c.want {
			t.Errorf("%s: FeatureCacheBudgetFor(%v, %d, %d) = %d, want %d", c.name, c.known, c.budget, c.kv, got, c.want)
		}
	}
}
