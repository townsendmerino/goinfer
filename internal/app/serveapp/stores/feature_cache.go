package stores

import (
	"container/list"
	"crypto/sha256"
	"fmt"
	"os"
	"slices"
	"sync"
	"time"
)

// FeatureCacheBudget is the maximum bytes a loaded model's vision/audio feature cache holds by default (256 MiB);
// FeatureCacheBudgetFor lowers it when the fit plan leaves less headroom after KV.
const FeatureCacheBudget = 256 << 20

// FeatureCacheBudgetFor computes the feature-cache byte cap for a model whose fit plan has budgetBytes of memory and
// prices kvBytes of KV against it: FeatureCacheBudget, or what the plan leaves after its KV when that is smaller. An
// unpriced plan (known false) keeps FeatureCacheBudget.
func FeatureCacheBudgetFor(known bool, budgetBytes, kvBytes int64) int {
	if !known {
		return FeatureCacheBudget
	}
	return int(min(FeatureCacheBudget, max(budgetBytes-kvBytes, 0)))
}

// FeatureCache caches encoded vision/audio float32 feature slices per raw media SHA-256 digest under an LRU byte cap.
type FeatureCache struct {
	mu     sync.Mutex
	budget int
	used   int
	order  *list.List // front = most recently used
	items  map[[sha256.Size]byte]*list.Element
}

type featureEntry struct {
	key   [sha256.Size]byte
	feats []float32
}

// NewFeatureCache constructs a FeatureCache bounded to budgetBytes.
func NewFeatureCache(budgetBytes int) *FeatureCache {
	return &FeatureCache{budget: budgetBytes, order: list.New(), items: map[[sha256.Size]byte]*list.Element{}}
}

// Budget returns the maximum byte budget configured for c.
func (c *FeatureCache) Budget() int { return c.budget }

// Get returns a copy of key's cached features and promotes it to most-recently-used, or (nil, false) on miss.
func (c *FeatureCache) Get(key [sha256.Size]byte) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(el)
	return slices.Clone(el.Value.(*featureEntry).feats), true
}

// Put stores a copy of feats under key, evicting least-recently-used entries until used + size <= budget. An entry
// larger than the entire budget is skipped without evicting existing entries.
func (c *FeatureCache) Put(key [sha256.Size]byte, feats []float32) {
	size := len(feats) * 4
	if size > c.budget {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.order.MoveToFront(el)
		return
	}
	for c.used+size > c.budget {
		oldest := c.order.Back()
		e := oldest.Value.(*featureEntry)
		c.order.Remove(oldest)
		delete(c.items, e.key)
		c.used -= len(e.feats) * 4
	}
	c.items[key] = c.order.PushFront(&featureEntry{key: key, feats: slices.Clone(feats)})
	c.used += size
}

// Wrap returns a closure that serves raw's encoded features from c on hit and otherwise runs compute and caches a
// successful result. Errors from compute are never cached.
func (c *FeatureCache) Wrap(raw []byte, compute func() ([]float32, error)) func() ([]float32, error) {
	key := sha256.Sum256(raw)
	return func() ([]float32, error) {
		if feats, ok := c.Get(key); ok {
			fmt.Fprintf(os.Stderr, "vision: reused the cached encode of this image (%d bytes, sha256 %x)\n", len(raw), key[:4])
			return feats, nil
		}
		t0 := time.Now()
		feats, err := compute()
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(os.Stderr, "vision: encoded a %d-byte image in %s (sha256 %x); a resend of it will skip this\n", len(raw), time.Since(t0).Round(time.Millisecond), key[:4])
		c.Put(key, feats)
		return feats, nil
	}
}
