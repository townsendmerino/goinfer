package serveapp

import (
	"container/list"
	"crypto/sha256"
	"fmt"
	"os"
	"sync"
	"time"
)

// A vision tower is the slow part of an image turn on the families whose tower runs on the CPU, and a chat client
// resends the conversation, images included, every turn. This caches the tower's OUTPUT per image, so the encode is paid
// once. The decoder's own image reuse keeps the KV of a resent image, but only on a resident non-recurrent model, so a
// hybrid like Qwen3.5 would otherwise re-encode every time.
//
// The features are a pure function of the image bytes and of the loaded model's fixed vision settings, and the cache is
// per loaded model, so a hit is bit-identical to recomputing. Keyed by SHA-256 of the raw bytes rather than the 64-bit
// imgHash the KV reuse uses: this one hands back numbers instead of skipping a prefix, and a collision there would be a
// wrong picture, not a missed optimisation.
const featureCacheBudget = 256 << 20 // bytes of cached features per loaded model

type featureCache struct {
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

func newFeatureCache(budgetBytes int) *featureCache {
	return &featureCache{budget: budgetBytes, order: list.New(), items: map[[sha256.Size]byte]*list.Element{}}
}

func (c *featureCache) get(key [sha256.Size]byte) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(el)
	return append([]float32(nil), el.Value.(*featureEntry).feats...), true // a copy: no caller can edit what the next one reads
}

func (c *featureCache) put(key [sha256.Size]byte, feats []float32) {
	size := len(feats) * 4
	if size > c.budget {
		return // one image bigger than the whole budget is not cached, and evicts nothing for it
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
	c.items[key] = c.order.PushFront(&featureEntry{key: key, feats: append([]float32(nil), feats...)})
	c.used += size
}

// wrap returns a features function that answers from the cache when it holds this image and otherwise runs compute and
// remembers the result. Failures are never cached. The log line says which happened and how long the encode took.
func (c *featureCache) wrap(raw []byte, compute func() ([]float32, error)) func() ([]float32, error) {
	key := sha256.Sum256(raw)
	return func() ([]float32, error) {
		if feats, ok := c.get(key); ok {
			fmt.Fprintf(os.Stderr, "vision: reused the cached encode of this image (%d bytes, sha256 %x)\n", len(raw), key[:4])
			return feats, nil
		}
		t0 := time.Now()
		feats, err := compute()
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(os.Stderr, "vision: encoded a %d-byte image in %s (sha256 %x); a resend of it will skip this\n", len(raw), time.Since(t0).Round(time.Millisecond), key[:4])
		c.put(key, feats)
		return feats, nil
	}
}
