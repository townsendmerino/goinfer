//go:build cuda && goinfer_testhooks

// Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
// only under -tags goinfer_testhooks so they are NOT part of the public API
// (audit B-08). See RELEASING.md. Mirrors decoder/testhooks_gen.go and gpu/testhooks_gen.go.

package cuda

// CacheStatsForTest sums the LRU cache hits/misses across all layers (C′ step 2 measurement). A
// miss is one expert's H2D DMA; hit rate = hits/(hits+misses) is the fraction of per-token expert
// bytes reuse saves.
func (r *cudaResident) CacheStatsForTest() (hits, misses uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			hits += c.hits
			misses += c.misses
		}
	}
	return hits, misses
}
