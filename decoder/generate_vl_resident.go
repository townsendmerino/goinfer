package decoder

import "fmt"

// residentUploadPrefill pushes a CPU-computed prefill's KV (every layer) into the resident GPU
// cache — the bridge that lets GenerateVL/GenerateQwenVL's bidirectional-image-block CPU
// prefill (prefillLogitsVL/prefillLogitsQwenVL) seed a resident decode instead of running the
// whole turn on CPU (gap 0, docs/multimodal.md). cache must hold exactly the prefill this turn
// just ran — nothing more, nothing skipped.
//
// Per layer, KVCache.LayerKV (not Keys/Vals, which read empty for a sliding-window/ring layer or
// an int8-quantized CPU cache) returns the live K/V in absolute position order plus base, the
// absolute position of its first row — 0 for a global layer or a ring that has not wrapped,
// greater than 0 once a local layer's ring has wrapped past its window. UploadKV places those
// rows at base..base+n-1 in the resident cache, matching where a resident Forward would have
// written them.
func (m *Model) residentUploadPrefill(cache *KVCache) error {
	for l := 0; l < m.w.arch.NumLayers; l++ {
		k, v, base := cache.LayerKV(l)
		if len(k) == 0 {
			continue // a layer with nothing live yet (shouldn't happen post-prefill, but not fatal)
		}
		if err := m.resident.UploadKV(l, base, k, v); err != nil {
			return fmt.Errorf("decoder: resident prefill upload layer %d (base=%d): %w", l, base, err)
		}
	}
	return nil
}
