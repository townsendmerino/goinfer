package decoder

import (
	"context"
	"fmt"
)

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

// residentImagePrefill builds the same spliced embedding rows prefillLogitsVL's CPU path would
// have built (embedN + the raw-feature splice at [imgPos,imgPos+imgLen), no embed scale — matching
// HF's masked_scatter) and hands them to the resident backend's ResidentImagePrefill, skipping the
// CPU prefill and the UploadKV bridge entirely. Any error is a DECLINE, not fatal — the caller
// falls through to the CPU-prefill+UploadKV bridge (gap 0, docs/multimodal.md).
func (m *Model) residentImagePrefill(ctx context.Context, rip ResidentImagePrefill, ids []int, imageEmbeds []float32, imgPos, imgLen int) (logits []float32, gpuPos int, err error) {
	hidden := m.w.arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, 0, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageEmbeds) != imgLen*hidden {
		return nil, 0, fmt.Errorf("decoder: imageEmbeds len %d, want %d (%d tokens × %d)", len(imageEmbeds), imgLen*hidden, imgLen, hidden)
	}
	h := m.embedN(ids)
	copy(h[imgPos*hidden:(imgPos+imgLen)*hidden], imageEmbeds) // raw projected features, no embed scale
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden]
	}
	logits, err = rip.PrefillImageLast(ctx, rows, 0, imgPos, imgPos+imgLen)
	if err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}
