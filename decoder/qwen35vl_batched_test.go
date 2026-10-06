package decoder

import (
	"context"
	"slices"
	"testing"
)

// P26 (docs/queue-performance.md): the Qwen3.5 image prefill runs every projection as one matmul over the prompt
// (prefillQwen35VLBatched) instead of one token at a time (prefillQwen35VLPerToken). The batched path is NOT claimed bit-identical,
// so the bound is stated here and was fixed before the first run: over several prompt layouts around the image on the tiny
// fixture, the last-token logits agree to cosine >= 0.999999, the argmax is the same, and the greedy continuation over the
// same 8 decode steps is the same, decoding from each path's own cache (so a KV or recurrent-state difference shows up).
// The test also asserts the batched path is ELIGIBLE on the fixture, so it cannot pass by comparing the per-token loop with itself.
func TestQwen35VL_batchedPrefillMatchesPerToken(t *testing.T) {
	m, g := loadQwen35VLTiny(t)
	feats := g.ImageFeatures
	imgTok := g.ImageToken
	layouts := []struct {
		name          string
		pre, post     int // extra text tokens before and after the golden prompt
		preID, postID int
	}{
		{"the golden prompt", 0, 0, 0, 0},
		{"three text tokens before the image", 3, 0, 5, 0},
		{"five text tokens after the image", 0, 5, 0, 7},
		{"text before and after", 4, 6, 9, 11},
	}
	for _, L := range layouts {
		ids := slices.Concat(slices.Repeat([]int{L.preID}, L.pre), g.InputIDs, slices.Repeat([]int{L.postID}, L.post))
		imgPos := g.ImageStart + L.pre
		pos, err := mropePositions(ids, imgTok, g.GridTHW, qwen35VLMerge)
		if err != nil {
			t.Fatalf("%s: %v", L.name, err)
		}
		prefill := func(batched bool) ([]float32, *KVCache) {
			cache := m.NewCache(len(ids) + 8)
			cache.mropePos = pos
			cache.mropeDelta = mropeDelta(pos, len(ids))
			var lg []float32
			var err error
			if batched {
				if !m.qwen35BatchNAnyPos(len(ids), cache) {
					t.Fatalf("%s: the batched path is not eligible on this fixture, so the comparison would be the per-token loop against itself", L.name)
				}
				lg, err = m.prefillQwen35VLBatched(context.Background(), ids, feats, imgPos, g.NImageTokens, cache)
			} else {
				lg, err = m.prefillQwen35VLPerToken(context.Background(), ids, feats, imgPos, g.NImageTokens, cache)
			}
			if err != nil {
				t.Fatalf("%s (batched=%v): %v", L.name, batched, err)
			}
			return slices.Clone(lg), cache
		}
		per, perCache := prefill(false)
		bat, batCache := prefill(true)
		if cos := logitCosine(bat, per); cos < 0.999999 {
			t.Errorf("%s: batched vs per-token last-logit cosine %.9f < 0.999999", L.name, cos)
		}
		if argmax(bat) != argmax(per) {
			t.Errorf("%s: argmax batched %d, per-token %d", L.name, argmax(bat), argmax(per))
		}
		decode := func(lg []float32, c *KVCache) []int {
			var out []int
			for range 8 {
				id := argmax(lg)
				out = append(out, id)
				var err error
				if lg, err = m.forward(id, c); err != nil {
					t.Fatalf("%s: decode: %v", L.name, err)
				}
			}
			return out
		}
		if a, b := decode(per, perCache), decode(bat, batCache); !slices.Equal(a, b) {
			t.Errorf("%s: continuation per-token %v, batched %v", L.name, a, b)
		}
	}
}
