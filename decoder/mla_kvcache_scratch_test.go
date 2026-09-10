package decoder

import "testing"

// TestNewCache_mlaSkipsKeysValsReservation is P-02 (audit-2026-09-10): an MLA family
// (DeepSeek-V2/V3, Kimi K2/V3) never writes c.keys[l]/c.vals[l] on any layer — the per-layer
// compressed latent (c.mlaLatent) is the whole KV store — but NewCache reserved full
// capHint*kvDim capacity for them anyway, on every layer. Asserts the reservation itself is
// gone (cap == 0), not just that nothing later reads it, and checks a NON-mla model right
// alongside to prove the fix is conditional on a.mla rather than a blanket capHint=0 that would
// silently break every other family's real KV growth.
func TestNewCache_mlaSkipsKeysValsReservation(t *testing.T) {
	const numLayers, numKVHeads, headDim, capHint = 3, 4, 8, 4096

	mlaModel := &Model{w: &Weights{arch: &Architecture{
		NumLayers: numLayers, NumKVHeads: numKVHeads, HeadDim: headDim,
		mla: &mlaParams{KVLoRARank: 16},
	}}}
	mc := mlaModel.NewCache(capHint)
	for l := range numLayers {
		if c := cap(mc.keys[l]); c != 0 {
			t.Errorf("mla model layer %d: cap(keys) = %d, want 0 — the wasted reservation is still happening", l, c)
		}
		if c := cap(mc.vals[l]); c != 0 {
			t.Errorf("mla model layer %d: cap(vals) = %d, want 0 — the wasted reservation is still happening", l, c)
		}
	}

	denseModel := &Model{w: &Weights{arch: &Architecture{
		NumLayers: numLayers, NumKVHeads: numKVHeads, HeadDim: headDim,
	}}}
	dc := denseModel.NewCache(capHint)
	wantCap := capHint * numKVHeads * headDim
	for l := range numLayers {
		if c := cap(dc.keys[l]); c != wantCap {
			t.Errorf("dense model layer %d: cap(keys) = %d, want %d — the fix over-reached onto a non-mla family", l, c, wantCap)
		}
		if c := cap(dc.vals[l]); c != wantCap {
			t.Errorf("dense model layer %d: cap(vals) = %d, want %d — the fix over-reached onto a non-mla family", l, c, wantCap)
		}
	}
}
