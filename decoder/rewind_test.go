package decoder

import "testing"

// TestTruncateTo_wrappedRingRewindIsInexact gates C1: TruncateTo must report whether a rewind is
// EXACT. On a wrapped sliding-window ring, a rewind of >1 position needs positions the ring
// physically evicted, so it can't be restored — the flag lets prefix-reuse / spec cold-prefill
// instead of reading stale history.
func TestTruncateTo_wrappedRingRewindIsInexact(t *testing.T) {
	const W, hd, nKV = 4, 2, 1
	c := NewKVCache(1, nKV, hd, W, 16)
	c.enableRings(W, func(l int) bool { return false }) // layer 0 local (ring)
	kvDim := nKV * hd
	for p := 0; p < 10; p++ { // 10 positions into a W=4 ring ⇒ wraps
		c.Append(0, make([]float32, kvDim), make([]float32, kvDim))
	}
	if c.Pos() != 10 {
		t.Fatalf("pos = %d, want 10", c.Pos())
	}
	if !c.TruncateTo(9) { // rewind by 1 — dropped slot is still the most recent ⇒ exact
		t.Error("rewind by 1 on a wrapped ring should be exact")
	}
	if c.TruncateTo(6) { // rewind by ≥2 — window [2,6) needs evicted positions ⇒ INEXACT
		t.Error("rewind by ≥2 on a wrapped ring must report inexact (C1)")
	}
}

// TestTruncateTo_raggedLayerUsesRecordedStride gates M11: a mid-sweep forward error leaves earlier
// layers with one extra (ragged) row; TruncateTo must slice by the per-layer stride recorded at
// first append, not len/pos (which mis-derives the width when pos < stride and leaves the layer
// permanently misaligned).
func TestTruncateTo_raggedLayerUsesRecordedStride(t *testing.T) {
	const hd, nKV = 8, 1
	c := NewKVCache(2, nKV, hd, 0, 16) // window 0 ⇒ global (append-forever) layers
	kvDim := nKV * hd
	k := make([]float32, kvDim)
	for p := 0; p < 2; p++ { // 2 clean positions across both layers
		c.Append(0, k, k)
		c.Append(1, k, k)
	}
	c.Append(0, k, k) // extra row on layer 0 only (mid-sweep error): 3 rows, pos still 2
	if c.Pos() != 2 {
		t.Fatalf("pos = %d, want 2", c.Pos())
	}
	c.TruncateTo(2)
	if got, want := len(c.keys[0]), 2*kvDim; got != want {
		t.Errorf("ragged layer truncated to %d elems, want %d (2 positions × stride %d) — len/pos mis-sliced it (M11)", got, want, kvDim)
	}
}
