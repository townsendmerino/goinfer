package decoder

import "testing"

// TestMellum2_slidingWindowEviction is the model-free proof of Mellum2's 3:1
// sliding/full interleave eviction (the real-checkpoint proof is
// TestMellum2_windowParity). On a synthetic small-window cache it asserts: (1)
// the 3:1 layer dispatch, (2) WindowStart evicts on sliding layers but not full
// ones, and (3) behaviorally — a sliding layer's attention output is INVARIANT to
// a key outside its window, while a full layer's output DOES change. Same
// eviction mechanic the Gemma window test exercises, here without a 24 GB model.
func TestMellum2_slidingWindowEviction(t *testing.T) {
	const W, hd, N = 4, 2, 8 // window 4, 8 stored positions → positions 0..3 evicted on sliding layers
	arch := &Architecture{
		NumHeads: 1, NumKVHeads: 1, HeadDim: hd, AttnScale: 1.0,
		SlidingWindow: W,
		layerIsGlobal: func(i int) bool { return (i+1)%4 == 0 }, // 3 sliding : 1 full (Mellum2)
	}

	// 1. Layer dispatch: 0,1,2 sliding; 3 full.
	for i, want := range []bool{false, false, false, true} {
		if arch.isGlobalLayer(i) != want {
			t.Errorf("layer %d isGlobal=%v, want %v", i, arch.isGlobalLayer(i), want)
		}
	}

	// 2. WindowStart at a position past the window.
	pos := N - 1 // 7
	build := func(blowUpPos0 bool) *KVCache {
		c := NewKVCache(1, 1, hd, W, N)
		for p := 0; p < N; p++ {
			k, v := []float32{float32(p + 1), 0}, []float32{float32(p + 1), float32(p + 1)}
			if blowUpPos0 && p == 0 {
				k, v = []float32{999, 0}, []float32{999, 999} // a key far OUTSIDE the [4,7] window
			}
			c.Append(0, k, v)
		}
		return c
	}
	c := build(false)
	if got := c.WindowStart(pos, false); got != pos-W+1 {
		t.Errorf("sliding WindowStart(%d) = %d, want %d (evicts positions < %d)", pos, got, pos-W+1, pos-W+1)
	}
	if got := c.WindowStart(pos, true); got != 0 {
		t.Errorf("full WindowStart(%d) = %d, want 0 (attends all)", pos, got)
	}

	// 3. Behavioral: blowing up an out-of-window key (position 0) must NOT change
	// a sliding layer's output, but MUST change a full layer's.
	q := []float32{1, 0}
	attend := func(c *KVCache, global bool) []float32 {
		ctx := make([]float32, hd)
		attendQuery(q, ctx, make([]float32, N), c, 0, pos, global, arch)
		return ctx
	}
	c2 := build(true)
	slid, slid2 := attend(c, false), attend(c2, false)
	full, full2 := attend(c, true), attend(c2, true)

	if !approxEqual(slid, slid2, 1e-6) {
		t.Errorf("sliding layer output changed when an OUT-of-window key changed: %v vs %v", slid, slid2)
	}
	if approxEqual(full, full2, 1e-6) {
		t.Errorf("full layer ignored a key it should attend (output unchanged): %v == %v", full, full2)
	}
	t.Logf("eviction OK: sliding invariant to evicted key (%v), full attends it (%v → %v)", slid, full, full2)
}

func approxEqual(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := float64(a[i] - b[i])
		if d < -tol || d > tol {
			return false
		}
	}
	return true
}
