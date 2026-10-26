package decoder

import "testing"

// KS-O4's G-MEM (docs/measurements/ks-o4-2026-10/README.md): the bytes gpt-oss's KV cache holds, counted, on the tiny fixture past its
// window, with the window layers on the ring (KS-O4) and with every layer append-forever (the layout before it). The drop must be exactly
// the window layers' append-forever bytes less the ring's 2 x window positions, and the ring layout must match the fit plan's pricing
// (kvBytesForCtx), which already assumed a ring.
func TestKSO4_kvHeldBytes(t *testing.T) {
	m, err := Load("testdata/gptoss_tiny.gguf", Options{DirectLoad: true})
	if err != nil {
		t.Fatalf("load the tiny gpt-oss fixture: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	n := 4*a.SlidingWindow + 3 // well past the window
	withRings := m.NewCache(n)
	appendForever := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, n, a.hasNoAttentionKVAt)
	appendForever.scr = newDecodeScratch(a)
	for range n {
		if _, err := m.forward(1, withRings); err != nil {
			t.Fatal(err)
		}
		if _, err := m.forward(1, appendForever); err != nil {
			t.Fatal(err)
		}
	}
	held := func(c *KVCache) (total int64) {
		for l := range c.numLayers {
			total += 4 * int64(len(c.keys[l])+len(c.vals[l]))
			if r := c.rings[l]; r != nil {
				total += 4 * int64(len(r.k)+len(r.v))
			}
		}
		return total
	}
	ring, forever := held(withRings), held(appendForever)
	var windowRows, globalRows int64
	for l := range a.NumLayers {
		row := 2 * 4 * int64(a.kvDimAt(l)) // K and V, f32
		if a.ringLayerAt(l) {
			windowRows += row
		} else {
			globalRows += row
		}
	}
	wantForever := (windowRows + globalRows) * int64(n)
	wantRing := globalRows*int64(n) + windowRows*int64(2*a.SlidingWindow)
	if forever != wantForever {
		t.Errorf("append-forever holds %d bytes, want %d (every layer, %d positions)", forever, wantForever, n)
	}
	if ring != wantRing {
		t.Errorf("with the ring it holds %d bytes, want %d (window layers at 2 x %d positions)", ring, wantRing, a.SlidingWindow)
	}
	if plan := kvBytesForCtx(a, n, false, false); plan != ring {
		t.Errorf("the fit plan prices %d bytes at %d positions; the ring layout holds %d", plan, n, ring)
	}
	t.Logf("%d positions, window %d: append-forever %d B, ring %d B (the fit plan prices %d B), a drop of %d B (%.1f%%)",
		n, a.SlidingWindow, forever, ring, kvBytesForCtx(a, n, false, false), forever-ring, 100*float64(forever-ring)/float64(forever))
}
