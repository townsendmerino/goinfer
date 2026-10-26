package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// KS-O4's G-RING (docs/measurements/ks-o4-2026-10/README.md): gpt-oss's window layers read and write the KV ring (Step 2) and give the
// same logits, word for word, as the same attention over append-forever rows (Step 1), on the tiny fixture, at a context past its window,
// over 64 greedy steps. The ring holds only the window; the append-forever cache holds every row; attention reads the same rows in the
// same order from either.
func TestKSO4_ringEqualsAppendForever(t *testing.T) {
	m, err := Load("testdata/gptoss_tiny.gguf", Options{DirectLoad: true})
	if err != nil {
		t.Fatalf("load the tiny gpt-oss fixture: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	if a.gptoss == nil || a.SlidingWindow <= 0 || !a.kvRingsOK() {
		t.Fatalf("the fixture must be a windowed gpt-oss with rings admitted: gptoss %v, window %d, rings %v", a.gptoss != nil, a.SlidingWindow, a.kvRingsOK())
	}
	rings := 0
	for l := range a.NumLayers {
		if a.ringLayerAt(l) {
			rings++
		}
	}
	if rings == 0 {
		t.Fatal("no layer of the fixture is a ring layer: the test would compare nothing")
	}
	n := 2*a.SlidingWindow + 10 // past the window, so the ring has wrapped before the first compared step
	const steps = 64
	withRings := m.NewCache(n + steps)
	appendForever := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, n+steps, a.hasNoAttentionKVAt)
	appendForever.scr = newDecodeScratch(a)
	for l := range a.NumLayers {
		if (withRings.rings[l] != nil) != a.ringLayerAt(l) || appendForever.rings[l] != nil {
			t.Fatalf("layer %d: the two caches are not the two layouts compared (ring %v, append-forever ring %v)", l, withRings.rings[l] != nil, appendForever.rings[l] != nil)
		}
	}
	_, _, _, _, _, _, vocab := m.Dims()
	r := rand.New(rand.NewSource(7))
	id := 1 + r.Intn(vocab-1)
	for i := range n + steps {
		got, err := m.forward(id, withRings)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.forward(id, appendForever)
		if err != nil {
			t.Fatal(err)
		}
		for j := range want {
			if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				t.Fatalf("position %d: logit %d is %v with the ring, %v append-forever", i, j, got[j], want[j])
			}
		}
		if i < n {
			id = 1 + r.Intn(vocab-1) // the prompt
		} else {
			id = argmaxF32(want) // greedy
		}
	}
	t.Logf("%d ring layers of %d, window %d: %d positions (%d prompt, %d greedy), every logit word equal", rings, a.NumLayers, a.SlidingWindow, n+steps, n, steps)
}
