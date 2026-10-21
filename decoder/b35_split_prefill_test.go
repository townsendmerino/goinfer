package decoder

import (
	"context"
	"math"
	"testing"
)

// TestQwen35SplitPrefill_bitIdentical is B35's first question on the CPU (docs/queue-engineering.md, B35, step 5): a Gated-DeltaNet
// prompt prefilled in two batched passes (runLayersQwen35N), [0,b) then [b,n), must leave exactly what one pass over [0,n) leaves:
// the last row's final-norm hidden state, every DeltaNet layer's recurrent state and conv window, and every full-attention layer's
// K/V rows. A restore at b is exact only if this holds. b runs over a cut inside the conv window, a mid-prompt cut, and B35's own
// shape (a 4-token tail, the empty think block). The fan-out is forced and counted, so the batched path cannot pass by never running.
func TestQwen35SplitPrefill_bitIdentical(t *testing.T) {
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Skipf("no qwen35-tiny fixture: %v", err)
	}
	defer m.Close()
	defer func(a int) { deltaNetMinWork = a }(deltaNetMinWork)
	deltaNetMinWork = 2

	const n = 61
	vocab := m.w.arch.VocabSize
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 5) % vocab
	}
	type arm struct {
		last  []float32
		cache *KVCache
	}
	prefill := func(cuts ...int) arm {
		cache := m.NewCache(n + 8)
		hidden := m.w.arch.HiddenDim
		var last []float32
		from := 0
		for _, to := range append(cuts, n) {
			if !m.qwen35BatchN(to-from, cache) {
				t.Fatalf("[%d,%d) cannot take the batched Qwen3.5 path", from, to)
			}
			hN, err := m.runLayersQwen35N(context.Background(), m.embedN(ids[from:to]), cache)
			if err != nil {
				t.Fatalf("runLayersQwen35N [%d,%d): %v", from, to, err)
			}
			last = append([]float32(nil), hN[(to-from-1)*hidden:]...)
			from = to
		}
		return arm{last, cache}
	}
	bits := func(t *testing.T, what string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: length %d vs %d", what, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s[%d]: one pass %v vs split %v", what, i, a[i], b[i])
			}
		}
	}
	before := deltaNetFanouts.Load()
	ref := prefill()
	if deltaNetFanouts.Load() == before {
		t.Fatal("the one-pass prefill never forked the DeltaNet fan-out: the batched path did not run")
	}
	for _, b := range []int{2, 30, n - 4} {
		got := prefill(b)
		bits(t, "last-row hidden state", ref.last, got.last)
		deltas, attn := 0, 0
		for l := range m.w.arch.NumLayers {
			if st := ref.cache.delta[l]; st != nil {
				deltas++
				gs := got.cache.delta[l]
				bits(t, "DeltaNet state", st.s, gs.s)
				if len(st.convWin) != len(gs.convWin) {
					t.Fatalf("b=%d layer %d: conv window %d rows vs %d", b, l, len(st.convWin), len(gs.convWin))
				}
				for i := range st.convWin {
					bits(t, "conv window", st.convWin[i], gs.convWin[i])
				}
				continue
			}
			if k := ref.cache.Keys(l); len(k) > 0 {
				attn++
				bits(t, "K rows", k, got.cache.Keys(l))
				bits(t, "V rows", ref.cache.Vals(l), got.cache.Vals(l))
			}
		}
		if deltas == 0 || attn == 0 {
			t.Fatalf("b=%d compared %d DeltaNet and %d attention layers: the fixture must have both", b, deltas, attn)
		}
	}
}
