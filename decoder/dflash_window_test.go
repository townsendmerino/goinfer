package decoder

import (
	"slices"
	"strings"
	"testing"
)

// TestDFlashMasks_config: each layer's mask from the config, as z-lab's reference reads it (Qwen3DFlashAttention.__init__).
func TestDFlashMasks_config(t *testing.T) {
	w, f, tr := 4096, false, true
	sl, fu := "sliding_attention", "full_attention"
	for _, c := range []struct {
		name   string
		cfg    dflashConfig
		causal []bool
		window []int
		err    string
	}{
		{"no layer types: every layer full", dflashConfig{NumHiddenLayers: 2}, nil, nil, ""},
		{"z-lab's retrained drafters", dflashConfig{NumHiddenLayers: 3, LayerTypes: []string{sl, sl, fu}, SlidingWindow: &w},
			[]bool{true, true, false}, []int{4096, 4096, 0}, ""},
		{"is_causal false: windowed, not causal", dflashConfig{NumHiddenLayers: 2, LayerTypes: []string{sl, fu}, SlidingWindow: &w, IsCausal: &f},
			[]bool{false, false}, []int{4096, 0}, ""},
		{"is_causal true: every layer causal", dflashConfig{NumHiddenLayers: 2, LayerTypes: []string{sl, fu}, SlidingWindow: &w, IsCausal: &tr},
			[]bool{true, true}, []int{4096, 0}, ""},
		{"a type per layer", dflashConfig{NumHiddenLayers: 3, LayerTypes: []string{sl, fu}, SlidingWindow: &w}, nil, nil, "layer_types"},
		{"a sliding layer needs a window", dflashConfig{NumHiddenLayers: 1, LayerTypes: []string{sl}}, nil, nil, "sliding_window"},
		{"an unknown type", dflashConfig{NumHiddenLayers: 1, LayerTypes: []string{"chunked_attention"}}, nil, nil, "layer type"},
	} {
		causal, window, err := dflashMasks(c.cfg)
		if c.err != "" {
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: err %v, want one naming %q", c.name, err, c.err)
			}
			continue
		}
		if err != nil || !slices.Equal(causal, c.causal) || !slices.Equal(window, c.window) {
			t.Errorf("%s: causal %v window %v err %v, want %v %v", c.name, causal, window, err, c.causal, c.window)
		}
	}
}

// TestDFlashTrunk_maskProperties checks what each mask means on the tiny drafter, without restating its formula: with causal
// layers a block row's output ignores later block rows, where full layers mix them; with a window of 3, context keys outside
// every row's window change nothing, and a visible one does.
func TestDFlashTrunk_maskProperties(t *testing.T) {
	d, err := LoadDFlashDrafter(writeTinyDFlash(t, false))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	const nCtx, nBlk = 8, 4
	be := &cpuBackend{}
	rows := func(seed, n int) [][]float32 {
		v := dflashWindowHashed(seed, n*d.hidden)
		out := make([][]float32, n)
		for i := range out {
			out[i] = v[i*d.hidden : (i+1)*d.hidden]
		}
		return out
	}
	run := func(fused, block [][]float32) [][]float32 {
		t.Helper()
		out, err := d.DraftBlock(be, fused, block)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	same := func(a, b []float32) bool { return slices.Equal(a, b) }
	fused, block := rows(1, nCtx), rows(2, nBlk)
	later := slices.Clone(block)
	later[nBlk-1] = rows(3, 1)[0] // a different last block row

	d.layerCausal, d.layerWindow = []bool{true, true}, []int{0, 0} // causal, no window
	a, b := run(fused, block), run(fused, later)
	if !same(a[0], b[0]) || !same(a[nBlk-2], b[nBlk-2]) {
		t.Error("causal layers: an earlier block row's output moved with the last row's input")
	}
	d.layerCausal, d.layerWindow = nil, nil // every layer full
	if a, b := run(fused, block), run(fused, later); same(a[0], b[0]) {
		t.Error("full layers: the first row's output ignored the last row; the property above would pass vacuously")
	}

	d.layerCausal, d.layerWindow = []bool{true, true}, []int{3, 3}
	// Block row 0 sits at position nCtx and sees keys down to nCtx-2; every later row sees less of the context.
	old, near := slices.Clone(fused), slices.Clone(fused)
	old[nCtx-3] = rows(4, 1)[0] // outside every row's window
	near[nCtx-1] = rows(4, 1)[0]
	base := run(fused, block)
	if o := run(old, block); !slices.EqualFunc(base, o, same) {
		t.Error("window 3: a context key outside every row's window changed the output")
	}
	if n := run(near, block); same(base[0], n[0]) {
		t.Error("window 3: a context key inside row 0's window did not change its output")
	}
}

// dflashWindowHashed is scripts/pin_dflash_window_trace.py's hashed(): n floats in [-0.5, 0.5], the same on both sides.
func dflashWindowHashed(seed, n int) []float32 {
	out := make([]float32, n)
	const m = 0xFFFFFFFF
	for i := range out {
		h := (uint64(i) + uint64(seed)*0x9E3779B9) * 2654435761 & m
		h ^= h >> 15
		h = h * 2246822519 & m
		h ^= h >> 13
		out[i] = float32(float64(h%2001)/2000.0 - 0.5)
	}
	return out
}
