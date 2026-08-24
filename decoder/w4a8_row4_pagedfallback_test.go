package decoder

import "testing"

// TestSerializedInt4Weights_neverRepacked_pagedFallback proves the plumbing
// brief's paged-MoE carve-out (docs/prompts/w4a8-plumbing.md, docs/task-w4a8-
// neon-bandwidth.md) BY TEST rather than by construction argument alone: a
// .giw-loaded (zero-copy mmap-alias-shaped, via SerializeWeights/
// LoadSerializedWeights — the same round-trip path real .giw loading uses)
// int4 model must have NO repacked row4 layout on any weight (repackW4A8Row4
// IfEligible is wired only into streamQuantized/quantizeWM, the GGUF/
// safetensors streaming paths — never into the .giw loader), and decode
// through the resulting fallback-only dispatch must still produce output
// IDENTICAL to a model whose weights WERE eligible for the row4 repack. That
// second half is the real proof: it's not enough that the paged path takes a
// different branch, it has to be the SAME branch MatmulBTW4A8Row4Into's own
// fallback already is (docs/task-w4a8-neon-bandwidth.md's bit-identity
// finding), cross-checked here end-to-end through a real model rather than
// only at the raw-kernel level.
func TestSerializedInt4Weights_neverRepacked_pagedFallback(t *testing.T) {
	path := prequantGGUF(t)

	m1, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (int4, row4-eligible): %v", err)
	}
	var repackedCount, int4Count int
	for _, wm := range m1.w.matmulWeights() {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		int4Count++
		if _, _, ok := wm.Int4Row4(); ok {
			repackedCount++
		}
	}
	if int4Count == 0 {
		t.Fatal("fixture has no int4 weights at all — test is not exercising anything")
	}
	t.Logf("GGUF-loaded (streamQuantized/quantizeWM path): %d/%d int4 weights repacked to row4", repackedCount, int4Count)

	blob, err := SerializeWeights(m1.w, "row4-paged-fallback-test")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize (.giw path): %v", err)
	}

	for i, wm := range w2.matmulWeights() {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		if _, _, ok := wm.Int4Row4(); ok {
			t.Fatalf("weight %d: .giw-loaded int4 tensor was repacked — the paged-MoE carve-out is broken (a resident row4 copy would pin RAM for a tensor paging expects to release)", i)
		}
	}
	t.Log(".giw-loaded (LoadSerializedWeights path): 0 int4 weights repacked, as required")

	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model from .giw-path weights: %v", err)
	}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	tok1 := greedyFirst(t, m1, prompt)
	tok2 := greedyFirst(t, m2, prompt)
	if tok1 != tok2 {
		t.Fatalf("greedy token differs between row4-eligible (%d) and paged-fallback-only (%d) dispatch — the two branches are not equivalent", tok1, tok2)
	}
	t.Logf("identical greedy token across row4 and fallback-only dispatch: %d", tok1)
}
