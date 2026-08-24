package decoder

import "testing"

// TestSerializedInt4Weights_row4Kind_matchesCanonical is TestSerializedInt4Weights_
// neverRepacked_pagedFallback's counterpart for the opt-in path: a .giw written via
// SerializeWeightsRow4 (weightMat kind 4, docs/task-w4a8-neon-bandwidth.md's "Format
// follow-on") must round-trip with Int4Row4() populated for every eligible int4
// tensor — the whole point of cutting the kind — and decode must be byte-identical
// to both (a) the same model loaded straight from GGUF (row4 built in RAM) and
// (b) a plain kind-3 .giw of the same weights (no row4 at all). Three paths,
// one answer, per this campaign's own bit-identity standard
// (TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical) — this is a storage
// format, never a numerics event.
func TestSerializedInt4Weights_row4Kind_matchesCanonical(t *testing.T) {
	path := prequantGGUF(t)

	mGGUF, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (int4, row4-eligible): %v", err)
	}
	var eligible, int4Count int
	for _, wm := range mGGUF.w.matmulWeights() {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		int4Count++
		if _, _, ok := wm.Int4Row4(); ok {
			eligible++
		}
	}
	if int4Count == 0 {
		t.Fatal("fixture has no int4 weights at all — test is not exercising anything")
	}
	if eligible == 0 {
		t.Skip("no int4 weight in this fixture is row4-eligible (shape/DotProd) — nothing for kind 4 to carry")
	}
	t.Logf("GGUF-loaded: %d/%d int4 weights row4-eligible", eligible, int4Count)

	// Kind 3 (default): the existing, unaffected path.
	blob3, err := SerializeWeights(mGGUF.w, "row4-kind-test-k3")
	if err != nil {
		t.Fatalf("SerializeWeights (kind 3): %v", err)
	}
	w3, err := LoadSerializedWeights(blob3)
	if err != nil {
		t.Fatalf("LoadSerializedWeights (kind 3): %v", err)
	}
	for i, wm := range w3.matmulWeights() {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		if _, _, ok := wm.Int4Row4(); ok {
			t.Fatalf("weight %d: kind-3 .giw carries a row4 layout — SerializeWeights must never emit kind 4", i)
		}
	}

	// Kind 4 (opt-in): every eligible tensor round-trips WITH Int4Row4() populated.
	blob4, err := SerializeWeightsRow4(mGGUF.w, "row4-kind-test-k4")
	if err != nil {
		t.Fatalf("SerializeWeightsRow4: %v", err)
	}
	if len(blob4) <= len(blob3) {
		t.Fatalf("kind-4 bundle (%d bytes) should be larger than kind-3 (%d bytes) — it carries both layouts", len(blob4), len(blob3))
	}
	w4, err := LoadSerializedWeights(blob4)
	if err != nil {
		t.Fatalf("LoadSerializedWeights (kind 4): %v", err)
	}
	srcWeights := mGGUF.w.matmulWeights()
	gotWeights := w4.matmulWeights()
	if len(srcWeights) != len(gotWeights) {
		t.Fatalf("matmulWeights() count changed across round-trip: src=%d got=%d", len(srcWeights), len(gotWeights))
	}
	var repackedOnLoad int
	for i, wm := range gotWeights {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		_, _, srcEligible := srcWeights[i].Int4Row4()
		_, _, gotOK := wm.Int4Row4()
		if srcEligible && !gotOK {
			t.Fatalf("weight %d: row4-eligible at GGUF load but kind-4 .giw round-trip lost it", i)
		}
		if !srcEligible && gotOK {
			t.Fatalf("weight %d: NOT row4-eligible at GGUF load but kind-4 .giw round-trip added it anyway", i)
		}
		if gotOK {
			repackedOnLoad++
		}
	}
	if repackedOnLoad != eligible {
		t.Fatalf("kind-4 .giw: %d/%d int4 weights carry Int4Row4() after round-trip, want %d (all eligible ones)", repackedOnLoad, int4Count, eligible)
	}
	t.Logf(".giw kind-4 round-trip: %d/%d int4 weights carry Int4Row4(), as expected", repackedOnLoad, int4Count)

	mK3, err := NewModel(w3, "cpu")
	if err != nil {
		t.Fatalf("new model from kind-3 weights: %v", err)
	}
	mK4, err := NewModel(w4, "cpu")
	if err != nil {
		t.Fatalf("new model from kind-4 weights: %v", err)
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	tokGGUF := greedyFirst(t, mGGUF, prompt)
	tokK3 := greedyFirst(t, mK3, prompt)
	tokK4 := greedyFirst(t, mK4, prompt)
	if tokGGUF != tokK3 || tokGGUF != tokK4 {
		t.Fatalf("greedy token differs across dispatch paths: gguf(row4-in-RAM)=%d kind3(fallback-only)=%d kind4(row4-from-disk)=%d",
			tokGGUF, tokK3, tokK4)
	}
	t.Logf("identical greedy token across GGUF/kind-3/kind-4 dispatch: %d", tokGGUF)
}
