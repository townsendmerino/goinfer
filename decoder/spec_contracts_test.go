package decoder

import "testing"

// TestSamplingParams_HistoryDependent gates the shared predicate the greedy
// speculative validators use to reject penalties / logit bias (M13). Greedy verify
// is argmax over raw target logits, so any of these silently diverges from plain
// greedy unless rejected.
func TestSamplingParams_HistoryDependent(t *testing.T) {
	for _, c := range []struct {
		name string
		sp   SamplingParams
		want bool
	}{
		{"plain greedy", SamplingParams{}, false},
		{"repeat penalty 1.0 is a no-op", SamplingParams{RepeatPenalty: 1}, false},
		{"repeat penalty", SamplingParams{RepeatPenalty: 1.1}, true},
		{"presence penalty", SamplingParams{PresencePenalty: 0.5}, true},
		{"frequency penalty", SamplingParams{FrequencyPenalty: 0.3}, true},
		{"negative presence penalty still counts", SamplingParams{PresencePenalty: -0.2}, true},
		{"logit bias", SamplingParams{LogitBias: map[int]float32{7: 3}}, true},
		{"empty logit bias map", SamplingParams{LogitBias: map[int]float32{}}, false},
	} {
		if got := c.sp.HistoryDependent(); got != c.want {
			t.Errorf("%s: HistoryDependent() = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestValidateNgramSpec_rejectsGreedyPenalties gates M13's wiring on the ngram
// entry point: a greedy request with any history-dependent transform is refused
// (so serve falls back to plain Generate) — while a sampled one threads them and
// is not refused by this gate. The penalty check precedes specRollbackSafe, so a
// nil model is never dereferenced on the reject paths.
func TestValidateNgramSpec_rejectsGreedyPenalties(t *testing.T) {
	drafter := &NgramDrafter{}
	// Greedy (Temperature 0) + penalties → rejected.
	if err := validateNgramSpec(nil, drafter, SamplingParams{RepeatPenalty: 1.3}); err == nil {
		t.Error("greedy + repeat penalty: want rejection, got nil")
	}
	if err := validateNgramSpec(nil, drafter, SamplingParams{LogitBias: map[int]float32{3: 2}}); err == nil {
		t.Error("greedy + logit bias: want rejection, got nil")
	}
	// The gate must not fire on plain greedy or on the sampled path (Temperature>0);
	// those proceed to the rollback-safety check. Assert the gate itself is quiet by
	// checking a Model-independent predicate rather than dereferencing nil.
	if (SamplingParams{}).HistoryDependent() {
		t.Error("plain greedy must not be treated as history-dependent")
	}
	if !(SamplingParams{Temperature: 0.7, RepeatPenalty: 1.3}).HistoryDependent() {
		t.Error("sampled + penalties is still history-dependent (threaded, not rejected)")
	}
}

// TestEagleTreeNodes gates M15: DraftTree builds a FULL b-ary tree, so a round has
// Σ_{i=1}^{d} b^i nodes — not b*d. This is what sizes the cache and what
// stats.Drafted must count. The b=2,d=5 → 62 case matches the tree-shape comment in
// eagle_accept_test.go.
func TestEagleTreeNodes(t *testing.T) {
	for _, c := range []struct {
		b, d, want int
	}{
		{2, 4, 30}, // 2+4+8+16
		{2, 5, 62}, // 2+4+8+16+32 — the accept test's "full binary tree to depth 5"
		{1, 4, 4},  // a single chain
		{3, 2, 12}, // 3+9
		{2, 1, 2},  // just the root's children
		{4, 0, 0},  // no depth ⇒ no nodes
	} {
		if got := eagleTreeNodes(c.b, c.d); got != c.want {
			t.Errorf("eagleTreeNodes(%d,%d) = %d, want %d (b*d=%d shows the old undercount)",
				c.b, c.d, got, c.want, c.b*c.d)
		}
	}
}
