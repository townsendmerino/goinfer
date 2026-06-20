package decoder

import (
	"context"
	"slices"
	"testing"
)

// TestNgramDrafter exercises the prompt-lookup logic directly — no model needed,
// so it always runs. Longest-suffix, most-recent-occurrence, capped at k.
func TestNgramDrafter(t *testing.T) {
	d := &NgramDrafter{} // defaults: MinMatch 2, MaxMatch 16
	cases := []struct {
		name string
		ctx  []int
		k    int
		want []int
	}{
		{"longest suffix wins", []int{1, 2, 3, 4, 1, 2, 3}, 2, []int{4, 1}},
		{"no match", []int{5, 6, 7}, 4, nil},
		{"repeat run", []int{9, 9, 9}, 1, []int{9}},
		{"cap at k", []int{1, 2, 3, 9, 9, 9, 1, 2, 3}, 2, []int{9, 9}},
		{"most recent occurrence", []int{1, 2, 0, 7, 1, 2, 8, 1, 2}, 1, []int{8}},
		{"too short for minmatch", []int{1, 2}, 4, nil},
		{"k=0 yields nothing", []int{1, 2, 3, 1, 2, 3}, 0, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := d.Draft(tc.ctx, tc.k)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("Draft(%v, %d) = %v, want %v", tc.ctx, tc.k, got, tc.want)
			}
		})
	}
}

// TestNgramSpeculativeGreedyParity is THE gate for the n-gram spoke: single-model
// n-gram speculative output must be token-identical to plain greedy, for every K
// and prompt — losslessly, regardless of how well the drafter guesses. Misses
// degenerate to plain decode; hits commit multiple tokens per pass. Acceptance is
// data-dependent here (unlike the same-model draft case), so we assert only
// correctness, not a rate.
func TestNgramSpeculativeGreedyParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	const n = 32
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	drafter := &NgramDrafter{}

	for pi, prompt := range specPrompts {
		refCh, _ := m.Generate(ctx, prompt, n, greedy)
		ref := collectTokens(refCh)
		for _, K := range []int{1, 4, 8} {
			ch, g, err := m.GenerateNgramSpeculative(ctx, prompt, n, drafter, K, greedy)
			if err != nil {
				t.Fatalf("prompt %d K=%d: %v", pi, K, err)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("prompt %d K=%d: stream err %v", pi, K, g.Err())
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("prompt %d K=%d: n-gram speculative != greedy\n got %v\n ref %v", pi, K, got, ref)
			}
		}
	}
}
