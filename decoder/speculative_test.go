package decoder

import (
	"context"
	"os"
	"slices"
	"testing"
)

func collectTokens(ch <-chan int) []int {
	var s []int
	for t := range ch {
		s = append(s, t)
	}
	return s
}

var specPrompts = [][]int{
	{785, 264, 6573, 311, 1438, 279, 2038, 25},
	{750, 1438, 4136, 3932, 262, 671},
	{2, 264, 729, 311, 11047, 279},
}

// TestSpeculativeGreedyParity is THE gate: greedy speculative output must be
// token-identical to plain target greedy. Using the same model as draft and
// target drives the all-accept + bonus path (and forwardN / TruncateTo) — the
// output must still exactly equal plain greedy for every K and prompt. K=1 also
// degenerates to plain decode.
func TestSpeculativeGreedyParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	const n = 32
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	for pi, prompt := range specPrompts {
		refCh, _ := m.Generate(ctx, prompt, n, greedy)
		ref := collectTokens(refCh)
		for _, K := range []int{1, 4, 8} {
			ch, g, err := m.GenerateSpeculative(ctx, prompt, n, m, K, greedy)
			if err != nil {
				t.Fatalf("prompt %d K=%d: %v", pi, K, err)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("prompt %d K=%d: stream err %v", pi, K, g.Err())
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("prompt %d K=%d: speculative != greedy\n got %v\n ref %v", pi, K, got, ref)
			}
			// Same model → the draft agrees everywhere → ~all accepted.
			if g.Spec != nil && g.Spec.AcceptanceRate() < 0.99 {
				t.Errorf("prompt %d K=%d: same-model acceptance %.3f, want ~1.0", pi, K, g.Spec.AcceptanceRate())
			}
		}
	}
}

// TestSpeculativeGreedyParity_draftTarget runs the real pair — 1.5B target, 0.5B
// draft — exercising the mismatch/correction path. Output must STILL be
// token-identical to plain 1.5B greedy (the target's distribution is preserved
// regardless of draft quality). Skips unless GINFER_SPEC_TARGET points at the
// 1.5B gguf.
func TestSpeculativeGreedyParity_draftTarget(t *testing.T) {
	tpath := os.Getenv("GINFER_SPEC_TARGET")
	if tpath == "" {
		t.Skip("set GINFER_SPEC_TARGET to the 1.5B gguf to run the draft≠target gate")
	}
	draft, err := loadBenchModel() // 0.5B
	if err != nil {
		t.Skipf("no draft model (%v)", err)
	}
	target, err := Load(tpath, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load target %s: %v", tpath, err)
	}
	const n = 40
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	for pi, prompt := range specPrompts {
		refCh, _ := target.Generate(ctx, prompt, n, greedy)
		ref := collectTokens(refCh)
		for _, K := range []int{1, 4, 8} {
			ch, g, err := target.GenerateSpeculative(ctx, prompt, n, draft, K, greedy)
			if err != nil {
				t.Fatalf("prompt %d K=%d: %v", pi, K, err)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("prompt %d K=%d: stream err %v", pi, K, g.Err())
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("prompt %d K=%d: speculative != target greedy\n got %v\n ref %v", pi, K, got, ref)
			}
			if g.Spec != nil {
				t.Logf("prompt %d K=%d: acceptance %.3f, %.2f tokens/round", pi, K, g.Spec.AcceptanceRate(), g.Spec.TokensPerRound())
			}
		}
	}
}
