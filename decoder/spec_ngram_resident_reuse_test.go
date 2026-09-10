package decoder

import (
	"context"
	"testing"
)

// TestGenNgramInto_residentReusesWarmPrefix is P-05 (audit-2026-09-10): R-03 already made this
// function COMMIT the accepted sequence correctly on exit (residentCommitIDs), but nothing on
// ENTRY ever consulted what that commit left behind — every round cold-prefilled the whole
// prompt from position 0 regardless, so a --spec/--drafter agent loop got no prefix reuse despite
// committing one every round. This drives two rounds back to back: round 1 cold (nothing
// committed yet), round 2 with a prompt that STRICTLY EXTENDS round 1's own committed sequence
// (the real agent-turn shape residentReuseLen exists for) — and asserts round 2's
// Generation.PrefillReused is nonzero, the same observable proof generate_vl.go's own reuse
// tests use, not just that generation still produces output.
func TestGenNgramInto_residentReusesWarmPrefix(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.DecodeRunnerEligible() {
		t.Skip("fixture is not resident-decode-runner-eligible; the other seam tests still gate the wiring")
	}

	drafter := &NgramDrafter{}
	greedy := SamplingParams{Temperature: 0}
	ctx := context.Background()

	prompt1 := []int{1, 2, 3}
	ch1, g1, err := m.GenerateNgramSpeculative(ctx, prompt1, 6, drafter, 4, greedy)
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}
	got1 := collectTokens(ch1)
	if g1.Err() != nil {
		t.Fatalf("round 1: gen.Err() = %v", g1.Err())
	}
	if g1.PrefillReused != 0 {
		t.Errorf("round 1: PrefillReused = %d, want 0 (nothing committed yet)", g1.PrefillReused)
	}
	if len(got1) == 0 {
		t.Fatal("test setup: round 1 produced no tokens to extend the prompt with")
	}

	// Round 2's prompt is round 1's own committed sequence (prompt1 + everything it emitted) plus
	// one more token — a strict extension, the only shape residentReuseLen ever credits.
	prompt2 := append(append([]int(nil), prompt1...), got1...)
	prompt2 = append(prompt2, 99)

	ch2, g2, err := m.GenerateNgramSpeculative(ctx, prompt2, 6, drafter, 4, greedy)
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}
	for range ch2 {
	}
	if g2.Err() != nil {
		t.Fatalf("round 2: gen.Err() = %v", g2.Err())
	}
	if g2.PrefillReused == 0 {
		t.Errorf("round 2: PrefillReused = 0, want > 0 — a strict extension of round 1's own "+
			"committed sequence (len %d) should have reused a warm prefix instead of cold-prefilling "+
			"the whole prompt again", len(prompt1)+len(got1))
	}
	// residentReuseLen caps at len(prompt2)-1 by contract (prefill must cover at least one
	// token to seed decode from), so the bound here is "all but the capped tail", not an exact
	// "all of round 1" figure.
	if want := len(prompt1) + len(got1) - 1; g2.PrefillReused < want {
		t.Errorf("round 2: PrefillReused = %d, want >= %d (all but the one-token seed floor)", g2.PrefillReused, want)
	}
}
