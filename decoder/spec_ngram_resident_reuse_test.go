package decoder

import (
	"context"
	"testing"
)

// TestGenNgramInto_residentReusesWarmPrefix is P-05 (audit-2026-09-10): genNgramInto commits the accepted sequence on
// exit (residentCommitIDs), so on ENTRY it must consult what that commit left behind instead of cold-prefilling the
// whole prompt from position 0 every round (a --spec/--drafter agent loop would otherwise get no prefix reuse). This
// drives two rounds back to back: round 1 cold, round 2 with a prompt that STRICTLY EXTENDS round 1's own committed
// sequence (the agent-turn shape residentReuseLen exists for), and asserts round 2's Generation.PrefillReused is
// nonzero, the same observable proof generate_vl.go's own reuse tests use, not just that generation still produces
// output.
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

// TestGenNgramInto_residentCommitMatchesPlain: a speculative generation that ends by reaching maxTokens leaves the
// resident holding exactly what plain decode leaves: prompt + every emitted token (plain decode forwards each token it
// emits, the last included). A round's trailing token forwarded only as the next round's seq[0] would make the next turn
// reuse one position less and re-prefill it; on Metal that one re-prefilled position (f16-MMA prefill, not
// bit-identical to decode) changes every later turn's output (docs/measurements/spec-vs-batching-metal-2026-09-27.md
// §4). Asserted as equal reuse on the next turn, spec against plain, at five lengths; every case ends at the
// trailing-token exit. The exit after a streamed draft token is not exercised here; its tokens are forwarded by the
// verify already.
func TestGenNgramInto_residentCommitMatchesPlain(t *testing.T) {
	greedy := SamplingParams{Temperature: 0}
	ctx := context.Background()
	prompt1 := []int{1, 2, 3, 1, 2, 3, 1, 2}
	for _, maxTok := range []int{1, 2, 5, 6, 9} {
		reuse := func(spec bool) (int, []int) {
			m, _ := loadWithFakeResident(t)
			if !m.DecodeRunnerEligible() {
				t.Skip("fixture is not resident-decode-runner-eligible")
			}
			var ch <-chan int
			var g *Generation
			var err error
			if spec {
				ch, g, err = m.GenerateNgramSpeculative(ctx, prompt1, maxTok, &NgramDrafter{}, 4, greedy)
				if err != nil {
					t.Fatalf("spec: %v", err)
				}
			} else {
				ch, g = m.Generate(ctx, prompt1, maxTok, greedy)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("round 1: %v", g.Err())
			}
			prompt2 := append(append(append([]int(nil), prompt1...), got...), 99, 98)
			ch2, g2 := m.Generate(ctx, prompt2, 2, greedy)
			for range ch2 {
			}
			if g2.Err() != nil {
				t.Fatalf("round 2: %v", g2.Err())
			}
			return g2.PrefillReused, got
		}
		plainReuse, plainIDs := reuse(false)
		specReuse, specIDs := reuse(true)
		if len(plainIDs) != len(specIDs) {
			t.Fatalf("maxTokens %d: plain emitted %d tokens, spec %d", maxTok, len(plainIDs), len(specIDs))
		}
		if specReuse != plainReuse {
			t.Errorf("maxTokens %d: the next turn reused %d positions after spec, %d after plain decode — spec left the "+
				"resident holding a different prefix", maxTok, specReuse, plainReuse)
		}
	}
}
