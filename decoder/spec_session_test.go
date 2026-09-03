package decoder

import (
	"context"
	"slices"
	"testing"
)

// TestSessionNgramSpecParity gates the Session-integrated speculative path: with a
// warm KV cache and prefix reuse, n-gram speculative output must stay token-
// identical to plain greedy — single-turn and across a second turn that reuses the
// first turn's prefix (the agent/chat regime the wiring targets).
func TestSessionNgramSpecParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const n = 24
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	for pi, prompt := range specPrompts {
		// Reference: plain greedy from a clean model (no session).
		ref := collectTokens(first(m.Generate(ctx, prompt, n, greedy)))

		// Turn 1: session speculative must match the reference.
		sess := m.NewSession(len(prompt) + 4*n)
		for _, mode := range []string{"fixed", "adaptive"} {
			s2 := m.NewSession(len(prompt) + 4*n)
			var got []int
			if mode == "fixed" {
				ch, g, err := s2.GenerateNgramSpeculative(ctx, prompt, n, &NgramDrafter{}, 8, greedy)
				if err != nil {
					t.Fatalf("prompt %d %s: %v", pi, mode, err)
				}
				got = collectTokens(ch)
				if g.Err() != nil {
					t.Fatalf("prompt %d %s: %v", pi, mode, g.Err())
				}
			} else {
				ch, g, err := s2.GenerateNgramSpeculativeAdaptive(ctx, prompt, n, &NgramDrafter{}, &AdaptiveDepth{MaxDraft: 8}, greedy)
				if err != nil {
					t.Fatalf("prompt %d %s: %v", pi, mode, err)
				}
				got = collectTokens(ch)
				if g.Err() != nil {
					t.Fatalf("prompt %d %s: %v", pi, mode, g.Err())
				}
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("prompt %d %s: session spec != plain greedy\n got %v\n ref %v", pi, mode, got, ref)
			}
		}

		// Turn 2: extend turn-1's output (prefix reuse) and compare against a fresh
		// plain greedy on the same extended prompt.
		ch, _, err := sess.GenerateNgramSpeculative(ctx, prompt, n, &NgramDrafter{}, 8, greedy)
		if err != nil {
			t.Fatalf("prompt %d turn1: %v", pi, err)
		}
		gen1 := collectTokens(ch)
		prompt2 := append(append(slices.Clone(prompt), gen1...), prompt[0]) // + one extra token
		ref2 := collectTokens(first(m.Generate(ctx, prompt2, n, greedy)))
		ch2, g2, err := sess.GenerateNgramSpeculative(ctx, prompt2, n, &NgramDrafter{}, 8, greedy)
		if err != nil {
			t.Fatalf("prompt %d turn2: %v", pi, err)
		}
		got2 := collectTokens(ch2)
		if g2.Err() != nil {
			t.Fatalf("prompt %d turn2: %v", pi, g2.Err())
		}
		if !slices.Equal(got2, ref2) {
			t.Fatalf("prompt %d turn2 (prefix reuse): session spec != plain greedy\n got %v\n ref %v", pi, got2, ref2)
		}
	}
}

// first adapts the (chan, *Generation) pair to just the channel for collectTokens.
func first(ch <-chan int, _ *Generation) <-chan int { return ch }

// TestSessionNgramAdaptiveThetaAtLeastOne_declinesToGenerate is TestNgramAdaptiveThetaAtLeastOne_declinesToGenerate's
// Session counterpart (P-16): Session.GenerateNgramSpeculativeAdaptive must also
// decline straight to Session.Generate when Theta >= 1, preserving warm-KV reuse.
func TestSessionNgramAdaptiveThetaAtLeastOne_declinesToGenerate(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const n = 16
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	prompt := specPrompts[0]

	ref := collectTokens(first(m.Generate(ctx, prompt, n, greedy)))

	cd := &countingDrafter{Drafter: &NgramDrafter{}}
	s := m.NewSession(len(prompt) + 4*n)
	ch, g, err := s.GenerateNgramSpeculativeAdaptive(ctx, prompt, n, cd, &AdaptiveDepth{MaxDraft: 8, Theta: 1}, greedy)
	if err != nil {
		t.Fatalf("GenerateNgramSpeculativeAdaptive: %v", err)
	}
	got := collectTokens(ch)
	if g.Err() != nil {
		t.Fatalf("stream err %v", g.Err())
	}
	if !slices.Equal(got, ref) {
		t.Fatalf("Theta>=1 output != plain Generate\n got %v\n ref %v", got, ref)
	}
	if cd.calls != 0 {
		t.Errorf("drafter.Draft called %d times, want 0 — Theta>=1 should decline before any n-gram scan", cd.calls)
	}

	if _, _, err := s.GenerateNgramSpeculativeAdaptive(ctx, prompt, n, nil, &AdaptiveDepth{Theta: 1}, greedy); err == nil {
		t.Error("nil drafter with Theta>=1: expected an error, got nil")
	}
}
