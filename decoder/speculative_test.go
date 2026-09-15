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
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
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

// TestGenerateSpeculative_residentContextCapFinishesCleanly is M-03's (docs/audit-2026-09-10.md)
// own gate: verifying past the resident context cap must finish the generation cleanly instead of
// the backend's checkCap refusing the whole round with a hard error (specRoundDraftWidth,
// spec_ngram.go — shared with genNgramInto). capPos is set to exactly len(prompt)+1 so round 1
// deterministically lands on the kRound==0 boundary (room for `cur` alone, no draft token) — the
// case needing its own draft-cache-sync branch in speculative.go (`case allAccept:` with no
// draftTok[kRound-1] to feed, since none were drafted — `cur` itself was never fed to the draft's
// cache this round, and must be fed explicitly instead).
//
// fakeResident's logits are position-only (argmax at position p is always p%vocab, regardless of
// the fed token — decoder/resident_seam_test.go), so the whole token sequence is hand-computable
// here rather than needing a second live reference call (which would also double-use the same
// fakeResident's mutated position state).
func TestGenerateSpeculative_residentContextCapFinishesCleanly(t *testing.T) {
	target, be := loadWithFakeResident(t)
	draft, err := Load(tinyFixture(t), Options{})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}
	t.Cleanup(func() { _ = draft.Close() })

	prompt := []int{1, 2}
	be.rf.capPos = len(prompt) + 1 // exactly one position of headroom beyond the prompt
	vocab := be.rf.vocab

	ch, gen, err := target.GenerateSpeculative(context.Background(), prompt, 100, draft, 4, SamplingParams{Temperature: 0})
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	got := collectTokens(ch)
	if err := gen.Err(); err != nil {
		t.Fatalf("resident overran the cap instead of finishing cleanly: %v", err)
	}

	// cur (prefill's seed) = argmax at position len(prompt)-1 = (len(prompt)-1)%vocab.
	// Round 1 verifies just cur at position len(prompt) (kRound==0) — its argmax becomes the
	// next pending token, emitted before the loop re-checks the cap and stops.
	want := []int{(len(prompt) - 1) % vocab, len(prompt) % vocab}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v (hand-computed from fakeResident's position-only logits)", got, want)
	}
	if be.rf.forwards != be.rf.capPos {
		t.Errorf("resident forwards = %d, want %d (prefill %d + the one kRound==0 verify)",
			be.rf.forwards, be.rf.capPos, len(prompt))
	}
}

// TestSpeculativeGreedyParity_draftTarget runs the real pair — 1.5B target, 0.5B
// draft — exercising the mismatch/correction path. Output must STILL be
// token-identical to plain 1.5B greedy (the target's distribution is preserved
// regardless of draft quality). Skips unless GOINFER_SPEC_TARGET points at the
// 1.5B gguf.
func TestSpeculativeGreedyParity_draftTarget(t *testing.T) {
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		t.Skip("set GOINFER_SPEC_TARGET to the 1.5B gguf to run the draft≠target gate")
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
