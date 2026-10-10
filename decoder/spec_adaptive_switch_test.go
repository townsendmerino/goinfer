package decoder

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"
)

// specAdaptiveTurn runs one turn of an adaptive -spec-adaptive conversation and returns its ids and PrefillReused.
func specAdaptiveTurn(t *testing.T, m *Model, prompt []int, maxTok int) mc3Turn {
	t.Helper()
	ch, gen, err := m.GenerateNgramSpeculative(context.Background(), prompt, maxTok, &NgramDrafter{}, 4, SamplingParams{})
	if err != nil {
		t.Fatalf("GenerateNgramSpeculative: %v", err)
	}
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("adaptive spec generation: %v", err)
	}
	return mc3Turn{ids, gen.PrefillReused}
}

// specAdaptiveConversation plays `turns` turns of an adaptive spec conversation, threading each turn's prompt from
// the last exactly as mc3Conversation does for plain Generate — the two must be comparable turn for turn.
func specAdaptiveConversation(t *testing.T, m *Model, c, turns, maxTok int) []mc3Turn {
	t.Helper()
	vocab := m.w.arch.VocabSize
	prompt := []int{1, 2, 3, (c*17 + 5) % vocab, (c*29 + 9) % vocab, (c*31 + 3) % vocab}
	var out []mc3Turn
	for tn := range turns {
		turn := specAdaptiveTurn(t, m, prompt, maxTok)
		out = append(out, turn)
		prompt = append(append(slices.Clone(prompt), turn.ids...), (c*7+tn)%vocab, (c*11+tn)%vocab)
	}
	return out
}

// TestSpecAdaptiveSwitch_matchesPlainDecode is the MC4 candidate's Step 0.1 (docs/tasks/task-concurrency-2026-09.md):
// a -spec-adaptive generation forced through spec -> batch -> spec transitions by 1-3 concurrent plain generations
// emits EXACTLY plain decode's ids, every turn — greedy, so this is exact equality, not a tolerance. mc3Fake panics
// on any resident access that escapes its guard (enter/leave), so a synchronization mistake in the switch fails this
// test directly, not just a numeric mismatch.
func TestSpecAdaptiveSwitch_matchesPlainDecode(t *testing.T) {
	const slots, turns, maxTok = 4, 2, 40
	// Reference: the same conversation, adaptive OFF, entirely alone — plain lossless spec decode's own contract
	// already proves this equals Model.Generate (TestNgramSpeculativeGreedyParity); this is the baseline the
	// concurrent, adaptive run must reproduce exactly.
	mAlone, _ := loadWithMC3Fake(t, slots)
	alone := specAdaptiveConversation(t, mAlone, 0, turns, maxTok)

	m, rf := loadWithMC3Fake(t, slots)
	rf.delay = 3 * time.Millisecond // forces real interleaving, per TestMC3_concurrentGenerationsMatchAlone's own note
	if got := m.EnableResidentConcurrency(slots); got != slots {
		t.Fatalf("EnableResidentConcurrency(%d) = %d", slots, got)
	}
	m.SetSpecAdaptive(true)

	var wg sync.WaitGroup
	var got []mc3Turn
	wg.Go(func() {
		got = specAdaptiveConversation(t, m, 0, turns, maxTok)
	})
	// 1-3 plain concurrent generations, staggered short/long so the spec conversation is sometimes alone
	// (claimExclusive succeeds, drafts) and sometimes not (yields), inside its own maxTok=40 lifetime.
	for c := 1; c <= 3; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			mc3Conversation(t, m, c, turns, maxTok-10*(c-1)) // 30, 20, 10 tokens: they finish and rejoin at different times
		}(c)
	}
	wg.Wait()

	if len(got) != len(alone) {
		t.Fatalf("got %d turns, want %d", len(got), len(alone))
	}
	for tn := range alone {
		a, b := alone[tn], got[tn]
		if !slices.Equal(a.ids, b.ids) {
			t.Errorf("turn %d: adaptive-concurrent %v, plain-alone %v", tn, b.ids, a.ids)
		}
		if a.reused != b.reused {
			t.Errorf("turn %d: adaptive-concurrent reused %d, plain-alone reused %d", tn, b.reused, a.reused)
		}
	}
	if rf.steps == 0 {
		t.Errorf("StepBatch never ran — the spec conversation never actually yielded into a shared batch, so this run did not exercise the switch")
	}
	t.Logf("batched steps %d serving %d tokens (largest %d); solo forwards %d", rf.steps, rf.stepSeqs, rf.maxStep, rf.soloFwds)
}

// TestSpecAdaptiveSwitch_offByDefaultUnaffected: with -spec-adaptive off, a spec generation's own exclusive claim goes
// through claimExclusive (holders == 0), not a bare CAS on resBusy: a bare CAS could succeed at the same moment an MC3
// holder, admitted via bt.claim, was mid-step (bt.claim only READS resBusy, it never sets it). Production never creates
// this combination (-spec without -spec-adaptive forces concurrency to 1 in openai.go's setConcurrency), but the test
// constructs it directly to prove the guard does not depend on that.
//
// Two outcomes are both correct: the spec generation wins the claim (resident, mc3Fake's path) or loses to an
// already-admitted holder and falls back to the staged CPU path (the tiny fixture's real weights, a different
// computation, not a degraded one). That is the M9 guard's contract: both still complete correctly, only the loser
// loses resident speed. NOT compared against an always-resident reference: unlike matchesPlainDecode, a mid-run
// fallback to a different code path is not expected to be id-identical to the path it did not take (two distinct valid
// greedy paths).
func TestSpecAdaptiveSwitch_offByDefaultUnaffected(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 4)
	rf.delay = 3 * time.Millisecond
	m.EnableResidentConcurrency(4)
	// SetSpecAdaptive intentionally not called.
	var wg sync.WaitGroup
	var got mc3Turn
	wg.Go(func() {
		got = specAdaptiveTurn(t, m, []int{1, 2, 3, 4, 5, 6}, 24)
	})
	for c := 1; c <= 2; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			mc3Conversation(t, m, c, 1, 24)
		}(c)
	}
	wg.Wait()
	if len(got.ids) != 24 {
		t.Fatalf("spec generation produced %d ids, want 24 (a complete, valid generation either way it resolved)", len(got.ids))
	}
	t.Logf("StepBatch ran %d times (0 = spec won the exclusive claim; >0 = it lost and fell back, others batched)", rf.steps)
}
