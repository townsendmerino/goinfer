package decoder

import (
	"context"
	"testing"
)

// SamplingParams.LogitProcessorGate: a lazy processor must leave every on-device fast path alone
// while its gate is closed, and take the full-logits path (and run the processor) exactly on the
// steps it opens. Pinned through the real decode loop with a fake resident that counts which
// forward it was asked for — no GPU, the committed tiny fixture.

// fakeGreedyResident adds the on-device greedy fast path to fakeResident and counts it apart from
// full-logits forwards. Its argmax is the argmax of the same row Forward returns, so the fast path
// and the full path agree on the token — exactly the property the real backends guarantee.
type fakeGreedyResident struct {
	*fakeResident
	argmaxCalls int
}

func (f *fakeGreedyResident) ForwardArgmax(emb []float32, pos int) (int, error) {
	row, err := f.fakeResident.Forward(emb, pos)
	if err != nil {
		return 0, err
	}
	f.fakeResident.forwards-- // counted as an argmax call, not a full forward
	f.argmaxCalls++
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best, nil
}

type fakeGreedyBackend struct {
	Backend
	rf *fakeGreedyResident
}

func (b *fakeGreedyBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeGreedyResident{fakeResident: &fakeResident{vocab: vocab}}
	return b.rf, true, nil
}
func (b *fakeGreedyBackend) Close() error { return nil }

func loadWithFakeGreedy(t *testing.T) (*Model, *fakeGreedyBackend) {
	t.Helper()
	be := &fakeGreedyBackend{}
	RegisterBackend("fake-greedy-gate", func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: "fake-greedy-gate"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture not resident-eligible")
	}
	return m, be
}

func gateDrain(m *Model, sp SamplingParams, n int) []int {
	stream, _ := m.Generate(context.Background(), []int{1, 2, 3}, n, sp)
	var out []int
	for id := range stream {
		out = append(out, id)
	}
	return out
}

func TestLogitProcessorGate_closedKeepsFastPath_openTakesFullLogits(t *testing.T) {
	m, be := loadWithFakeGreedy(t)
	const n = 8
	// Reference: no processor at all — every decode step after the first uses ForwardArgmax.
	ref := gateDrain(m, SamplingParams{}, n)
	if be.rf.argmaxCalls == 0 {
		t.Fatal("the fake's greedy fast path was never used without a processor; the test cannot see anything")
	}

	// An UNGATED no-op processor: today's behaviour, fast path off for the whole turn.
	be.rf.argmaxCalls, be.rf.forwards = 0, 0
	calls := 0
	ungated := gateDrain(m, SamplingParams{LogitProcessor: func([]int, []float32) { calls++ }}, n)
	if be.rf.argmaxCalls != 0 || calls == 0 {
		t.Fatalf("ungated processor: %d argmax calls, %d processor calls — expected 0 and >0", be.rf.argmaxCalls, calls)
	}

	// A gated processor whose gate NEVER opens: identical to having no processor.
	be.rf.argmaxCalls, be.rf.forwards = 0, 0
	calls = 0
	closed := gateDrain(m, SamplingParams{
		LogitProcessor:     func([]int, []float32) { calls++ },
		LogitProcessorGate: func([]int) bool { return false },
	}, n)
	if calls != 0 {
		t.Errorf("closed gate: processor ran %d times", calls)
	}
	if be.rf.argmaxCalls == 0 {
		t.Error("closed gate: the greedy fast path was not used — the gate cost the fast path anyway")
	}

	// A gate that opens once 3 tokens exist: those steps take Forward and run the processor.
	be.rf.argmaxCalls, be.rf.forwards = 0, 0
	var sawLen []int
	opened := gateDrain(m, SamplingParams{
		LogitProcessor:     func(gen []int, _ []float32) { sawLen = append(sawLen, len(gen)) },
		LogitProcessorGate: func(gen []int) bool { return len(gen) >= 3 },
	}, n)
	if len(sawLen) == 0 || sawLen[0] != 3 {
		t.Errorf("open-at-3 gate: processor first saw %v generated tokens, want 3", sawLen)
	}
	if be.rf.argmaxCalls == 0 || be.rf.forwards == 0 {
		t.Errorf("open-at-3 gate: argmax=%d full=%d — want both paths used (fast before, full after)", be.rf.argmaxCalls, be.rf.forwards)
	}

	// A no-op processor must not change greedy output on any of the paths.
	for name, got := range map[string][]int{"ungated": ungated, "closed": closed, "opened": opened} {
		if len(got) != len(ref) {
			t.Fatalf("%s: %d tokens vs %d", name, len(got), len(ref))
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Errorf("%s: token %d = %d, want %d (a no-op processor changed greedy output)", name, i, got[i], ref[i])
			}
		}
	}
}

// The gate masks exactly the steps it opens: a processor that forbids everything but one token,
// behind a gate that opens after 2 tokens, must force that token from step 3 on and nowhere else.
func TestLogitProcessorGate_masksOnlyOpenSteps(t *testing.T) {
	m, _ := loadWithFakeGreedy(t)
	ref := gateDrain(m, SamplingParams{}, 6)
	const forced = 7
	got := gateDrain(m, SamplingParams{
		LogitProcessor: func(_ []int, logits []float32) {
			for i := range logits {
				if i != forced {
					logits[i] = float32(-1e30)
				}
			}
		},
		LogitProcessorGate: func(gen []int) bool { return len(gen) >= 2 },
	}, 6)
	if len(got) < 3 {
		t.Fatalf("got %d tokens", len(got))
	}
	if got[0] != ref[0] || got[1] != ref[1] {
		t.Errorf("steps before the gate opened changed: %v vs %v", got[:2], ref[:2])
	}
	for i := 2; i < len(got); i++ {
		if got[i] != forced {
			t.Errorf("step %d after the gate opened = %d, want the forced %d", i, got[i], forced)
		}
	}
}
