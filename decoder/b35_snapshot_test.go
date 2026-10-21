package decoder

import (
	"context"
	"fmt"
	"io"
	"slices"
	"testing"
)

// B35 (docs/queue-engineering.md) on the hash fake: a recurrent model's prefix snapshot is taken at KVScope.Keep rounded down to a pass, during
// the cold prefill, and a later prompt that repeats that prefix restores it and continues; its reply equals a cold run of the same prompt. The
// fake's restore to anything but the snapshot's own end corrupts the history, as a DeltaNet state from another position would, so a restore
// the decoder should have refused changes the reply.

// recurrentSnapFake is snapFake whose snapshots carry the state (decoder.ResidentRecurrentSnapshots) when carries is set.
type recurrentSnapFake struct {
	*snapFake
	carries bool
}

func (f *recurrentSnapFake) KVSnapshotRecurrent() bool { return f.carries }

func (f *recurrentSnapFake) RestoreKV(h ResidentKVSnapshot, slot, end int) error {
	if err := f.snapFake.RestoreKV(h, slot, end); err != nil {
		return err
	}
	if s := h.(*snapFakeSnap); end != s.rows {
		f.enter()
		for i := range f.hist[slot] {
			f.hist[slot][i] ^= 0x5bd1e995 // the state belongs to position s.rows, not end
		}
		f.leave()
	}
	return nil
}

type recurrentSnapBackend struct {
	Backend
	rf      *recurrentSnapFake
	carries bool
}

func (b *recurrentSnapBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &recurrentSnapFake{snapFake: &snapFake{mc3Fake: newMC3Fake(vocab, 1), markAlign: b35Align}, carries: b.carries}
	return b.rf, true, nil
}
func (b *recurrentSnapBackend) Close() error { return nil }

const b35Align = 64

func loadRecurrentSnapFake(t *testing.T, carries bool, budget int64) (*Model, *recurrentSnapFake) {
	t.Helper()
	be := &recurrentSnapBackend{carries: carries}
	name := fmt.Sprintf("fake-b35-%d", mc3FakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load("../testdata/qwen35-tiny", Options{Backend: name, KVHostBytes: budget})
	if err != nil {
		t.Skipf("no qwen35-tiny: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() || !m.hasRecurrentState() {
		t.Fatalf("resident %v, recurrent %v: the fixture must load as a resident recurrent model", m.ResidentActive(), m.hasRecurrentState())
	}
	old, oldW := kvSnapMinPrompt, stderrKV
	kvSnapMinPrompt, stderrKV = 256, io.Discard
	t.Cleanup(func() { kvSnapMinPrompt, stderrKV = old, oldW })
	return m, be.rf
}

// b35Run generates prompt with keep as KVScope.Keep and returns the reply and how many rows the prefill reused.
func b35Run(t *testing.T, m *Model, prompt []int, keep int) ([]int, int) {
	t.Helper()
	ch, gen := m.Generate(WithKVScope(context.Background(), KVScope{Keep: keep}), prompt, 6, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, gen.PrefillReused
}

// cold is prompt's reply on a fresh model, no snapshot anywhere.
func b35Cold(t *testing.T, prompt []int) []int {
	t.Helper()
	m, _ := loadRecurrentSnapFake(t, true, 0)
	ids, _ := b35Run(t, m, prompt, 0)
	return ids
}

// The first turn: 1,200 ids whose last 4 the next turn rewrites, so Keep is 1,196 and the snapshot lands on the pass boundary at 1,152.
func b35Turn1(vocab int) ([]int, int) {
	return snapToks(1200, 11, vocab), 1196
}

func TestB35_restoreAtKeepEqualsCold(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, keep := b35Turn1(vocab)
	if _, reused := b35Run(t, m, a, keep); reused != 0 || rf.taken != 1 {
		t.Fatalf("turn 1: reused %d, %d snapshots taken; want a cold prefill and one snapshot at the mark", reused, rf.taken)
	}
	b := append(slices.Clone(a[:keep]), snapToks(80, 12, vocab)...)
	got, reused := b35Run(t, m, b, len(b)-4)
	if reused != 1152 || rf.restored != 1 {
		t.Fatalf("turn 2: reused %d with %d restores, want the snapshot's 1152 rows restored", reused, rf.restored)
	}
	if want := b35Cold(t, b); !slices.Equal(got, want) {
		t.Fatalf("turn 2 after the restore %v, a cold run %v", got, want)
	}
}

func TestB35_onlyTheSnapshotsOwnEnd(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, keep := b35Turn1(vocab)
	b35Run(t, m, a, keep)
	c := append(slices.Clone(a[:1100]), snapToks(80, 13, vocab)...) // shares 1,100 ids: the 1,152-row state is not its prefix's
	got, reused := b35Run(t, m, c, 0)
	if reused != 0 || rf.restored != 0 {
		t.Fatalf("a prompt sharing 1,100 ids restored %d rows (%d restores): a recurrent snapshot serves only its own end", reused, rf.restored)
	}
	if want := b35Cold(t, c); !slices.Equal(got, want) {
		t.Fatalf("the cold prefill after the refusal %v, a cold run %v", got, want)
	}
}

func TestB35_aShortSuffixRunsCold(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, keep := b35Turn1(vocab)
	b35Run(t, m, a, keep)
	d := append(slices.Clone(a[:1152]), snapToks(3, 14, vocab)...) // 3 ids after the snapshot: under the minimum pass
	got, reused := b35Run(t, m, d, 0)
	if reused != 0 || rf.restored != 0 {
		t.Fatalf("a 3-id suffix restored %d rows (%d restores): a suffix under the minimum pass prefills cold", reused, rf.restored)
	}
	if want := b35Cold(t, d); !slices.Equal(got, want) {
		t.Fatalf("the cold prefill after the decline %v, a cold run %v", got, want)
	}
}

func TestB35_noStateNoSnapshot(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, false, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, keep := b35Turn1(vocab)
	b35Run(t, m, a, keep)
	if m.KeepsRecurrentSnapshots() || rf.taken != 0 {
		t.Fatalf("a recurrent model whose snapshots do not carry its state took %d snapshots (keeps %v)", rf.taken, m.KeepsRecurrentSnapshots())
	}
}
