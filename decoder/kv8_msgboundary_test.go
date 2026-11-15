package decoder

import (
	"context"
	"slices"
	"testing"
)

// KV8's message-boundary checkpoint (docs/measurements/kv8-msgboundary-2026-11/README.md, M0) on the hash fake: a recurrent model keeps a
// second snapshot where the prompt's last message starts (KVScope.MsgBoundary rounded down to a pass), beside B35's Keep one, so a prompt
// that rewrites the last message from near its start restores instead of prefilling from 0. The fake's pass is 64 rows.

// msgRun generates prompt under sc and returns the reply and how many rows the prefill reused.
func msgRun(t *testing.T, m *Model, prompt []int, sc KVScope) ([]int, int) {
	t.Helper()
	ch, gen := m.Generate(WithKVScope(context.Background(), sc), prompt, 6, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, gen.PrefillReused
}

// msgTurn1 is 1,200 ids whose last message starts at 1,000 and whose last 4 ids the next turn rewrites: the marks fall on the passes at
// 960 (the message) and 1,152 (Keep 1,196).
func msgTurn1(vocab int) ([]int, KVScope) {
	return snapToks(1200, 11, vocab), KVScope{Keep: 1196, MsgBoundary: 1000}
}

func TestKV8Msg_aRewrittenLastMessageRestoresTheBoundary(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, sc := msgTurn1(vocab)
	if _, reused := msgRun(t, m, a, sc); reused != 0 || rf.taken != 2 {
		t.Fatalf("turn 1: reused %d, %d snapshots taken; want a cold prefill and two snapshots (the message's start and Keep)", reused, rf.taken)
	}
	b35Held(t, m, 960, 1152)
	b := append(slices.Clone(a[:1003]), snapToks(200, 12, vocab)...) // the last message rewritten from its 4th id
	got, reused := msgRun(t, m, b, KVScope{Keep: len(b) - 4, MsgBoundary: 1100})
	if reused != 960 || rf.restored != 1 {
		t.Fatalf("turn 2: reused %d with %d restores, want the message boundary's 960 rows restored", reused, rf.restored)
	}
	if want := b35Cold(t, b); !slices.Equal(got, want) {
		t.Fatalf("turn 2 after the restore %v, a cold run %v", got, want)
	}
}

// Keep's own snapshot is not replaced by the message's, which it shares a prefix with: the next turn that matches through Keep restores the deeper point.
func TestKV8Msg_keepStaysBesideTheBoundary(t *testing.T) {
	m, _ := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, sc := msgTurn1(vocab)
	msgRun(t, m, a, sc)
	c := append(slices.Clone(a[:1196]), snapToks(80, 13, vocab)...)
	got, reused := msgRun(t, m, c, KVScope{})
	if reused != 1152 {
		t.Fatalf("a turn matching through Keep reused %d, want Keep's 1152 rows", reused)
	}
	if want := b35Cold(t, c); !slices.Equal(got, want) {
		t.Fatalf("after the restore %v, a cold run %v", got, want)
	}
}

// A request that restored a snapshot and takes both points drops the one it restored from once.
func TestKV8Msg_aRestoredTurnTakesBothPoints(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, true, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, sc := msgTurn1(vocab)
	msgRun(t, m, a, sc) // held: 960, 1152
	b := append(slices.Clone(a[:1003]), snapToks(200, 12, vocab)...)
	msgRun(t, m, b, KVScope{Keep: len(b) - 4, MsgBoundary: 1100}) // restores 960; takes 1088 and 1152
	if rf.restored != 1 {
		t.Fatalf("restores %d, want 1", rf.restored)
	}
	b35Held(t, m, 1088, 1152, 1152)
	st, _ := m.KVSnapshotStats()
	var want int64
	for _, s := range st.Snapshots {
		want += int64(s.Rows) * 1024 // the fake's snapshot is 1 KiB a row
	}
	if st.Used != want {
		t.Fatalf("the store charges %d bytes for snapshots of %d: one was released twice or not at all", st.Used, want)
	}
}

// A model whose snapshots do not carry its state takes none, whatever the scope says.
func TestKV8Msg_noStateNoBoundarySnapshot(t *testing.T) {
	m, rf := loadRecurrentSnapFake(t, false, 1<<40)
	_, _, _, _, _, _, vocab := m.Dims()
	a, sc := msgTurn1(vocab)
	msgRun(t, m, a, sc)
	if rf.taken != 0 {
		t.Fatalf("took %d snapshots", rf.taken)
	}
}

// M3: the store's budget holds with two snapshots a request. With room for one, Keep's stays: the message's, taken first, is the least recently
// used when Keep's is reserved, and the store drops it rather than refuse the deeper point. With room for neither, a request takes none.
func TestKV8Msg_theBudgetHoldsWithTwoSnapshotsARequest(t *testing.T) {
	for _, c := range []struct {
		name   string
		budget int64
		held   []int
	}{
		{"room for both", (960 + 1152) * 1024, []int{960, 1152}},
		{"room for one", 1200 * 1024, []int{1152}},
		{"room for none", 900 * 1024, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, _ := loadRecurrentSnapFake(t, true, c.budget)
			_, _, _, _, _, _, vocab := m.Dims()
			a, sc := msgTurn1(vocab)
			msgRun(t, m, a, sc)
			b35Held(t, m, c.held...)
			st, _ := m.KVSnapshotStats()
			if st.Used > st.Budget {
				t.Fatalf("the store holds %d bytes of a %d budget", st.Used, st.Budget)
			}
		})
	}
}

// The cut (docs/measurements/kv8-msgboundary-2026-11/README.md, second slice, C0 and C1): a recurrent model's snapshots are taken at their
// points rounded down to a 64-row block, not to a pass. Here the passes are 128 rows, so the grid would give 896 and 1,152.

// cutRun loads the 128-row-pass fake and runs turn 1: 1,236 ids, the last message from 1,000, Keep 1,230.
func cutTurn1(t *testing.T) (*Model, *recurrentSnapFake, []int, int) {
	t.Helper()
	m, rf := loadRecurrentSnapFakeAlign(t, true, 1<<40, 128)
	_, _, _, _, _, _, vocab := m.Dims()
	a := snapToks(1236, 11, vocab)
	msgRun(t, m, a, KVScope{Keep: 1230, MsgBoundary: 1000})
	return m, rf, a, vocab
}

func TestKV8Cut_snapshotsLandOnTheirBlocks(t *testing.T) {
	m, rf, a, vocab := cutTurn1(t)
	b35Held(t, m, 960, 1216)
	b := append(slices.Clone(a[:1003]), snapToks(200, 12, vocab)...)
	got, reused := msgRun(t, m, b, KVScope{Keep: len(b) - 4, MsgBoundary: 1100})
	if reused != 960 || rf.restored != 1 {
		t.Fatalf("a rewritten last message reused %d rows (%d restores), want the block-rounded boundary's 960", reused, rf.restored)
	}
	if want := b35Cold(t, b); !slices.Equal(got, want) {
		t.Fatalf("reply after the restore %v, cold %v", got, want)
	}
}

// A request that restored the off-grid 960 still takes its own Keep snapshot, and the next turn restores it.
func TestKV8Cut_aRestoredTurnKeepsItsOwnPoint(t *testing.T) {
	m, _, a, vocab := cutTurn1(t)
	b := append(slices.Clone(a[:1003]), snapToks(200, 12, vocab)...) // 1,203 ids
	msgRun(t, m, b, KVScope{Keep: len(b) - 4, MsgBoundary: 1100})
	b35Held(t, m, 1088, 1152, 1216)
	c := append(slices.Clone(b[:len(b)-4]), snapToks(80, 13, vocab)...)
	got, reused := msgRun(t, m, c, KVScope{})
	if reused != 1152 {
		t.Fatalf("the turn after an edit reused %d rows, want the edit turn's own Keep point 1152", reused)
	}
	if want := b35Cold(t, c); !slices.Equal(got, want) {
		t.Fatalf("reply after the restore %v, cold %v", got, want)
	}
}

// A point that would leave a final pass under the resident's minimum is lowered a block.
func TestKV8Cut_theFinalPassKeepsItsMinimum(t *testing.T) {
	m, _ := loadRecurrentSnapFakeAlign(t, true, 1<<40, 128)
	_, _, _, _, _, _, vocab := m.Dims()
	a := snapToks(1219, 11, vocab) // Keep 1,217 rounds to 1,216, 3 rows from the end: under the 8-row minimum
	msgRun(t, m, a, KVScope{Keep: 1217})
	b35Held(t, m, 1152)
}
