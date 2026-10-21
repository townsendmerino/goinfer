package decoder

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// copyFake is mc3Fake with a slot-to-slot copy (ResidentKVCopier). mc3Fake's rows are a hash of each position's input
// and its logits a hash of the slot's whole history, so a copy that moves the wrong rows changes every later token.
// plant makes the copy wrong in one way, to show the gate goes red: "short" copies a block less than it claims, "wrong"
// writes another slot than dst (a slot index mixed up), "noop" reports success and copies nothing.
type copyFake struct {
	*mc3Fake
	plant          string
	copies         int
	declinePrefill bool
}

// PrefillLast declines while declinePrefill is set, as a backend whose batched pass refuses a prompt at call time.
func (f *copyFake) PrefillLast(ctx context.Context, embs [][]float32, startPos int) ([]float32, error) {
	if f.declinePrefill {
		return nil, fmt.Errorf("copyFake: batched prefill declined")
	}
	return f.mc3Fake.PrefillLast(ctx, embs, startPos)
}

func (f *copyFake) CopyKV(src, dst, start, end int) error {
	f.enter()
	defer f.leave()
	if src == dst || start%64 != 0 || end%64 != 0 || start > end {
		return fmt.Errorf("copyFake: bad copy %d->%d [%d,%d)", src, dst, start, end)
	}
	if len(f.hist[src]) < end || len(f.hist[dst]) < start {
		return &KVCopyDecline{Reason: "copyFake: the rows are not there"}
	}
	f.copies++
	switch f.plant {
	case "noop":
		return nil
	case "short":
		end -= 64
	case "wrong":
		for s := range f.hist {
			if s != src && s != dst {
				dst = s
				break
			}
		}
	}
	if len(f.hist[dst]) < start {
		return nil
	}
	f.hist[dst] = append(slices.Clone(f.hist[dst][:start]), f.hist[src][start:min(end, len(f.hist[src]))]...)
	return nil
}

type copyFakeBackend struct {
	Backend
	rf    *copyFake
	slots int
	plant string
}

func (b *copyFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &copyFake{mc3Fake: newMC3Fake(vocab, b.slots), plant: b.plant}
	return b.rf, true, nil
}
func (b *copyFakeBackend) Close() error { return nil }

func loadWithCopyFake(t *testing.T, slots int, plant string) (*Model, *copyFake) {
	t.Helper()
	be := &copyFakeBackend{slots: slots, plant: plant}
	name := fmt.Sprintf("fake-copy-%d", mc3FakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with the copy fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

// kv2Preamble is 327 tokens: five whole blocks and 7 rows that a hit can never take.
func kv2Preamble(vocab int) []int {
	out := make([]int, 327)
	for i := range out {
		out[i] = (i*37 + 11) % vocab
	}
	return out
}

// kv2Conv is one conversation's state across turns.
type kv2Conv struct {
	prompt []int
	turns  []mc3Turn
	shared []int
}

// newKV2Conv is conversation c's first prompt: the preamble, then 70 tokens of its own, so another conversation would
// give up more than a block of its slot and copies the preamble instead of taking the slot.
func newKV2Conv(pre []int, c, vocab int) *kv2Conv {
	p := slices.Clone(pre)
	for i := range 70 {
		p = append(p, (c*17+i*5+3)%vocab)
	}
	return &kv2Conv{prompt: p}
}

// turn runs the conversation's next turn and extends its prompt with the reply and a new user turn.
func (cv *kv2Conv) turn(t *testing.T, ctx context.Context, m *Model, c, maxTok int) {
	t.Helper()
	vocab := m.w.arch.VocabSize
	ch, gen := m.Generate(ctx, cv.prompt, maxTok, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("conversation %d turn %d: %v", c, len(cv.turns), err)
	}
	tn := len(cv.turns)
	cv.turns = append(cv.turns, mc3Turn{ids, gen.PrefillReused})
	cv.shared = append(cv.shared, gen.PrefillShared)
	cv.prompt = append(append(slices.Clone(cv.prompt), ids...), (c*7+tn)%vocab, (c*11+tn)%vocab)
}

// kv2Alone is every conversation's turns, each conversation run by itself on a fresh model of one slot: nothing to
// share, no other conversation's history anywhere.
func kv2Alone(t *testing.T, convs, turns, maxTok int, edit bool) [][]mc3Turn {
	out := make([][]mc3Turn, convs)
	for c := range convs {
		m, _ := loadWithCopyFake(t, 1, "")
		pre := kv2Preamble(m.w.arch.VocabSize)
		cv := newKV2Conv(pre, c, m.w.arch.VocabSize)
		if edit && c == 1 {
			cv = kv2EditOf(t, pre, m, maxTok)
		}
		for range turns {
			cv.turn(t, context.Background(), m, c, maxTok)
		}
		out[c] = cv.turns
	}
	return out
}

// kv2EditOf is conversation 1 as an edit of conversation 0: conversation 0's first prompt, its reply, and a different
// second user turn, so its prompt holds rows conversation 0's decode wrote.
func kv2EditOf(t *testing.T, pre []int, m *Model, maxTok int) *kv2Conv {
	t.Helper()
	a := newKV2Conv(pre, 0, m.w.arch.VocabSize)
	first := slices.Clone(a.prompt)
	ref, _ := loadWithCopyFake(t, 1, "")
	ch, gen := ref.Generate(context.Background(), first, maxTok, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatal(err)
	}
	return &kv2Conv{prompt: append(append(first, ids...), 201, 202, 203)}
}

// TestKV2_scheduleIndependence is G3 of docs/measurements/kv2-2026-10-14.md on the decoder: three conversations
// sharing a 327-token preamble, in each registered schedule and slot count, emit exactly the tokens they emit run
// alone, and the schedules that can share do. Each planted copy defect must turn it red.
func TestKV2_scheduleIndependence(t *testing.T) {
	const turns, maxTok = 2, 12
	type sched struct {
		name  string
		order []int // conversation per step, in order; each step is that conversation's next turn
		edit  bool
	}
	scheds := []sched{
		{"(a) A1 B1 C1, then their second turns", []int{0, 1, 2, 0, 1, 2}, false},
		{"(b) C1 B1 A1, then C B A", []int{2, 1, 0, 2, 1, 0}, false},
		{"(c) interleaved: A1 B1 A2 C1 B2 C2", []int{0, 1, 0, 2, 1, 2}, false},
		{"(f) B is an edit of A", []int{0, 1, 0, 1, 2, 2}, true},
	}
	run := func(t *testing.T, slots int, s sched, plant string) (mismatch []string, shared int) {
		alone := kv2Alone(t, 3, turns, maxTok, s.edit)
		m, rf := loadWithCopyFake(t, slots, plant)
		pre := kv2Preamble(m.w.arch.VocabSize)
		convs := make([]*kv2Conv, 3)
		for c := range convs {
			convs[c] = newKV2Conv(pre, c, m.w.arch.VocabSize)
		}
		if s.edit {
			convs[1] = kv2EditOf(t, pre, m, maxTok)
		}
		for _, c := range s.order {
			convs[c].turn(t, context.Background(), m, c, maxTok)
		}
		for c, cv := range convs {
			for tn, got := range cv.turns {
				if !slices.Equal(got.ids, alone[c][tn].ids) {
					mismatch = append(mismatch, fmt.Sprintf("conversation %d turn %d: %v, alone %v", c, tn, got.ids, alone[c][tn].ids))
				}
				shared += cv.shared[tn]
			}
		}
		if rf.copies == 0 && shared > 0 {
			t.Fatalf("PrefillShared %d with no copy made", shared)
		}
		return mismatch, shared
	}
	for _, slots := range []int{1, 2, 4} {
		for _, s := range scheds {
			t.Run(fmt.Sprintf("slots%d/%s", slots, s.name), func(t *testing.T) {
				mismatch, shared := run(t, slots, s, "")
				for _, e := range mismatch {
					t.Error(e)
				}
				if slots > 1 && shared == 0 {
					t.Errorf("no turn took a copy: the schedule never shared, so it checks nothing")
				}
				t.Logf("rows copied across the schedule: %d", shared)
			})
		}
	}
	for _, plant := range []string{"short", "wrong", "noop"} {
		t.Run("planted/"+plant, func(t *testing.T) {
			if mismatch, _ := run(t, 3, scheds[0], plant); len(mismatch) == 0 {
				t.Errorf("a %q copy left every reply equal to its reply alone: the gate cannot see that defect", plant)
			} else {
				t.Logf("red, as it must be (%d turns differ; first: %s)", len(mismatch), mismatch[0])
			}
		})
	}
}

// TestKV2_busySourceUnderMC3 is G3's schedule (d): B arrives while A is still decoding. A's prompt was published when
// its prefill ended, so B copies the preamble from A's busy slot, and both reply exactly as they do alone.
func TestKV2_busySourceUnderMC3(t *testing.T) {
	const maxTok = 40
	alone := kv2Alone(t, 2, 1, maxTok, false)
	m, rf := loadWithCopyFake(t, 2, "")
	rf.delay = 2 * time.Millisecond // A's decode takes long enough for B to arrive during it
	if got := m.EnableResidentConcurrency(2); got != 2 {
		t.Fatalf("EnableResidentConcurrency(2) = %d", got)
	}
	pre := kv2Preamble(m.w.arch.VocabSize)
	a, b := newKV2Conv(pre, 0, m.w.arch.VocabSize), newKV2Conv(pre, 1, m.w.arch.VocabSize)
	chA, genA := m.Generate(context.Background(), a.prompt, maxTok, SamplingParams{})
	var idsA []int
	idsA = append(idsA, <-chA) // A has prefilled: its prompt is published
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for id := range chA {
			idsA = append(idsA, id)
		}
	}()
	b.turn(t, context.Background(), m, 1, maxTok)
	wg.Wait()
	if err := genA.Err(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(idsA, alone[0][0].ids) {
		t.Errorf("A: %v, alone %v", idsA, alone[0][0].ids)
	}
	if !slices.Equal(b.turns[0].ids, alone[1][0].ids) {
		t.Errorf("B: %v, alone %v", b.turns[0].ids, alone[1][0].ids)
	}
	if b.shared[0] != 320 {
		t.Errorf("B copied %d rows from A's busy slot, want the preamble's 320", b.shared[0])
	}
}

// TestKV2_scopes is G4(c) on the decoder: a hit never crosses a namespace, and Off takes nothing from another
// sequence while a conversation still continues its own slot.
func TestKV2_scopes(t *testing.T) {
	const maxTok = 8
	for _, tc := range []struct {
		name       string
		a, b       KVScope
		wantShared bool
	}{
		{"one namespace shares", KVScope{}, KVScope{}, true},
		{"two namespaces share nothing", KVScope{Namespace: "key-1"}, KVScope{Namespace: "key-2"}, false},
		{"off shares nothing", KVScope{Off: true}, KVScope{Off: true}, false},
		{"B off, A not: B still takes nothing", KVScope{}, KVScope{Off: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := loadWithCopyFake(t, 2, "")
			pre := kv2Preamble(m.w.arch.VocabSize)
			a, b := newKV2Conv(pre, 0, m.w.arch.VocabSize), newKV2Conv(pre, 1, m.w.arch.VocabSize)
			a.turn(t, WithKVScope(context.Background(), tc.a), m, 0, maxTok)
			b.turn(t, WithKVScope(context.Background(), tc.b), m, 1, maxTok)
			if got := b.turns[0].reused > 0; got != tc.wantShared {
				t.Errorf("B reused %d (shared %d), want sharing %v", b.turns[0].reused, b.shared[0], tc.wantShared)
			}
			// A continues its own slot under its own scope, whatever B's was.
			a.turn(t, WithKVScope(context.Background(), tc.a), m, 0, maxTok)
			if a.turns[1].reused == 0 || a.shared[1] != 0 {
				t.Errorf("A's second turn reused %d (shared %d): a conversation must continue its own slot", a.turns[1].reused, a.shared[1])
			}
		})
	}
}

// TestKV2_optionOff: Options.KVShareOff (serve's -kv-share off) stops every copy on the model, whatever the request's
// scope says, and leaves a conversation's own continuation alone.
func TestKV2_optionOff(t *testing.T) {
	const maxTok = 8
	m, rf := loadWithCopyFake(t, 2, "")
	m.kvShareOff = true // Options.KVShareOff
	pre := kv2Preamble(m.w.arch.VocabSize)
	a, b := newKV2Conv(pre, 0, m.w.arch.VocabSize), newKV2Conv(pre, 1, m.w.arch.VocabSize)
	a.turn(t, context.Background(), m, 0, maxTok)
	b.turn(t, context.Background(), m, 1, maxTok)
	if b.turns[0].reused != 0 || rf.copies != 0 {
		t.Errorf("B reused %d with %d copies under KVShareOff, want none", b.turns[0].reused, rf.copies)
	}
	a.turn(t, context.Background(), m, 0, maxTok)
	if a.turns[1].reused == 0 {
		t.Error("A's second turn reused nothing: KVShareOff must leave a continuation alone")
	}
}

// TestKV2_declinedPrefillSharesNothing is G4(a) on the decoder: a pass whose batched prefill declined ran on decode's
// kernels, not the class the manager was told, so its rows name no producer and another conversation copies none of them.
func TestKV2_declinedPrefillSharesNothing(t *testing.T) {
	const maxTok = 8
	m, rf := loadWithCopyFake(t, 2, "")
	pre := kv2Preamble(m.w.arch.VocabSize)
	a, b := newKV2Conv(pre, 0, m.w.arch.VocabSize), newKV2Conv(pre, 1, m.w.arch.VocabSize)
	rf.declinePrefill = true
	a.turn(t, context.Background(), m, 0, maxTok)
	rf.declinePrefill = false
	for s := 0; s < 2; s++ {
		if q := m.kv.Seq(s); q != nil {
			for _, r := range q.Classes() {
				if r.Class != kvmgr.ClassUnknown {
					t.Errorf("slot %d rows [%d, %d) recorded as class %d after a declined prefill", s, r.Start, r.Start+r.Len, r.Class)
				}
			}
		}
	}
	b.turn(t, context.Background(), m, 1, maxTok)
	if b.shared[0] != 0 || rf.copies != 0 {
		t.Errorf("B copied %d rows from a sequence whose prefill declined", b.shared[0])
	}
}

// TestKV2_windowGroupRollback: on a model with sliding-window layers a request that rolls a slot back by 400 rows (a
// client dropping or editing its last reply) takes the slot in place, within the rollback floor (the task doc's §5.12);
// without window groups the same request gives up more than a block, so it copies into the other slot instead.
func TestKV2_windowGroupRollback(t *testing.T) {
	for _, window := range []int{0, 1024} {
		m, rf := loadWithCopyFake(t, 2, "")
		m.w.arch.SlidingWindow = window
		m.rebuildProgramForTest()
		vocab := m.w.arch.VocabSize
		first := make([]int, 1000)
		for i := range first {
			first[i] = (i*37 + 11) % vocab
		}
		ch, g := m.Generate(context.Background(), first, 8, SamplingParams{})
		for range ch {
		}
		if g.Err() != nil {
			t.Fatal(g.Err())
		}
		edit := append(slices.Clone(first[:600]), 7, 8, 9)
		ch, g = m.Generate(context.Background(), edit, 4, SamplingParams{})
		for range ch {
		}
		if g.Err() != nil {
			t.Fatal(g.Err())
		}
		inPlace := g.PrefillReused == 600 && g.PrefillShared == 0
		if window > 0 && !inPlace {
			t.Errorf("window %d: reused %d, copied %d (copies %d), want 600 in place", window, g.PrefillReused, g.PrefillShared, rf.copies)
		}
		if window == 0 && (inPlace || g.PrefillShared != 576) {
			t.Errorf("no window: reused %d, copied %d, want a copy of 576 into the other slot", g.PrefillReused, g.PrefillShared)
		}
	}
}

// TestKV2_ownRollbackStaysInPlace: a conversation that edits a later turn (giving up more than a block of its own
// slot) continues in that slot, reusing its history, decode rows included; a new conversation that shares only part
// of its first prompt still copies instead of cutting it.
func TestKV2_ownRollbackStaysInPlace(t *testing.T) {
	m, rf := loadWithCopyFake(t, 2, "")
	vocab := m.w.arch.VocabSize
	run := func(p []int, maxTok int) ([]int, *Generation) {
		ch, g := m.Generate(context.Background(), p, maxTok, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if g.Err() != nil {
			t.Fatal(g.Err())
		}
		return ids, g
	}
	p1 := make([]int, 1000)
	for i := range p1 {
		p1[i] = (i*37 + 11) % vocab
	}
	r1, _ := run(p1, 8)
	turn := func(seed int) []int {
		out := make([]int, 70)
		for i := range out {
			out[i] = (seed*13 + i*7) % vocab
		}
		return out
	}
	p2 := append(append(slices.Clone(p1), r1...), turn(1)...)
	run(p2, 8)
	edit := append(append(slices.Clone(p1), r1...), turn(2)...) // gives up turn 1's 70 tokens and its reply
	_, g := run(edit, 4)
	if g.PrefillReused != len(p1)+len(r1) || g.PrefillShared != 0 {
		t.Errorf("own rollback: reused %d, copied %d, want %d in place", g.PrefillReused, g.PrefillShared, len(p1)+len(r1))
	}
	copies := rf.copies
	other := append(slices.Clone(p1[:900]), turn(3)...)
	_, g = run(other, 4)
	if g.PrefillShared == 0 || rf.copies == copies {
		t.Errorf("a new conversation sharing 900 tokens of the first prompt: reused %d, copied %d, want a copy", g.PrefillReused, g.PrefillShared)
	}
}
