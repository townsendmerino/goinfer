package decoder

import (
	"context"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// twoClassFake is copyFake with a second kernel class, as Metal's and CUDA's residents have: a batched pass
// (PrefillLast) writes rows that differ from the rows decode writes for the same input, so a reply depends on which
// kernel wrote each row of its history. It reports a kernel floor of 1 and an exact tail, so the decoder records every
// batched row fast and every decode row exact (Model.kvClassMap), and a copy takes only batched rows. On copyFake, with
// one class, a reply re-prefilled from a copy equals the reply that continued in place.
type twoClassFake struct{ *copyFake }

// batchedTag is mixed into every row a batched pass writes.
const batchedTag = 0x9E3779B97F4A7C15

func (f *twoClassFake) PrefillKernelFloor() int { return 1 }
func (f *twoClassFake) PrefillTailExact() bool  { return true }

func (f *twoClassFake) PrefillLast(ctx context.Context, embs [][]float32, startPos int) ([]float32, error) {
	if _, err := f.copyFake.PrefillLast(ctx, embs, startPos); err != nil {
		return nil, err
	}
	h := f.hist[f.bound]
	acc := uint64(1469598103934665603)
	for p := range h {
		if p >= startPos {
			h[p] ^= batchedTag
		}
		acc = (acc ^ h[p]) * 1099511628211
	}
	out := make([]float32, f.vocab)
	out[acc%uint64(f.vocab)] = 1 // mc3Fake.fwd's logits, over the tagged rows
	return out, nil
}

type twoClassFakeBackend struct {
	Backend
	rf    *twoClassFake
	slots int
}

func (b *twoClassFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &twoClassFake{&copyFake{mc3Fake: newMC3Fake(vocab, b.slots)}}
	return b.rf, true, nil
}
func (b *twoClassFakeBackend) Close() error { return nil }

func loadWithTwoClassFake(t *testing.T, slots int) (*Model, *twoClassFake) {
	t.Helper()
	be := &twoClassFakeBackend{slots: slots}
	name := fmt.Sprintf("fake-twoclass-%d", mc3FakeBackends.Add(1))
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
		t.Fatalf("Load with the two-class fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

// fu1Turn is one served turn: its prompt and reply, where it ran and what it reused, and the bound slot's rows after it.
type fu1Turn struct {
	prompt, ids                  []int
	slot, reused, shared, copies int
	rows                         []uint64
	continuing                   bool // its conversation's previous turn's slot served no other turn in between
}

// fu1Next is conversation c's next prompt after turns: the first prompt; its reply and 500 tokens; a rollback to the
// first prompt and its reply plus 2 tokens, diverging from the slot's sequence right there; that turn's reply and 500
// tokens; and a second rollback to the same point.
func fu1Next(c int, turns []fu1Turn, pre []int, vocab int) []int {
	tn := len(turns)
	if tn == 0 {
		return newKV2Conv(pre, c, vocab).prompt
	}
	prev := turns[tn-1]
	full := append(slices.Clone(prev.prompt), prev.ids...)
	if tn%2 == 0 {
		base := append(slices.Clone(turns[0].prompt), turns[0].ids...)
		e := (full[len(base)] + 1) % vocab
		return append(base, e, (e+97)%vocab)
	}
	for i := range 500 {
		full = append(full, (c*37+tn*11+i*3+5)%vocab)
	}
	return full
}

// fu1Gen runs one greedy turn of 12 tokens on m and returns what it emitted.
func fu1Gen(t *testing.T, m *Model, prompt []int) ([]int, *Generation) {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, 12, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	return ids, g
}

// TestKV2FU1_ownRollbackScheduleIndependence is FU1-G3 (docs/tasks/task-kv-architecture-2026-10.md §6, KV2 follow-up
// 1): three conversations sharing KV2's preamble, each rolling back twice to its first prompt and first reply after a
// long turn (giving up more than it keeps), in every registered schedule at 1, 2 and 4 slots. A turn whose slot still
// holds its conversation, and that keeps the first prompt of that slot's lineage, continues it in place, copies nothing
// and reuses what it kept. Every turn emits and holds exactly what its reference does: its conversation's lineage
// replayed alone on one slot, restarted cold at every turn that could not continue.
func TestKV2FU1_ownRollbackScheduleIndependence(t *testing.T) {
	const convs, turns = 3, 5
	scheds := []struct {
		name  string
		order []int
	}{
		{"(a) round robin", []int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}},
		{"(b) round robin reversed", []int{2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0}},
		{"(c) each rollback after its own long turn", []int{0, 1, 0, 0, 2, 1, 1, 2, 0, 2, 0, 1, 2, 1, 2}},
		{"(d) one conversation at a time", []int{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}},
	}
	for seed := int64(1); seed <= 20; seed++ {
		var order []int
		for c := range convs {
			for range turns {
				order = append(order, c)
			}
		}
		rand.New(rand.NewSource(seed)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		scheds = append(scheds, struct {
			name  string
			order []int
		}{fmt.Sprintf("random seed %d", seed), order})
	}
	deepAt := map[int]int{}
	for _, slots := range []int{1, 2, 4} {
		for _, s := range scheds {
			t.Run(fmt.Sprintf("slots%d/%s", slots, s.name), func(t *testing.T) {
				m, rf := loadWithTwoClassFake(t, slots)
				vocab := m.w.arch.VocabSize
				pre := kv2Preamble(vocab)
				got := make([][]fu1Turn, convs)
				last := map[int][2]int{} // slot -> the (conversation, turn) it last served
				for _, c := range s.order {
					p := fu1Next(c, got[c], pre, vocab)
					copies := rf.copies
					ids, g := fu1Gen(t, m, p)
					x := fu1Turn{prompt: p, ids: ids, slot: m.kvSlots().Bound(), reused: g.PrefillReused, shared: g.PrefillShared,
						copies: rf.copies - copies}
					x.rows = slices.Clone(rf.hist[x.slot])
					if tn := len(got[c]); tn > 0 {
						prev := got[c][tn-1]
						x.continuing = last[prev.slot] == [2]int{c, tn - 1}
					}
					last[x.slot] = [2]int{c, len(got[c])}
					got[c] = append(got[c], x)
				}
				deep := 0
				for c := range convs {
					var ref *Model
					var refRf *twoClassFake
					lineFirst := 0 // the prompt length of the turn the lineage started at: the slot's FirstPrompt
					for tn, x := range got[c] {
						if !x.continuing {
							ref, refRf = loadWithTwoClassFake(t, 1)
							lineFirst = len(x.prompt)
						}
						ids, g := fu1Gen(t, ref, x.prompt)
						where := fmt.Sprintf("conversation %d turn %d (slot %d, reused %d, copied %d, continuing %v)", c, tn, x.slot, x.reused, x.shared, x.continuing)
						if !slices.Equal(x.ids, ids) {
							t.Errorf("%s: emitted %v, reference %v", where, x.ids, ids)
						}
						if !slices.Equal(x.rows, refRf.hist[0]) {
							t.Errorf("%s: the slot's %d rows differ from the reference's %d", where, len(x.rows), len(refRf.hist[0]))
						}
						if !x.continuing {
							continue
						}
						prev := got[c][tn-1]
						full := append(slices.Clone(prev.prompt), prev.ids...)
						keep := min(len(x.prompt)-1, commonPrefixLen(full, x.prompt))
						if keep < lineFirst-1 {
							continue // it diverges inside the lineage's first prompt: main's rule, not this item's
						}
						if x.slot != prev.slot || x.shared != 0 || x.copies != 0 || x.reused != keep || g.PrefillReused != keep {
							t.Errorf("%s: want its previous slot %d in place, nothing copied, %d reused (the reference reused %d)", where, prev.slot, keep, g.PrefillReused)
						} else if len(full)-keep > keep {
							deep++
						}
					}
				}
				deepAt[slots] += deep
				if slots == 4 && deep == 0 {
					t.Error("no rollback continued in place giving up more than it kept: the schedule checks nothing of FU1")
				}
				t.Logf("rollbacks continued in place giving up more than they kept: %d; CopyKV calls: %d", deep, rf.copies)
			})
		}
	}
	if deepAt[2] == 0 {
		t.Error("at 2 slots no rollback continued in place giving up more than it kept: the slot count checks nothing of FU1")
	}
	t.Logf("rollbacks continued in place giving up more than they kept, by slot count: %v", deepAt)
}
