package decoder

import (
	"context"
	"slices"
	"strconv"
	"testing"
)

// KV7 step 2's policy (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8, M4) on the hash fake: a prompt whose tail after the stable section is
// longer than the rollback takes its snapshot at the boundary, between two passes of the cold prefill; one whose tail fits takes step one's end snapshot;
// a request restores the snapshot its window can serve, not the longest match; and there is one snapshot per prompt.

// PrefillMarkAlign and PrefillLastMarked make snapFake a decoder.ResidentPrefillMarked.
func (f *snapFake) PrefillMarkAlign() int { return f.markAlign }

func (f *snapFake) PrefillLastMarked(ctx context.Context, embs [][]float32, startPos, mark int, at func(pos int)) ([]float32, error) {
	var lg []float32
	fired := false
	for i := 0; i < len(embs); {
		n := min(f.markAlign, len(embs)-i)
		f.enter()
		f.events = append(f.events, 'P')
		for j := 0; j < n; j++ {
			lg = f.fwd(f.bound, embs[i+j], startPos+i+j)
		}
		f.leave()
		i += n
		if pos := startPos + i; at != nil && mark > 0 && i < len(embs) {
			switch {
			case f.markLate && pos > mark && !fired:
				fired = true
				f.marks++
				at(mark)
			case !f.markLate && pos == mark:
				f.marks++
				at(mark)
			}
		}
	}
	return lg, nil
}

const (
	step2Align    = 128
	step2Boundary = 1000 // the stable section: the system and tools, as ids
	step2Floor    = 448  // the windowed rollback the fake stands for
)

// step2Prompt is the stable section followed by a tail of tail ids from a stream named by seed.
func step2Prompt(vocab, tail, seed int) []int {
	return append(slices.Clone(snapToks(step2Boundary, 1, vocab)), snapToks(tail, seed, vocab)...)
}

// step2Gen runs prompt under a scope that carries the boundary (0: none, as a library caller) and returns the generation.
func step2Gen(t *testing.T, m *Model, prompt []int, boundary int) ([]int, *Generation) {
	t.Helper()
	ctx := WithKVScope(context.Background(), KVScope{Boundary: boundary})
	ch, gen := m.Generate(ctx, prompt, 6, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, gen
}

// step2Session is the card's life for a Claude-Code-shaped client: a first session A (a long tail after the stable section), an unrelated conversation that
// takes the one slot, and a second session B that shares only the stable section and its own long tail. It reports why the policy failed, "" when it held.
func step2Session(t *testing.T, markLate bool) string {
	t.Helper()
	m, f := loadWithSnapFakeMarked(t, 1<<30, "", step2Floor, step2Align, 64, markLate)
	_, _, _, _, _, _, vocab := m.Dims()
	a, j, b := step2Prompt(vocab, 700, 2), snapToks(900, 3, vocab), step2Prompt(vocab, 700, 4)
	step2Gen(t, m, a, step2Boundary)
	want := step2Boundary / step2Align * step2Align
	st, _ := m.KVSnapshotStats() // after the first session alone: an unrelated conversation takes its own snapshot later
	switch {
	case f.taken != 1 || st.Held != 1:
		return "the cold session took " + strconv.Itoa(f.taken) + " snapshots and holds " + strconv.Itoa(st.Held) + ", want one"
	case st.Snapshots[0].Rows != want:
		return "the snapshot holds " + strconv.Itoa(st.Snapshots[0].Rows) + " rows, want the boundary rounded down to a pass: " + strconv.Itoa(want)
	case f.marks != 1:
		return "the mark fired " + strconv.Itoa(f.marks) + " times, want once"
	}
	step2Gen(t, m, j, 0) // takes the one slot
	reply, gen := step2Gen(t, m, b, step2Boundary)
	if gen.KVSnapRestored != want {
		return "the second session restored " + strconv.Itoa(gen.KVSnapRestored) + " rows, want " + strconv.Itoa(want)
	}
	if st2, _ := m.KVSnapshotStats(); st2.Unqualified != 0 {
		return "the second session counted as unqualified"
	}
	// the reply is the reply of a load with no snapshots at all
	m0, _ := loadWithSnapFakeMarked(t, 0, "", step2Floor, step2Align, 64, false)
	step2Gen(t, m0, a, step2Boundary)
	step2Gen(t, m0, j, 0)
	if want0, _ := step2Gen(t, m0, b, step2Boundary); !slices.Equal(reply, want0) {
		return "the restored reply differs from the computed one"
	}
	return ""
}

// TestKVSnapshotBoundary_restoresTheStablePrefix: the second session restores the boundary rows and its reply is the computed reply.
func TestKVSnapshotBoundary_restoresTheStablePrefix(t *testing.T) {
	if why := step2Session(t, false); why != "" {
		t.Fatal(why)
	}
}

// TestKVSnapshotBoundary_aSnapshotTakenOnePassLateIsCaught: the same session with the callback one pass late, when the ring has moved on, does not hold.
func TestKVSnapshotBoundary_aSnapshotTakenOnePassLateIsCaught(t *testing.T) {
	if why := step2Session(t, true); why == "" {
		t.Fatal("a snapshot taken one pass late passed the session: the ring check does not see a moved ring")
	}
}

// TestKVSnapshotBoundary_endSnapshotWhenTheTailFits: a tail inside the rollback takes step one's end snapshot, with no mark.
func TestKVSnapshotBoundary_endSnapshotWhenTheTailFits(t *testing.T) {
	m, f := loadWithSnapFakeMarked(t, 1<<30, "", step2Floor, step2Align, 64, false)
	_, _, _, _, _, _, vocab := m.Dims()
	a, j, b := step2Prompt(vocab, 40, 2), snapToks(900, 3, vocab), step2Prompt(vocab, 40, 4)
	step2Gen(t, m, a, step2Boundary)
	st, _ := m.KVSnapshotStats()
	if f.marks != 0 || f.taken != 1 || st.Snapshots[0].Rows != len(a)/64*64 {
		t.Fatalf("marks %d, taken %d, rows %v: want no mark and one end snapshot of %d rows", f.marks, f.taken, st.Snapshots, len(a)/64*64)
	}
	step2Gen(t, m, j, 0)
	if _, gen := step2Gen(t, m, b, step2Boundary); gen.KVSnapRestored != step2Boundary/64*64 {
		t.Fatalf("restored %d rows, want %d", gen.KVSnapRestored, step2Boundary/64*64)
	}
}

// TestKVSnapshotBoundary_restoreTakesTheOneItsWindowServes: with an end snapshot (whose ring cannot serve the shared prefix) and a boundary snapshot of
// one family held, a session restores the boundary one and is not counted unqualified.
func TestKVSnapshotBoundary_restoreTakesTheOneItsWindowServes(t *testing.T) {
	m, f := loadWithSnapFakeMarked(t, 1<<30, "", step2Floor, step2Align, 64, false)
	_, _, _, _, _, _, vocab := m.Dims()
	a0, a1, j, b := step2Prompt(vocab, 700, 2), step2Prompt(vocab, 700, 5), snapToks(900, 3, vocab), step2Prompt(vocab, 700, 4)
	step2Gen(t, m, a0, 0) // no boundary: step one's end snapshot, 1,664 rows
	step2Gen(t, m, j, 0)  // the slot is another conversation's, so the next prefill is cold
	step2Gen(t, m, a1, step2Boundary)
	step2Gen(t, m, j, 0)
	var rows []int
	st, _ := m.KVSnapshotStats()
	for _, x := range st.Snapshots {
		rows = append(rows, x.Rows)
	}
	if !slices.Contains(rows, 1664) || !slices.Contains(rows, step2Boundary/step2Align*step2Align) {
		t.Fatalf("holds snapshots of %v rows, want the end one (1664) and the boundary one", rows)
	}
	before := st.Unqualified // a1 itself found only the unusable end snapshot and was counted then
	_, gen := step2Gen(t, m, b, step2Boundary)
	st, _ = m.KVSnapshotStats()
	if want := step2Boundary / step2Align * step2Align; gen.KVSnapRestored != want || st.Unqualified != before {
		t.Fatalf("restored %d rows (want %d), unqualified %d (was %d, want unchanged); taken %d", gen.KVSnapRestored, want, st.Unqualified, before, f.taken)
	}
}

// TestKVSnapshotBoundary_afterACompaction: a request that keeps the stable section and the first reminder and then continues with a summary restores the
// boundary rows and prefills the rest.
func TestKVSnapshotBoundary_afterACompaction(t *testing.T) {
	m, _ := loadWithSnapFakeMarked(t, 1<<30, "", step2Floor, step2Align, 64, false)
	_, _, _, _, _, _, vocab := m.Dims()
	a, j := step2Prompt(vocab, 700, 2), snapToks(900, 3, vocab)
	y := append(slices.Clone(a[:step2Boundary+50]), snapToks(300, 9, vocab)...) // the same section and 50 ids of the first reminder, then the summary
	step2Gen(t, m, a, step2Boundary)
	step2Gen(t, m, j, 0)
	if _, gen := step2Gen(t, m, y, step2Boundary); gen.KVSnapRestored != step2Boundary/step2Align*step2Align {
		t.Fatalf("restored %d rows, want %d", gen.KVSnapRestored, step2Boundary/step2Align*step2Align)
	}
}

// TestKVSnapshotBoundary_noMarkOffAPassBoundary: a request that continues a slot (its prefill starts past the boundary) fires no mark.
func TestKVSnapshotBoundary_noMarkOffAPassBoundary(t *testing.T) {
	m, f := loadWithSnapFakeMarked(t, 1<<30, "", step2Floor, step2Align, 64, false)
	_, _, _, _, _, _, vocab := m.Dims()
	a := step2Prompt(vocab, 700, 2)
	step2Gen(t, m, a, step2Boundary)
	before := f.marks
	step2Gen(t, m, append(slices.Clone(a), snapToks(80, 7, vocab)...), step2Boundary)
	if f.marks != before {
		t.Fatalf("a continuation fired the mark (%d, was %d)", f.marks, before)
	}
}
