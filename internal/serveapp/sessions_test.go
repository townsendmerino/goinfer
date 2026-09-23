package serveapp

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestCommonPrefix(t *testing.T) {
	cases := []struct {
		a, b []int
		want int
	}{
		{nil, nil, 0},
		{[]int{1, 2, 3}, []int{1, 2, 3}, 3},
		{[]int{1, 2, 3}, []int{1, 2, 3, 4}, 3},
		{[]int{1, 2, 9}, []int{1, 2, 3}, 2},
		{[]int{5}, []int{6}, 0},
	}
	for _, c := range cases {
		if got := commonPrefix(c.a, c.b); got != c.want {
			t.Errorf("commonPrefix(%v,%v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// TestBestExtend covers the LRU's selection brain for the exact-containment
// cases: pick the session whose whole token sequence is a prefix of the prompt
// (a genuine continuation), longest first — and crucially do NOT pick a session
// that merely shares a system-prompt preamble (its later turns aren't contained
// in the prompt), so two conversations don't evict each other. The partial-match
// cases L-15 adds (a session whose tail diverges but whose lead is still worth
// reusing) are covered separately below.
func TestBestExtend(t *testing.T) {
	sys := []int{1, 2, 3, 4} // shared system preamble

	// Two conversations, each = sys + its own turns.
	convA := []int{1, 2, 3, 4, 10, 11, 12}
	convB := []int{1, 2, 3, 4, 20, 21}
	sessions := [][]int{convA, convB}

	// Continuing conversation A: A's tokens are a prefix of the new prompt → pick A.
	promptA := append(append([]int(nil), convA...), 13, 14)
	if got := bestExtend(sessions, promptA); got != 0 {
		t.Errorf("continuing A: bestExtend = %d, want 0", got)
	}

	// A brand-new conversation that only shares the system preamble: neither
	// session's full tokens are contained → no reuse (don't hijack A or B).
	fresh := append(append([]int(nil), sys...), 99, 98, 97)
	if got := bestExtend(sessions, fresh); got != -1 {
		t.Errorf("fresh conversation: bestExtend = %d, want -1", got)
	}

	// Exact-prompt repeat of B (regeneration): B fully contained → pick B.
	if got := bestExtend(sessions, append([]int(nil), convB...)); got != 1 {
		t.Errorf("repeat B: bestExtend = %d, want 1", got)
	}

	// Empty session list.
	if got := bestExtend(nil, promptA); got != -1 {
		t.Errorf("empty: bestExtend = %d, want -1", got)
	}
}

// TestColdTierOverflow covers the disk-tier bookkeeping that needs no model:
// the cold tier is capped at demotedMax (oldest blobs deleted), and dropCold
// removes an entry and its blob.
func TestColdTierOverflow(t *testing.T) {
	dir := t.TempDir()
	mk := func(name string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("kv"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	gone := func(name string) bool {
		_, err := os.Stat(filepath.Join(dir, name))
		return os.IsNotExist(err)
	}

	// cold is most-recently-demoted first; c1 is the oldest.
	l := &sessionLRU{demotedMax: 2, cold: []*coldSession{
		{tokens: []int{3}, path: mk("c3")},
		{tokens: []int{2}, path: mk("c2")},
		{tokens: []int{1}, path: mk("c1")},
	}}
	l.evictColdOverflow()
	if len(l.cold) != 2 {
		t.Fatalf("after overflow trim len(cold) = %d, want 2", len(l.cold))
	}
	if !gone("c1") {
		t.Errorf("oldest cold blob c1 should have been deleted")
	}
	if gone("c2") || gone("c3") {
		t.Errorf("c2/c3 should survive the trim")
	}

	l.dropCold(0) // drop the newest (c3)
	if len(l.cold) != 1 || l.cold[0].tokens[0] != 2 {
		t.Fatalf("after dropCold cold = %+v, want just c2", l.cold)
	}
	if !gone("c3") {
		t.Errorf("dropCold should have deleted c3's blob")
	}
}

func TestMoveToFront(t *testing.T) {
	cases := []struct {
		in   []int
		i    int
		want []int
	}{
		{[]int{0, 1, 2, 3}, 2, []int{2, 0, 1, 3}},
		{[]int{0, 1, 2, 3}, 0, []int{0, 1, 2, 3}}, // no-op
		{[]int{0, 1, 2, 3}, 3, []int{3, 0, 1, 2}}, // evict-coldest case
		{[]int{5, 6}, 1, []int{6, 5}},
	}
	for _, c := range cases {
		got := slices.Clone(c.in)
		moveToFront(got, c.i)
		if !slices.Equal(got, c.want) {
			t.Errorf("moveToFront(%v, %d) = %v, want %v", c.in, c.i, got, c.want)
		}
	}
}

// TestBestExtend_stopStringTokensReuse is L-15's fix for the P-18 gate: a
// session that ends with tokens generated AFTER a stop-string hit is committed
// to the session's Tokens() (openai.go's streamTokens appends every generated id
// to ids regardless of the stop cut, and Session.Generate's commit records the
// whole thing), even though only the text up to the cut point ever reached the
// client. The client's NEXT prompt is built from what it actually saw, so it
// never contains those invisible tokens — the old whole-containment rule missed
// on every such turn, forcing a cold prefill plus an eviction of a session that
// was almost entirely reusable. bestExtend now picks it anyway (the decoder's
// own rewindForReuse truncates to the shared prefix, discarding just the
// invisible tail).
func TestBestExtend_stopStringTokensReuse(t *testing.T) {
	turn1 := []int{1, 2, 3, 4} // system preamble + turn 1
	visible := append(append([]int(nil), turn1...), 10, 11, 12)
	// The session actually stored a few more tokens: whatever the model emitted
	// after the stop string before generation was cancelled — committed, never sent.
	stored := append(append([]int(nil), visible...), 13, 14)
	sessions := [][]int{stored}

	// Turn 3 continues from what the client saw, not from what the session stored.
	turn3 := append(append([]int(nil), visible...), 20, 21)
	if got := bestExtend(sessions, turn3); got != 0 {
		t.Errorf("bestExtend = %d, want 0 — the shared prefix (through the invisible post-stop tokens) should still be reused", got)
	}
}

// TestBestExtend_editedLastMessageReuse is P-18's other named case: the client
// edits its last message and resends. Everything before the edit is still a
// genuine shared prefix with the session that generated the ORIGINAL reply, even
// though that session's full token list is no longer contained in the new
// prompt at all (they diverge mid-conversation, not just at the tail).
func TestBestExtend_editedLastMessageReuse(t *testing.T) {
	shared := []int{1, 2, 3, 4, 10, 11, 12}                       // system preamble + turn 1
	original := append(append([]int(nil), shared...), 20, 21, 22) // turn 2, first draft
	sessions := [][]int{original}

	edited := append(append([]int(nil), shared...), 30, 31) // turn 2, edited
	if got := bestExtend(sessions, edited); got != 0 {
		t.Errorf("bestExtend = %d, want 0 — everything before the edit should still be reused", got)
	}
}
