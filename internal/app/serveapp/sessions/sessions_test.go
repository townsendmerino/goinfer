package sessions

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/scheduler"
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
		if got := scheduler.CommonPrefix(c.a, c.b); got != c.want {
			t.Errorf("scheduler.CommonPrefix(%v,%v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// TestBestExtend covers the LRU's selection brain for the exact-containment cases: pick the session whose whole token
// sequence is a prefix of the prompt (a genuine continuation), longest first — and crucially do NOT pick a session
// that merely shares a system-prompt preamble (its later turns aren't contained in the prompt), so two conversations
// don't evict each other.
func TestBestExtend(t *testing.T) {
	sys := []int{1, 2, 3, 4} // shared system preamble

	// Two conversations, each = sys + its own turns.
	convA := []int{1, 2, 3, 4, 10, 11, 12}
	convB := []int{1, 2, 3, 4, 20, 21}
	lists := [][]int{convA, convB}

	// Continuing conversation A: A's tokens are a prefix of the new prompt -> pick A.
	promptA := append(slices.Clone(convA), 13, 14)
	if got := scheduler.BestExtend(lists, promptA); got != 0 {
		t.Errorf("continuing A: BestExtend = %d, want 0", got)
	}

	// A brand-new conversation that only shares the system preamble: neither
	// session's full tokens are contained -> no reuse (don't hijack A or B).
	fresh := append(slices.Clone(sys), 99, 98, 97)
	if got := scheduler.BestExtend(lists, fresh); got != -1 {
		t.Errorf("fresh conversation: BestExtend = %d, want -1", got)
	}

	// Exact-prompt repeat of B (regeneration): B fully contained -> pick B.
	if got := scheduler.BestExtend(lists, slices.Clone(convB)); got != 1 {
		t.Errorf("repeat B: BestExtend = %d, want 1", got)
	}

	// Empty session list.
	if got := scheduler.BestExtend(nil, promptA); got != -1 {
		t.Errorf("empty: BestExtend = %d, want -1", got)
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
	l := &LRU{demotedMax: 2, cold: []*coldSession{
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

// TestBestExtend_stopStringTokensReuse covers a session that ends with tokens generated after a stop-string hit: the
// client's next prompt never contains those invisible tokens, so BestExtend must still reuse the shared prefix.
func TestBestExtend_stopStringTokensReuse(t *testing.T) {
	turn1 := []int{1, 2, 3, 4} // system preamble + turn 1
	visible := append(slices.Clone(turn1), 10, 11, 12)
	stored := append(slices.Clone(visible), 13, 14)
	lists := [][]int{stored}

	turn3 := append(slices.Clone(visible), 20, 21)
	if got := scheduler.BestExtend(lists, turn3); got != 0 {
		t.Errorf("BestExtend = %d, want 0 — the shared prefix should still be reused", got)
	}
}

// TestBestExtend_editedLastMessageReuse covers a client editing its last message and resending: everything before the
// edit is still a genuine shared prefix with the session that generated the original reply.
func TestBestExtend_editedLastMessageReuse(t *testing.T) {
	shared := []int{1, 2, 3, 4, 10, 11, 12}              // system preamble + turn 1
	original := append(slices.Clone(shared), 20, 21, 22) // turn 2, first draft
	lists := [][]int{original}

	edited := append(slices.Clone(shared), 30, 31) // turn 2, edited
	if got := scheduler.BestExtend(lists, edited); got != 0 {
		t.Errorf("BestExtend = %d, want 0 — everything before the edit should still be reused", got)
	}
}

// TestPickSession_sparePreambleGoesFresh ensures two interleaved conversations sharing only a chat-template preamble
// do not hijack each other's single session while the LRU still has spare capacity.
func TestPickSession_sparePreambleGoesFresh(t *testing.T) {
	pre := []int{1, 2, 3, 4, 5, 6, 7}                 // the chat template's shared lead
	convA := append(slices.Clone(pre), 100, 101, 102) // conversation A's turn 1 + reply ...
	for i := range 160 {
		convA = append(convA, 200+i)
	}
	lists := [][]int{convA}
	convB := append(slices.Clone(pre), 900, 901) // conversation B's first turn: shares only the preamble

	if got := Pick(lists, convB, true); got != -1 {
		t.Errorf("spare capacity, preamble-only match: Pick = %d, want -1 (fresh) — reusing would truncate A's %d tokens to %d", got, len(convA), len(pre))
	}
	// A full LRU has no room for a fresh session anyway: reusing the only candidate is the eviction.
	if got := Pick(lists, convB, false); got != 0 {
		t.Errorf("no spare capacity: Pick = %d, want 0", got)
	}
	// A's own continuation still reuses, with room or without.
	nextA := append(slices.Clone(convA), 300, 301)
	for _, spare := range []bool{true, false} {
		if got := Pick(lists, nextA, spare); got != 0 {
			t.Errorf("continuing A (spare=%v): Pick = %d, want 0", spare, got)
		}
	}
}

// TestPickSession_keepsP18Reuse pins that the spare-capacity rule leaves partial matches (stop-string tail or edited
// last message) reusing their own session.
func TestPickSession_keepsP18Reuse(t *testing.T) {
	turn1 := []int{1, 2, 3, 4}
	visible := append(slices.Clone(turn1), 10, 11, 12)
	stored := append(slices.Clone(visible), 13, 14)
	if got := Pick([][]int{stored}, append(slices.Clone(visible), 20, 21), true); got != 0 {
		t.Errorf("stop-string tail: Pick = %d, want 0", got)
	}
	shared := []int{1, 2, 3, 4, 10, 11, 12}
	original := append(slices.Clone(shared), 20, 21, 22)
	if got := Pick([][]int{original}, append(slices.Clone(shared), 30, 31), true); got != 0 {
		t.Errorf("edited last message: Pick = %d, want 0", got)
	}
}

// TestSessions_persistedStateIsOwnerOnly pins that persisted KV blobs and their directory are owner-only (0600/0700).
func TestSessions_persistedStateIsOwnerOnly(t *testing.T) {
	if FilePerm != 0o600 {
		t.Errorf("session blobs are written %#o, want 0o600", FilePerm)
	}
	if DirPerm != 0o700 {
		t.Errorf("the session directory is created %#o, want 0o700", DirPerm)
	}
	b, err := os.ReadFile("sessions.go")
	if err != nil {
		t.Fatal(err)
	}
	for line := range strings.SplitSeq(string(b), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		for _, world := range []string{"0o644", "0o755", "0o666", "0o777"} {
			if strings.Contains(line, world) {
				t.Errorf("sessions.go still writes something %s — a KV blob is the conversation: %s",
					world, strings.TrimSpace(line))
			}
		}
	}
}
