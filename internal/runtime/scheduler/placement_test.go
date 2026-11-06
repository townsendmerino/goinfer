package scheduler

import "testing"

// TestPickDonor_shortBusy: a busy slice shorter than hits, or nil, marks the sessions past its end idle instead of
// panicking; a busy session is never the donor; among equal hits the least recent wins.
func TestPickDonor_shortBusy(t *testing.T) {
	for _, c := range []struct {
		name       string
		hits       []int
		busy       []bool
		donor, hit int
	}{
		{"nil busy", []int{64, 128, 128}, nil, 2, 128},
		{"short busy", []int{64, 128, 128}, []bool{false, true}, 2, 128},
		{"busy best", []int{64, 128}, []bool{false, true}, 0, 64},
		{"no hit", []int{0, 0}, nil, -1, 0},
	} {
		if d, h := PickDonor(c.hits, c.busy); d != c.donor || h != c.hit {
			t.Errorf("%s: PickDonor = (%d, %d), want (%d, %d)", c.name, d, h, c.donor, c.hit)
		}
	}
}

// BestExtend's floor: a preamble two sessions share never wins a prompt by itself, so conversations that open the same way keep
// their own sessions; an exact continuation and a partial match beyond what the session shares with every other one do win, the
// longest such match wins, and a session passed as nil (busy) is never chosen.
func TestBestExtend_choosesAMatchPastTheSharedPreamble(t *testing.T) {
	a, b := []int{1, 2, 3, 4}, []int{1, 2, 3, 9}
	for _, c := range []struct {
		name     string
		sessions [][]int
		prompt   []int
		want     int
	}{
		{"only the shared preamble", [][]int{a, b}, []int{1, 2, 3, 7}, -1},
		{"an exact continuation", [][]int{a, b}, []int{1, 2, 3, 4, 5}, 0},
		{"a partial match past the preamble", [][]int{{1, 2, 3, 4, 5, 6}, {1, 2, 9}}, []int{1, 2, 3, 4, 7}, 0},
		{"the longest qualifying match", [][]int{{1, 2, 3}, {1, 2, 3, 4, 5}}, []int{1, 2, 3, 4, 5, 6}, 1},
		{"a busy session", [][]int{nil, {1, 2}}, []int{1, 2, 3}, 1},
		{"no match", [][]int{{5}, {6}}, []int{1}, -1},
		{"no sessions", nil, []int{1}, -1},
	} {
		if got := BestExtend(c.sessions, c.prompt); got != c.want {
			t.Errorf("%s: BestExtend = %d, want %d", c.name, got, c.want)
		}
	}
}

// PickSession's room rule: with a fresh session to spare, a prompt that would keep less of its candidate than it overwrites
// takes the fresh one, unless it keeps the candidate conversation's whole first prompt (the conversation rolling back). With
// no room the candidate is reused, since that reuse is the eviction. A firsts slice shorter than sessions means unknown.
func TestPickSession_roomRuleAndTheRollbackException(t *testing.T) {
	ten := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	keeps3, keeps6 := []int{1, 2, 3, 99}, []int{1, 2, 3, 4, 5, 6, 99}
	for _, c := range []struct {
		name   string
		firsts []int
		prompt []int
		spare  bool
		want   int
	}{
		{"keeps 3 of 10, room to spare", []int{0}, keeps3, true, -1},
		{"keeps 3 of 10, no room", []int{0}, keeps3, false, 0},
		{"keeps 6 of 10, room to spare", []int{0}, keeps6, true, 0},
		{"keeps 3, the whole 3-token first prompt", []int{3}, keeps3, true, 0},
		{"keeps 3 of a 4-token first prompt", []int{4}, keeps3, true, -1},
		{"no firsts", nil, keeps3, true, -1},
	} {
		if got := PickSession([][]int{ten}, c.firsts, c.prompt, c.spare); got != c.want {
			t.Errorf("%s: PickSession = %d, want %d", c.name, got, c.want)
		}
	}
}

// ColdestIdle is the least recently used idle session: busy is in recency order, so the last idle one.
func TestColdestIdle_isTheLeastRecentIdleSession(t *testing.T) {
	for _, c := range []struct {
		busy []bool
		want int
	}{
		{[]bool{false, true, false, true}, 2},
		{[]bool{false}, 0},
		{[]bool{true, true}, -1},
		{nil, -1},
	} {
		if got := ColdestIdle(c.busy); got != c.want {
			t.Errorf("ColdestIdle(%v) = %d, want %d", c.busy, got, c.want)
		}
	}
}
