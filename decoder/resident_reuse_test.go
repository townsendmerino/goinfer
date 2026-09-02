package decoder

import "testing"

// TestResidentReuseLen pins the prefix rule, including the two cases that make it SAFE rather
// than merely fast: it never claims the whole prompt (the seed's logits must be recomputed),
// and it stops at the first divergent id rather than resynchronising later — a client that
// edits its last message shifts tokenisation, and everything after the edit is a different
// conversation even where the text looks similar.
func TestResidentReuseLen(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cached []int
		prompt []int
		want   int
	}{
		{"nothing cached", nil, []int{1, 2, 3}, 0},
		{"empty prompt", []int{1, 2, 3}, nil, 0},
		{"exact prefix extension (the agent-turn case)", []int{1, 2, 3}, []int{1, 2, 3, 4, 5}, 3},
		{"cache longer than prompt", []int{1, 2, 3, 4, 5}, []int{1, 2, 3}, 2},
		{"identical — must still leave a seed token", []int{1, 2, 3}, []int{1, 2, 3}, 2},
		{"diverges at 0", []int{9, 2, 3}, []int{1, 2, 3, 4}, 0},
		{"diverges midway", []int{1, 2, 9, 9}, []int{1, 2, 3, 4}, 2},
		{"a later match must NOT resynchronise", []int{1, 9, 3, 4}, []int{1, 2, 3, 4}, 1},
		{"single-token prompt", []int{1}, []int{1}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{resIDs: tc.cached}
			if got := m.residentReuseLen(tc.prompt); got != tc.want {
				t.Errorf("residentReuseLen(%v | cached %v) = %d, want %d", tc.prompt, tc.cached, got, tc.want)
			}
		})
	}
}

// TestResidentReuseLen_neverClaimsTheSeed is the invariant stated separately because a
// regression here is silent: reusing the whole prompt leaves generateInto with no seed logits
// and nothing to recompute them from.
func TestResidentReuseLen_neverClaimsTheSeed(t *testing.T) {
	for n := 1; n <= 8; n++ {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = i + 1
		}
		m := &Model{resIDs: ids}
		if got := m.residentReuseLen(ids); got >= n {
			t.Errorf("prompt of %d identical tokens reused %d — must leave at least one to prefill", n, got)
		}
	}
}

// TestResidentForgetIDs: forgetting must be total, because a PARTIALLY stale id list is worse
// than none — it would match a prefix that the cache no longer holds.
func TestResidentForgetIDs(t *testing.T) {
	m := &Model{}
	m.residentCommitIDs([]int{1, 2}, []int{3, 4})
	if len(m.resIDs) != 4 {
		t.Fatalf("commit recorded %v, want prompt+generated", m.resIDs)
	}
	m.residentForgetIDs()
	if m.resIDs != nil {
		t.Errorf("forget left %v, want nil", m.resIDs)
	}
	if got := m.residentReuseLen([]int{1, 2, 3, 4, 5}); got != 0 {
		t.Errorf("after forgetting, reuse must be 0, got %d", got)
	}
}

// TestResidentCommitIDs_copies: the recorded list must not alias the caller's slices, or a
// later append by the caller silently rewrites what we believe the cache holds.
func TestResidentCommitIDs_copies(t *testing.T) {
	prompt := []int{1, 2, 3}
	generated := []int{4, 5}
	m := &Model{}
	m.residentCommitIDs(prompt, generated)
	prompt[0], generated[0] = 99, 99
	if m.resIDs[0] == 99 || m.resIDs[3] == 99 {
		t.Errorf("resIDs aliases the caller's slice: %v", m.resIDs)
	}
}
