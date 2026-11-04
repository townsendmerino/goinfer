package kvmgr

import "testing"

// TestPickSlotFor_deepOwnRollback is the slot rule of KV2 follow-up 1 (docs/tasks/task-kv-architecture-2026-10.md §6):
// a request that keeps its slot's whole first prompt takes that slot in place even when it gives up more than it keeps,
// with an empty slot free and with the least recently used slot elsewhere; a busy own slot is still never taken; and
// without first-prompt lengths the keep-at-least-as-much-as-it-overwrites rule holds.
func TestPickSlotFor_deepOwnRollback(t *testing.T) {
	// slot 0: a conversation whose first prompt is 397 tokens, now 921 long; the request keeps 409 and gives up 512.
	for _, tc := range []struct {
		name         string
		scores, lens []int
		firsts       []int
		uses         []uint64
		busy         []bool
		want         int
	}{
		{"an empty slot is free", []int{409, 0}, []int{921, 0}, []int{397, 0}, []uint64{2, 0}, nil, 0},
		{"no empty slot, the least recently used is another", []int{409, 327}, []int{921, 900}, []int{397, 397}, []uint64{2, 1}, nil, 0},
		{"a regenerate of the first prompt after a long history", []int{396, 0}, []int{2000, 0}, []int{397, 0}, []uint64{1, 0}, nil, 0},
		{"the own slot is busy", []int{409, 0}, []int{921, 0}, []int{397, 0}, []uint64{2, 0}, []bool{true, false}, 1},
		{"diverges inside the first prompt", []int{327, 0}, []int{921, 0}, []int{397, 0}, []uint64{2, 0}, nil, 1},
		{"no first prompt known", []int{409, 0}, []int{921, 0}, []int{0, 0}, []uint64{2, 0}, nil, 1},
		{"no first-prompt lengths at all", []int{409, 0}, []int{921, 0}, nil, []uint64{2, 0}, nil, 1},
	} {
		if got := PickSlotFor(tc.scores, tc.lens, tc.firsts, tc.uses, tc.busy, BlockRows); got != tc.want {
			t.Errorf("%s: PickSlotFor = %d, want %d", tc.name, got, tc.want)
		}
	}
}
