package decoder

import (
	"errors"
	"slices"
	"testing"
)

// slotResident is a fake ResidentKVSlotter: it records every UseKVSlot bind. Its ResidentForward half is
// fakeResident's and is never exercised here — residentAcquire only chooses and binds.
type slotResident struct {
	*fakeResident
	n, bound int
	binds    []int
	fail     bool
}

func (s *slotResident) KVSlots() int { return s.n }
func (s *slotResident) UseKVSlot(i int) error {
	if s.fail {
		return errors.New("slotResident: forced bind failure")
	}
	s.bound = i
	s.binds = append(s.binds, i)
	return nil
}

func TestPickResidentSlot(t *testing.T) {
	for _, tc := range []struct {
		name   string
		scores []int
		lens   []int
		uses   []uint64
		want   int
	}{
		{"continuation of slot 1", []int{3, 170}, []int{170, 172}, []uint64{1, 2}, 1},
		{"preamble-only match takes the empty slot", []int{7, 0}, []int{170, 0}, []uint64{1, 0}, 1},
		{"nothing matches: first empty slot", []int{0, 0, 0}, []int{50, 0, 0}, []uint64{3, 0, 0}, 1},
		{"full, preamble-only: least recently used", []int{7, 7, 7}, []int{200, 180, 190}, []uint64{5, 2, 9}, 1},
		{"full, nothing matches: least recently used", []int{0, 0}, []int{10, 20}, []uint64{8, 4}, 1},
		{"a P-18 partial match (keeps most) still reuses", []int{0, 9}, []int{0, 11}, []uint64{0, 1}, 1},
		{"keep == discard reuses", []int{5, 0}, []int{10, 0}, []uint64{1, 0}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := pickResidentSlot(tc.scores, tc.lens, tc.uses, nil); got != tc.want {
				t.Errorf("pickResidentSlot(%v, %v, %v) = %d, want %d", tc.scores, tc.lens, tc.uses, got, tc.want)
			}
		})
	}
}

// TestPickResidentSlot_skipsBusy pins MC3's rule: a slot another running generation holds is never picked, whichever
// rule would have chosen it, and with every slot busy there is no pick.
func TestPickResidentSlot_skipsBusy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		scores []int
		lens   []int
		uses   []uint64
		busy   []bool
		want   int
	}{
		{"the continuation's slot is busy: the empty one", []int{3, 170, 0}, []int{170, 172, 0}, []uint64{1, 2, 0}, []bool{false, true, false}, 2},
		{"the empty slot is busy: LRU among the free", []int{0, 0, 0}, []int{50, 0, 40}, []uint64{3, 0, 1}, []bool{false, true, false}, 2},
		{"the LRU slot is busy: the next least recent", []int{0, 0, 0}, []int{10, 20, 30}, []uint64{8, 4, 6}, []bool{false, true, false}, 2},
		{"all busy", []int{5, 5}, []int{10, 10}, []uint64{1, 2}, []bool{true, true}, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := pickResidentSlot(tc.scores, tc.lens, tc.uses, tc.busy); got != tc.want {
				t.Errorf("pickResidentSlot(%v, %v, %v, %v) = %d, want %d", tc.scores, tc.lens, tc.uses, tc.busy, got, tc.want)
			}
		})
	}
}

// TestResidentAcquire_interleavedConversationsKeepTheirSlots is MC1's decoder-side gate: two conversations that share
// only a chat-template lead, interleaved turn by turn, each reuse their OWN full history every turn — the 1-client
// pattern — with a resident of two slots, where one slot (MC0: 7 tokens reused per turn) thrashes.
func TestResidentAcquire_interleavedConversationsKeepTheirSlots(t *testing.T) {
	rs := &slotResident{n: 2}
	m := &Model{resident: rs}
	pre := []int{1, 2, 3}
	conv := map[string][]int{"A": append(slices.Clone(pre), 100), "B": append(slices.Clone(pre), 200)}
	committed := map[string]int{}
	gen := 10
	for turn := range 4 {
		for _, c := range []string{"A", "B"} {
			prompt := conv[c]
			got := m.residentAcquire(prompt, nil, nil)
			if want := committed[c]; got != want {
				t.Fatalf("turn %d %s: residentAcquire reused %d, want %d (its own committed history)", turn, c, got, want)
			}
			m.residentForgetIDs()
			reply := make([]int, 12) // replies are much longer than the shared lead, as they are in practice
			for i := range reply {
				reply[i] = gen + i
			}
			gen += len(reply)
			m.residentCommitIDs(prompt, reply, nil, nil)
			committed[c] = len(prompt) + len(reply)
			conv[c] = append(append(slices.Clone(prompt), reply...), 300+turn) // the next turn extends it
		}
	}
	if want := []int{1, 0, 1, 0, 1, 0, 1}; !slices.Equal(rs.binds, want) {
		t.Errorf("binds = %v, want %v (B's first turn takes the empty slot, then they alternate)", rs.binds, want)
	}
}

// TestResidentAcquire_fullEvictsLeastRecentlyUsed: a third conversation with two slots evicts the least recently used
// one, never the slot the most recent conversation is on. It still reuses what the evicted slot shares with it (the
// 2-token lead): those positions hold the same tokens.
func TestResidentAcquire_fullEvictsLeastRecentlyUsed(t *testing.T) {
	rs := &slotResident{n: 2}
	m := &Model{resident: rs}
	for _, p := range [][]int{{1, 2, 100}, {1, 2, 200}} { // A on slot 0, then B on slot 1
		m.residentAcquire(p, nil, nil)
		m.residentForgetIDs()
		m.residentCommitIDs(p, []int{7, 8, 9, 10}, nil, nil)
	}
	if got := m.residentAcquire([]int{1, 2, 300}, nil, nil); got != 2 || rs.bound != 0 {
		t.Errorf("third conversation: reused %d on slot %d, want the 2-token lead on slot 0 (A, the least recently used)", got, rs.bound)
	}
}

// TestResidentAcquire_recurrentKeepsOneSlot: a recurrent family's state is not part of a KV slot, so it never binds.
func TestResidentAcquire_recurrentKeepsOneSlot(t *testing.T) {
	rs := &slotResident{n: 4}
	m := recurrentTestModel(nil)
	m.resident = rs
	for _, p := range [][]int{{1, 2, 100}, {1, 2, 200}, {1, 2, 100, 5}} {
		m.residentAcquire(p, nil, nil)
		m.residentForgetIDs()
		m.residentCommitIDs(p, []int{9}, nil, nil)
	}
	if len(rs.binds) != 0 {
		t.Errorf("recurrent family bound slots %v, want none", rs.binds)
	}
}

// TestResidentAcquire_bindFailureGoesCold: if the resident refuses a bind, the generation must not trust any slot's
// bookkeeping — it reuses nothing on whatever the resident still has bound.
func TestResidentAcquire_bindFailureGoesCold(t *testing.T) {
	rs := &slotResident{n: 2}
	m := &Model{resident: rs}
	m.residentAcquire([]int{1, 2, 100}, nil, nil)
	m.residentForgetIDs()
	m.residentCommitIDs([]int{1, 2, 100}, []int{7, 8, 9, 10}, nil, nil)
	rs.fail = true
	if got := m.residentAcquire([]int{1, 2, 200}, nil, nil); got != 0 {
		t.Errorf("after a failed bind: reused %d, want 0", got)
	}
	if m.resIDs != nil {
		t.Errorf("after a failed bind the bound slot's bookkeeping must be forgotten, got %v", m.resIDs)
	}
}
