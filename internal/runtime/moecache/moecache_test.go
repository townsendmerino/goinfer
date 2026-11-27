package moecache

import (
	"slices"
	"testing"
)

func TestSlotCache_LRUAndRollback(t *testing.T) {
	c := NewSlotCache(8, 3)

	// Cold fill of 3 slots.
	for wantSlot, exp := range []uint32{2, 5, 1} {
		s, hit := c.Admit(exp)
		if hit || s != wantSlot {
			t.Fatalf("Admit(%d) = (%d, %v), want (%d, false)", exp, s, hit, wantSlot)
		}
	}
	// Hit on expert 2 refreshes its recency, making expert 5 the oldest.
	if s, hit := c.Admit(2); !hit || s != 0 {
		t.Fatalf("Admit(2) = (%d, %v), want (0, true)", s, hit)
	}
	// Miss on expert 7 evicts expert 5 (slot 1).
	if s, hit := c.Admit(7); hit || s != 1 {
		t.Fatalf("Admit(7) = (%d, %v), want (1, false)", s, hit)
	}
	// Expert 5 is now a miss and evicts expert 1 (slot 2).
	if s, hit := c.Admit(5); hit || s != 2 {
		t.Fatalf("Admit(5) = (%d, %v), want (2, false)", s, hit)
	}

	if c.Hits() != 1 || c.Misses() != 5 {
		t.Fatalf("stats = (%d, %d), want (1, 5)", c.Hits(), c.Misses())
	}

	// Rollback of slot 2 (expert 5) makes expert 5 miss into empty slot 2 next time.
	c.Rollback([]PendingAdmit{{Slot: 2, Expert: 5}})
	if s, hit := c.Admit(5); hit || s != 2 {
		t.Fatalf("after Rollback Admit(5) = (%d, %v), want (2, false)", s, hit)
	}

	c.RecordStageDemand([]uint32{2, 5, 2, 7})
	if c.Stages() != 1 || c.Distinct() != 3 {
		t.Fatalf("demand = (%d, %d), want (1, 3)", c.Stages(), c.Distinct())
	}

	resized := NewSlotCache(8, 4)
	resized.CopyStatsFrom(c)
	if resized.Hits() != c.Hits() || resized.Misses() != c.Misses() || resized.Stages() != 1 || resized.Distinct() != 3 {
		t.Fatalf("CopyStatsFrom mismatch: got (%d,%d,%d,%d)", resized.Hits(), resized.Misses(), resized.Stages(), resized.Distinct())
	}
	resized.ResetStats()
	if resized.Hits() != 0 || resized.Misses() != 0 || resized.Stages() != 0 || resized.Distinct() != 0 {
		t.Fatalf("ResetStats left non-zero counters")
	}
}

func TestClaimBatch_MatchesSequentialAndPanicsOverCapacity(t *testing.T) {
	requests := [][]int{
		{0, 1, 2, 3},
		{2, 2, 3},
		{1, 4},
		{4, 0},
	}

	seqSlots := []int{-1, -1, -1, -1}
	seqWhere := map[int]int{}
	var seqLRU []int
	var seqHits, seqCold, seqEvict int

	batSlots := []int{-1, -1, -1, -1}
	batWhere := map[int]int{}
	var batLRU []int
	var batHits, batCold, batEvict int

	for reqIdx, ids := range requests {
		wantOut := make([]int, len(ids))
		for i, e := range ids {
			if s, ok := seqWhere[e]; ok {
				seqHits++
				seqLRU = TouchSlot(seqLRU, s)
				wantOut[i] = s
				continue
			}
			s := PickSlot(seqSlots, seqLRU)
			if old := seqSlots[s]; old >= 0 {
				delete(seqWhere, old)
				seqEvict++
			} else {
				seqCold++
			}
			seqSlots[s] = e
			seqWhere[e] = s
			seqLRU = TouchSlot(seqLRU, s)
			wantOut[i] = s
		}

		var claim BatchClaim
		claim, batLRU = ClaimBatch(ids, batSlots, batWhere, batLRU)
		batHits += claim.Hits
		batCold += claim.ColdStarts
		batEvict += claim.Evictions
		for _, ms := range claim.Misses {
			batWhere[ms.Expert] = ms.Slot
		}

		if !slices.Equal(wantOut, claim.Slots) {
			t.Fatalf("req %d slots = %v, want %v", reqIdx, claim.Slots, wantOut)
		}
		if !slices.Equal(seqLRU, batLRU) || !slices.Equal(seqSlots, batSlots) {
			t.Fatalf("req %d state mismatch: seqLRU=%v batLRU=%v", reqIdx, seqLRU, batLRU)
		}
		if seqHits != batHits || seqCold != batCold || seqEvict != batEvict {
			t.Fatalf("req %d counters mismatch", reqIdx)
		}
	}

	defer func() {
		if recover() == nil {
			t.Fatal("ClaimBatch accepted 5 distinct misses into 4 slots without panicking")
		}
	}()
	ClaimBatch([]int{0, 1, 2, 3, 4}, []int{-1, -1, -1, -1}, map[int]int{}, nil)
}

func TestPoolFittedRun_AndGrouping(t *testing.T) {
	routes := [][]uint32{
		{3, 1},
		{1, 4},
		{4, 2},
		{5, 6},
	}
	end, union := PoolFittedRun(0, len(routes), 4, 2, func(i int) []uint32 { return routes[i] })
	if end != 3 || !slices.Equal(union, []int{3, 1, 4, 2}) {
		t.Fatalf("PoolFittedRun = (%d, %v), want (3, [3 1 4 2])", end, union)
	}

	flat := []uint32{3, 1, 1, 4, 4, 3}
	wgt := []float32{0.7, 0.3, 0.6, 0.4, 0.2, 0.8}
	experts, byE := GroupByExpert(flat, 3, 2)
	if !slices.Equal(experts, []uint32{1, 3, 4}) || len(byE[3]) != 2 || byE[3][1] != (Pair{Row: 2, Rank: 1}) {
		t.Fatalf("GroupByExpert = (%v, %v)", experts, byE)
	}

	weighted := GroupWeightedByExpert(flat, wgt, 3, 2, 5)
	if len(weighted[4]) != 2 || weighted[4][0].Weight != 0.4 || weighted[4][1].Row != 2 {
		t.Fatalf("GroupWeightedByExpert[4] = %+v", weighted[4])
	}

	entries := SlotBatchEntries([][]uint32{{2, 0}, {2, 1}, {2, 0}}, 2, 2)
	// Slot 2 has pairs 0, 2, 4 -> two entries [2,2,0,2] and [2,1,4,0]; slot 0 has pairs 1, 5 -> [0,2,1,5]; slot 1 has pair 3 -> [1,1,3,0].
	wantEntries := []uint32{2, 2, 0, 2, 2, 1, 4, 0, 0, 2, 1, 5, 1, 1, 3, 0}
	if !slices.Equal(entries, wantEntries) {
		t.Fatalf("SlotBatchEntries = %v, want %v", entries, wantEntries)
	}
}
