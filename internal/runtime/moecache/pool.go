package moecache

import (
	"fmt"
	"slices"
)

// PickSlot returns the first free slot index (slotExpert[s] < 0), or the least-recently-used tail of lru when all slots
// are occupied.
func PickSlot(slotExpert, lru []int) int {
	for s, e := range slotExpert {
		if e < 0 {
			return s
		}
	}
	return lru[len(lru)-1]
}

// TouchSlot returns a slice with slot moved to the most-recently-used front of lru.
func TouchSlot(lru []int, slot int) []int {
	out := make([]int, 0, len(lru)+1)
	out = append(out, slot)
	for _, x := range lru {
		if x != slot {
			out = append(out, x)
		}
	}
	return out
}

// BatchMiss describes one expert miss claimed during ClaimBatch for subsequent staging I/O.
type BatchMiss struct {
	Index  int
	Expert int
	Slot   int
}

// BatchClaim holds the sequential slot-selection outcome of ClaimBatch before concurrent staging I/O runs.
type BatchClaim struct {
	Slots        []int
	Misses       []BatchMiss
	Hits         int
	ColdStarts   int
	Evictions    int
	EvictedSlots []int
}

// ClaimBatch resolves slots for ids sequentially against slotExpert, where, and lru, deduplicating repeated expert IDs
// within the batch and updating slotExpert and evicted entries in where immediately while leaving newly missed experts
// out of where until the caller finishes staging their bytes. Panics if distinct misses exceed len(slotExpert).
func ClaimBatch(ids []int, slotExpert []int, where map[int]int, lru []int) (BatchClaim, []int) {
	res := BatchClaim{Slots: make([]int, len(ids))}
	seen := make(map[int]int, len(ids))
	claimed := make(map[int]bool, len(ids))
	for i, e := range ids {
		if s, ok := seen[e]; ok {
			res.Hits++
			res.Slots[i] = s
			continue
		}
		if s, ok := where[e]; ok {
			res.Hits++
			lru = TouchSlot(lru, s)
			res.Slots[i] = s
			seen[e] = s
			continue
		}
		s := PickSlot(slotExpert, lru)
		if claimed[s] {
			panic(fmt.Sprintf("moecache: batch of %d distinct experts exceeds this pool's %d slots (id %d re-picked slot %d) — the caller must ensure N >= top-k",
				len(ids), len(slotExpert), e, s))
		}
		claimed[s] = true
		if old := slotExpert[s]; old >= 0 {
			delete(where, old)
			res.Evictions++
			res.EvictedSlots = append(res.EvictedSlots, s)
		} else {
			res.ColdStarts++
		}
		slotExpert[s] = e
		lru = TouchSlot(lru, s)
		res.Slots[i] = s
		seen[e] = s
		res.Misses = append(res.Misses, BatchMiss{Index: i, Expert: e, Slot: s})
	}
	return res, lru
}

// PoolFittedRun returns the exclusive end row j >= start (with start <= j <= n) of the longest contiguous run whose
// union of topK routed experts fits within nSlots, along with those unique expert IDs in first-seen order.
func PoolFittedRun(start, n, nSlots, topK int, ids func(i int) []uint32) (end int, union []int) {
	in := make(map[int]bool)
	j := start
	for ; j < n; j++ {
		var add []int
		for _, id := range ids(j)[:topK] {
			eid := int(id)
			if !in[eid] && !slices.Contains(add, eid) {
				add = append(add, eid)
			}
		}
		if len(union)+len(add) > nSlots {
			break
		}
		for _, eid := range add {
			in[eid] = true
		}
		union = append(union, add...)
	}
	return j, union
}
