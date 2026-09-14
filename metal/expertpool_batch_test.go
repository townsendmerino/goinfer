//go:build darwin

package metal

import (
	"encoding/binary"
	"testing"
)

// stagingFor builds a deterministic synthetic stageFn for expertpool tests — mirrors
// TestExpertPool_lruAndStaging's own stage() (kept package-private and duplicated rather than
// shared, since a shared helper touching two test files invites exactly the "which one is the
// oracle" confusion this repo's own measurement-discipline notes warn about).
func stagingFor(nGuW, nDW, nGuS, nDS int) stageFn {
	return func(e int) ([]byte, []uint16, []byte, []uint16) {
		guW := make([]byte, nGuW*4)
		dW := make([]byte, nDW*4)
		for i := range nGuW {
			binary.LittleEndian.PutUint32(guW[i*4:], uint32(e))
		}
		for i := range nDW {
			binary.LittleEndian.PutUint32(dW[i*4:], uint32(e))
		}
		guS := make([]uint16, nGuS)
		dS := make([]uint16, nDS)
		for i := range guS {
			guS[i] = uint16(e)
		}
		for i := range dS {
			dS[i] = uint16(e)
		}
		return guW, guS, dW, dS
	}
}

// samePoolState reports whether two pools built from the same sequence of requests ended up in
// IDENTICAL states — not just "returned the right bytes this time", but structurally the same, so
// the NEXT request would make the same eviction decision on either. Comparing lru element-by-
// element (not just as sets) matters: two pools that agree on WHICH slots are occupied but
// disagree on recency order would pass a where/slotExpert comparison and then evict differently
// on the very next miss.
func samePoolState(t *testing.T, seq, batch *expertPool) {
	t.Helper()
	if seq.hits != batch.hits || seq.stages != batch.stages ||
		seq.coldStarts != batch.coldStarts || seq.evictions != batch.evictions {
		t.Fatalf("counters diverged: sequential {hits=%d stages=%d cold=%d evict=%d} vs "+
			"batch {hits=%d stages=%d cold=%d evict=%d}",
			seq.hits, seq.stages, seq.coldStarts, seq.evictions,
			batch.hits, batch.stages, batch.coldStarts, batch.evictions)
	}
	if len(seq.where) != len(batch.where) {
		t.Fatalf("where map size diverged: sequential=%d batch=%d", len(seq.where), len(batch.where))
	}
	for e, s := range seq.where {
		if bs, ok := batch.where[e]; !ok || bs != s {
			t.Fatalf("expert %d resident in slot %d (sequential) vs slot %d ok=%v (batch)", e, s, bs, ok)
		}
	}
	if len(seq.lru) != len(batch.lru) {
		t.Fatalf("lru length diverged: sequential=%d batch=%d", len(seq.lru), len(batch.lru))
	}
	for i := range seq.lru {
		if seq.lru[i] != batch.lru[i] {
			t.Fatalf("lru order diverged at index %d: sequential=%v batch=%v", i, seq.lru, batch.lru)
		}
	}
}

// TestExpertPoolBatch_matchesSequential is the correctness gate for M-12 (audit-metal-
// 2026-09-12.md): ensureResidentBatch must produce EXACTLY the same eviction decisions, staged
// contents, and bookkeeping as calling ensureResident once per id in order — the only thing
// allowed to differ is that the I/O for a batch's misses runs concurrently instead of serially.
// Run with -race: the whole point of the fix is spawning goroutines over a pool's own state, and a
// race here would be exactly the kind of bug this rewrite could introduce while looking correct on
// a single-threaded run.
func TestExpertPoolBatch_matchesSequential(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nGuW, nGuS, nDW, nDS = 8, 4, 4, 2
	const N = 4 // deliberately smaller than some request batches below, to force in-batch eviction

	holds := func(s expertSlot, e int) bool {
		return s.guW.U32s()[0] == uint32(e) && s.dW.U32s()[0] == uint32(e) &&
			s.guS.U16s()[0] == uint16(e) && s.dS.U16s()[0] == uint16(e)
	}

	// Each request is one "token"'s routed set, sized <= N throughout — matching production's own
	// invariant (newExpertPool's doc comment: "N must be >= the router's top-k", enforced at build
	// time by TestMoESlotsViaOptions_belowTopKRefusesWithNumbers) — a batch with MORE distinct
	// experts than slots cannot happen in production and is covered separately by
	// TestExpertPoolBatch_overCapacityPanics below. Deliberately includes: multiple cold misses in
	// one batch (request 1, filling every slot), a duplicate id within one batch (request 2), a
	// mix of hit+miss forcing an in-batch eviction of the just-resident set (request 2's `4`), and
	// a request that evicts the then-current LRU (request 3).
	requests := [][]int{
		{0, 1, 2, 3}, // cold start: 4 distinct misses fill all 4 slots
		{2, 2, 3},    // hits, plus a duplicate id within one batch
		{1, 4},       // 1 is a hit; 4 is a miss and must evict the current LRU (0, untouched since request 1)
		{4, 0},       // 4 is now a hit; 0 was evicted above -> miss, evicts the new LRU
	}

	seqPool := newExpertPool(d, N, nGuW, nGuS, nDW, nDS, stagingFor(nGuW, nDW, nGuS, nDS))
	batchPool := newExpertPool(d, N, nGuW, nGuS, nDW, nDS, stagingFor(nGuW, nDW, nGuS, nDS))

	for reqN, ids := range requests {
		seqSlots := make([]expertSlot, len(ids))
		for i, e := range ids {
			seqSlots[i] = seqPool.ensureResident(e)
		}
		batchSlots := batchPool.ensureResidentBatch(ids)

		for i, e := range ids {
			if !holds(seqSlots[i], e) {
				t.Fatalf("request %d: sequential result for expert %d holds wrong bytes", reqN, e)
			}
			if !holds(batchSlots[i], e) {
				t.Fatalf("request %d: batch result for expert %d holds wrong bytes", reqN, e)
			}
		}
		samePoolState(t, seqPool, batchPool)
	}
	t.Logf("expert pool batch vs sequential: %d requests, %d total stages, agree throughout",
		len(requests), seqPool.stages)
}

// TestExpertPoolBatch_overCapacityPanics confirms ensureResidentBatch fails LOUD, not silently
// wrong, on the one input newExpertPool's own doc comment says must never happen in production
// (N >= top-k): a batch whose DISTINCT expert count exceeds the pool's slot count would otherwise
// have two misses land on the same physical slot (the second eviction re-picks a slot a miss
// earlier in the SAME batch just claimed, since pickSlot's free/LRU-tail logic has nowhere else to
// go) — racing two goroutines over one buffer and leaving p.where mapping two experts to one slot.
func TestExpertPoolBatch_overCapacityPanics(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nGuW, nGuS, nDW, nDS = 8, 4, 4, 2
	const N = 4
	p := newExpertPool(d, N, nGuW, nGuS, nDW, nDS, stagingFor(nGuW, nDW, nGuS, nDS))

	defer func() {
		if recover() == nil {
			t.Fatal("ensureResidentBatch accepted 5 distinct experts into a 4-slot pool — should panic")
		}
	}()
	p.ensureResidentBatch([]int{0, 1, 2, 3, 4})
}

// TestExpertPoolBatch_pread mirrors the above but through the stagePread path (the one M-12 is
// actually about — GOINFER_MOE_PREAD's fast path), confirming the concurrent-pread branch
// produces byte-identical results to sequential ensureResident too, not just the mmap-copy branch.
func TestExpertPoolBatch_pread(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nGuW, nGuS, nDW, nDS = 8, 4, 4, 2
	const N = 3

	newPreadPool := func() *expertPool {
		p := newExpertPool(d, N, nGuW, nGuS, nDW, nDS, nil)
		p.stagePread = func(e int, s expertSlot) {
			guW := make([]uint32, nGuW)
			dW := make([]uint32, nDW)
			for i := range guW {
				guW[i] = uint32(e)
			}
			for i := range dW {
				dW[i] = uint32(e)
			}
			copy(s.guW.U32s(), guW)
			copy(s.dW.U32s(), dW)
			guS, dS := s.guS.U16s(), s.dS.U16s()
			for i := range guS {
				guS[i] = uint16(e)
			}
			for i := range dS {
				dS[i] = uint16(e)
			}
		}
		return p
	}
	holds := func(s expertSlot, e int) bool {
		return s.guW.U32s()[0] == uint32(e) && s.dW.U32s()[0] == uint32(e) &&
			s.guS.U16s()[0] == uint16(e) && s.dS.U16s()[0] == uint16(e)
	}

	seqPool, batchPool := newPreadPool(), newPreadPool()
	requests := [][]int{{0, 1, 2}, {1, 4, 5}, {5, 5, 0}} // every request has <= N=3 distinct ids
	for reqN, ids := range requests {
		seqSlots := make([]expertSlot, len(ids))
		for i, e := range ids {
			seqSlots[i] = seqPool.ensureResident(e)
		}
		batchSlots := batchPool.ensureResidentBatch(ids)
		for i, e := range ids {
			if !holds(seqSlots[i], e) || !holds(batchSlots[i], e) {
				t.Fatalf("request %d: expert %d not staged correctly via pread", reqN, e)
			}
		}
		samePoolState(t, seqPool, batchPool)
	}
	if seqPool.preads == 0 || batchPool.preads == 0 {
		t.Fatal("test setup: stagePread path did not engage (preads counter stayed 0)")
	}
}
