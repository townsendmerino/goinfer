//go:build darwin

package metal

import (
	"encoding/binary"
	"testing"

	"github.com/townsendmerino/aikit/gpu"
)

// holdsInPool reports whether slot s (a view returned by pool) holds expert e's bytes. Reads
// through the pool's own contiguous base buffers at slot s's stride offset — s.guW etc are
// Buffer.At()-offset VIEWS, and U32s()/U16s() ignore that offset (see expertSlot's doc comment in
// expertpool.go), so this must NOT call s.guW.U32s() directly.
func holdsInPool(pool *expertPool, s expertSlot, e int) bool {
	return pool.guW.U32s()[s.slot*pool.nGuW] == uint32(e) && pool.dW.U32s()[s.slot*pool.nDW] == uint32(e) &&
		pool.guS.U16s()[s.slot*pool.nGuS] == uint16(e) && pool.dS.U16s()[s.slot*pool.nDS] == uint16(e)
}

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
			if !holdsInPool(seqPool, seqSlots[i], e) {
				t.Fatalf("request %d: sequential result for expert %d holds wrong bytes", reqN, e)
			}
			if !holdsInPool(batchPool, batchSlots[i], e) {
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
			// s.guW/dW/guS/dS are Buffer.At()-offset VIEWS onto the pool's contiguous base buffers
			// (expertpool.go's expertSlot doc comment) — gpu.Upload respects that offset, but a
			// plain U32s()/U16s()+copy() would silently write slot 0 every time, so this must use
			// gpu.Upload directly (mirrors production's copyBytesToU32Buf/copyU16sToBuf, which are
			// unexported to the metal package and take []byte/[]uint16 rather than []uint32).
			guW := make([]uint32, nGuW)
			dW := make([]uint32, nDW)
			for i := range guW {
				guW[i] = uint32(e)
			}
			for i := range dW {
				dW[i] = uint32(e)
			}
			guS := make([]uint16, nGuS)
			dS := make([]uint16, nDS)
			for i := range guS {
				guS[i] = uint16(e)
			}
			for i := range dS {
				dS[i] = uint16(e)
			}
			if err := gpu.Upload(s.guW, guW); err != nil {
				t.Fatalf("upload guW: %v", err)
			}
			if err := gpu.Upload(s.dW, dW); err != nil {
				t.Fatalf("upload dW: %v", err)
			}
			if err := gpu.Upload(s.guS, guS); err != nil {
				t.Fatalf("upload guS: %v", err)
			}
			if err := gpu.Upload(s.dS, dS); err != nil {
				t.Fatalf("upload dS: %v", err)
			}
		}
		return p
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
			if !holdsInPool(seqPool, seqSlots[i], e) || !holdsInPool(batchPool, batchSlots[i], e) {
				t.Fatalf("request %d: expert %d not staged correctly via pread", reqN, e)
			}
		}
		samePoolState(t, seqPool, batchPool)
	}
	if seqPool.preads == 0 || batchPool.preads == 0 {
		t.Fatal("test setup: stagePread path did not engage (preads counter stayed 0)")
	}
}
