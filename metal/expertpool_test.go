//go:build darwin

package metal

import (
	"encoding/binary"
	"testing"
)

// TestExpertPool_lruAndStaging isolates the paging primitive (expertpool.go) against a synthetic
// stage function — the pool's correctness (LRU eviction, cold start, same-expert-twice, staged
// contents) proven before it is wired into a forward. stage(e) fills every buffer word with e, so a
// slot's contents directly reveal which expert it holds — that is how staged-content correctness and
// eviction are checked, not just the bookkeeping.
func TestExpertPool_lruAndStaging(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const nGuW, nGuS, nDW, nDS = 8, 4, 4, 2
	// stage now returns nibble BYTES for the word buffers (mirrors production int4DirectBytes). Fill
	// each 4-byte word with LE(e) so a byte-copy into the u32 slot reproduces expert id e on LE.
	stage := func(e int) ([]byte, []uint16, []byte, []uint16) {
		guW := make([]byte, nGuW*4)
		dW := make([]byte, nDW*4)
		for i := 0; i < nGuW; i++ {
			binary.LittleEndian.PutUint32(guW[i*4:], uint32(e))
		}
		for i := 0; i < nDW; i++ {
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
	// slot must actually hold expert e's bytes (staging correctness), across BOTH gate|up and down.
	holds := func(s expertSlot, e int) bool {
		return s.guW.U32s()[0] == uint32(e) && s.dW.U32s()[0] == uint32(e) &&
			s.guS.U16s()[0] == uint16(e) && s.dS.U16s()[0] == uint16(e)
	}

	const N = 4
	p := newExpertPool(d, N, nGuW, nGuS, nDW, nDS, stage)

	// (1) cold start: 4 distinct experts fill the 4 free slots — 4 stages, contents correct.
	for e := 0; e < 4; e++ {
		if s := p.ensureResident(e); !holds(s, e) {
			t.Fatalf("cold start: slot for expert %d holds wrong bytes", e)
		}
	}
	if p.stages != 4 {
		t.Fatalf("cold start: want 4 stages, got %d", p.stages)
	}

	// (2) same-expert-twice: a hit does not re-stage, and returns the right contents.
	if s := p.ensureResident(2); !holds(s, 2) || p.stages != 4 {
		t.Fatalf("hit on resident expert 2 re-staged or wrong contents (stages=%d)", p.stages)
	}

	// (3) eviction under pressure: LRU order is [2,3,1,0] (0 is LRU after 0,1,2,3 then touch 2).
	// Requesting expert 4 must evict slot holding expert 0 (the LRU), stage 4, leave 1/2/3 resident.
	s4 := p.ensureResident(4)
	if !holds(s4, 4) || p.stages != 5 {
		t.Fatalf("eviction: expert 4 not staged correctly (stages=%d)", p.stages)
	}
	if _, ok := p.where[0]; ok {
		t.Fatalf("eviction: expert 0 should have been evicted (it was LRU) but is still resident")
	}
	for _, e := range []int{1, 2, 3, 4} {
		if _, ok := p.where[e]; !ok {
			t.Fatalf("eviction: expert %d should still be resident", e)
		}
	}

	// (4) evicted expert re-faults: requesting 0 again is a miss (it was evicted), re-stages.
	if s := p.ensureResident(0); !holds(s, 0) || p.stages != 6 {
		t.Fatalf("re-fault: expert 0 should re-stage after eviction (stages=%d)", p.stages)
	}

	// (5) LRU recency honoured: after (4), touching 3 then inserting a new expert must evict the
	// genuine LRU, not a recently-touched one. State now (MRU→LRU): 0,4,3,2 → 1 was evicted in (3)?
	// No: (3) evicted 0; slots hold {1,2,3,4}; (4) evicted the LRU (1) to stage 0 → slots {0,2,3,4}.
	if _, ok := p.where[1]; ok {
		t.Fatalf("re-fault: expert 1 (LRU at that point) should have been evicted")
	}
	p.ensureResident(3)       // touch 3 → MRU
	s7 := p.ensureResident(7) // miss → evicts current LRU (2), stages 7
	if !holds(s7, 7) || p.stages != 7 {
		t.Fatalf("LRU recency: expert 7 not staged (stages=%d)", p.stages)
	}
	if _, ok := p.where[2]; ok {
		t.Fatalf("LRU recency: expert 2 (LRU) should have been evicted, not the recently-touched 3")
	}
	if _, ok := p.where[3]; !ok {
		t.Fatalf("LRU recency: expert 3 was just touched and must remain resident")
	}
	t.Logf("expert pool: %d slots, %d stages over the sequence — LRU eviction + staging correct", N, p.stages)
}
