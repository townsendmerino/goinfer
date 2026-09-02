package decoder

import (
	"math/rand"
	"testing"
)

// C-05: A CANCELLED BATCHED PREFILL ON A WRAPPED RING LEAVES STALE K/V THAT THE NEXT WARM TURN
// READS AS HISTORY.
//
// G18 made the batched sweep abortable per LAYER. Layers below the abort point have already
// commitBatch'd — their ring count is startPos+K — while c.pos is still startPos, because advanceTo
// runs only at the end of a completed sweep. reconcile then truncates to c.pos, which rewinds those
// rings by K on a wrapped window: ring.truncate cannot restore the rows the commit evicted and
// returns false. reconcile discarded that bool, so the session stayed WARM over K/V RoPE'd at
// positions the next turn will read as much earlier history. The request that was cancelled reports
// a clean end, so nothing else in the stack notices.
//
// This drives reconcile directly. A full serve round-trip would need a client that disconnects
// mid-prefill; the cache state it produces is what is built here, and it is the state that matters.
func TestSession_cancelledSweepOnAWrappedRingGoesCold(t *testing.T) {
	const nKV, hd, W = 1, 4, 64
	const startPos, K = 200, 32 // startPos > W, so the ring has wrapped
	kvDim := nKV * hd
	arch := ringTestArch(1, nKV, hd, W, func(int) bool { return false }) // all local

	cache := NewKVCache(1, nKV, hd, W, startPos+K)
	cache.enableRings(W, arch.isGlobalLayer)
	rng := rand.New(rand.NewSource(53))

	// Turn 1 completed: startPos positions committed and advanced.
	hist := randVec(rng, startPos*kvDim)
	cache.commitBatch(0, 0, startPos, hist, hist)
	cache.advanceTo(startPos)
	if cache.rings[0].count <= W {
		t.Fatalf("the premise broke: ring count %d has not wrapped past W=%d", cache.rings[0].count, W)
	}

	// Turn 2, CANCELLED mid-sweep: layer 0 committed, advanceTo never ran.
	next := randVec(rng, K*kvDim)
	cache.commitBatch(0, startPos, K, next, next)
	if cache.Pos() != startPos {
		t.Fatalf("the premise broke: c.pos = %d, want %d — advanceTo must NOT have run",
			cache.Pos(), startPos)
	}
	if cache.rings[0].count != startPos+K {
		t.Fatalf("the premise broke: ring count = %d, want %d — the layer must have committed",
			cache.rings[0].count, startPos+K)
	}

	// The token stream the caller emitted claims more than the cache holds.
	seq := make([]int, startPos+K)
	for i := range seq {
		seq[i] = i + 1
	}
	s := &Session{cache: cache, tokens: seq}
	s.reconcile(seq)

	if s.tokens != nil {
		t.Errorf("session stayed warm with %d token(s) after an INEXACT rewind. The next request "+
			"that extends this conversation matches that prefix, rewinds to a no-op \"exact\", and "+
			"reads the cancelled turn's rows as history for earlier positions", len(s.tokens))
	}
	if p := cache.Pos(); p != 0 {
		t.Errorf("cache.Pos() = %d after an inexact rewind, want 0 (cold) — a ring cannot restore "+
			"the rows the aborted commit evicted, so cold prefill is the only exact answer", p)
	}
}

// The other half of the same predicate: an EXACT rewind must leave the session warm. A fix that
// went cold unconditionally would pass the test above and throw away every session's KV reuse.
func TestSession_exactRewindStaysWarm(t *testing.T) {
	const nKV, hd, W = 1, 4, 64
	kvDim := nKV * hd
	arch := ringTestArch(1, nKV, hd, W, func(int) bool { return false })

	cache := NewKVCache(1, nKV, hd, W, 128)
	cache.enableRings(W, arch.isGlobalLayer)
	rng := rand.New(rand.NewSource(59))
	// Never wraps: count stays <= W, so every rewind is exact.
	const n = 32
	cache.commitBatch(0, 0, n, randVec(rng, n*kvDim), randVec(rng, n*kvDim))
	cache.advanceTo(n)

	seq := make([]int, n)
	for i := range seq {
		seq[i] = i + 1
	}
	s := &Session{cache: cache, tokens: seq}
	s.reconcile(seq)

	if len(s.tokens) != n {
		t.Errorf("session went cold (%d tokens) on an EXACT rewind — every warm reuse would be "+
			"thrown away", len(s.tokens))
	}
	if p := cache.Pos(); p != n {
		t.Errorf("cache.Pos() = %d, want %d", p, n)
	}
}
