package decoder

import (
	"math"
	"math/rand"
	"sync/atomic"
	"testing"
)

// syntheticGroupedArch returns an Architecture whose GQA ratio is exactly
// attnGroupedNEONSize (6) — the group size aikit's NEON port covers — so a
// test against it can actually exercise attendGroupedHeads rather than
// silently falling back to attendOneHead (R13 Gate 4's own "bit-identity
// hides dispatch inertness" concern, applied here instead of assumed away).
func syntheticGroupedArch(hd int) *Architecture {
	return &Architecture{
		NumHeads:   attnGroupedNEONSize * 2, // two kv heads' worth
		NumKVHeads: 2,
		HeadDim:    hd,
		AttnScale:  1 / math.Sqrt(float64(hd)),
	}
}

func syntheticDecodeCache(t *testing.T, arch *Architecture, nKeys int, seed int64) (*KVCache, []float32) {
	t.Helper()
	kvDim := arch.NumKVHeads * arch.HeadDim
	cache := NewKVCache(1, arch.NumKVHeads, arch.HeadDim, 0, nKeys+1, nil)
	rng := rand.New(rand.NewSource(seed))
	for range nKeys {
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		for j := range k {
			k[j] = float32(rng.NormFloat64()) * 0.1
			v[j] = float32(rng.NormFloat64()) * 0.1
		}
		cache.Append(0, k, v)
	}
	q := make([]float32, arch.NumHeads*arch.HeadDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.1
	}
	return cache, q
}

func runDecodeAttend(t *testing.T, arch *Architecture, cache *KVCache, q []float32, nKeys int) []float32 {
	t.Helper()
	hd := arch.HeadDim
	qDim := arch.NumHeads * hd
	keys, vals := cache.Keys(0), cache.Vals(0)
	ctx := make([]float32, qDim)
	pool := []headWorkerScratch{{
		qh: make([]float32, hd), ch: make([]float32, hd),
		scores:      make([]float32, nKeys),
		avAcc:       make([]float64, hd),
		groupScores: make([]float32, nKeys*attnGroupedNEONSize),
		groupCtx:    make([]float32, hd*attnGroupedNEONSize),
		groupAvAcc:  make([]float64, hd*attnGroupedNEONSize),
	}}
	// pos = nKeys-1 (0-indexed): this decode step attends to every key already
	// appended, matching causalAttention's own convention (Append happens
	// before the attend for the position being decoded).
	attendBatchedHeads(q, ctx, keys, vals, 0, cache, 0, nKeys-1, 1, true, arch, true, pool)
	return ctx
}

// TestAttendGroupedHeads_matchesPerHead is R13 Gate (2) (bit-identical, no
// golden change — this compares directly rather than against a stored
// golden) and Gate (4) (the wiring proof: attnGroupedRuns must be nonzero
// with grouping on, at a shape that is actually eligible for it).
func TestAttendGroupedHeads_matchesPerHead(t *testing.T) {
	arch := syntheticGroupedArch(64)
	const nKeys = 200 // >= attnGroupedMinKeys, so the grouped path is eligible
	cache, q := syntheticDecodeCache(t, arch, nKeys, 0x5119)

	t.Setenv("GOINFER_ATTN_GROUPED", "0")
	before := atomic.LoadInt64(&attnGroupedRuns)
	ungrouped := runDecodeAttend(t, arch, cache, q, nKeys)
	if got := atomic.LoadInt64(&attnGroupedRuns) - before; got != 0 {
		t.Fatalf("GOINFER_ATTN_GROUPED=0: attnGroupedRuns advanced by %d, want 0", got)
	}

	t.Setenv("GOINFER_ATTN_GROUPED", "1")
	before = atomic.LoadInt64(&attnGroupedRuns)
	grouped := runDecodeAttend(t, arch, cache, q, nKeys)
	// Two kv heads' worth of grouped calls, one per kv head (NumKVHeads=2).
	if got := atomic.LoadInt64(&attnGroupedRuns) - before; got != int64(arch.NumKVHeads) {
		t.Fatalf("GOINFER_ATTN_GROUPED=1: attnGroupedRuns advanced by %d, want %d (one per kv head) — the grouped path did not actually run", got, arch.NumKVHeads)
	}

	for i := range ungrouped {
		if math.Float32bits(ungrouped[i]) != math.Float32bits(grouped[i]) {
			t.Fatalf("idx %d: ungrouped %v (%08x) vs grouped %v (%08x) — not bit-identical",
				i, ungrouped[i], math.Float32bits(ungrouped[i]), grouped[i], math.Float32bits(grouped[i]))
		}
	}
}

// TestAttendGroupedHeads_belowNWinGateStaysUngrouped is the nWin gate check:
// below attnGroupedMinKeys, the grouped path must not run even when enabled,
// per the brief's "below which the per-head path runs unchanged".
func TestAttendGroupedHeads_belowNWinGateStaysUngrouped(t *testing.T) {
	arch := syntheticGroupedArch(64)
	nKeys := attnGroupedMinKeys - 1
	cache, q := syntheticDecodeCache(t, arch, nKeys, 0x511A)

	t.Setenv("GOINFER_ATTN_GROUPED", "1")
	before := atomic.LoadInt64(&attnGroupedRuns)
	runDecodeAttend(t, arch, cache, q, nKeys)
	if got := atomic.LoadInt64(&attnGroupedRuns) - before; got != 0 {
		t.Fatalf("nKeys=%d (< attnGroupedMinKeys=%d), GOINFER_ATTN_GROUPED=1: attnGroupedRuns advanced by %d, want 0 — the nWin gate did not hold",
			nKeys, attnGroupedMinKeys, got)
	}
}

// TestAttendGroupedHeads_concurrentWorkers exercises the ACTUAL fan-out path
// (len(pool) > 1, one goroutine per worker) rather than the serial pool[0]
// arm every other test in this file takes — R13 Gate (3) needs `go test
// -race` to see the concurrent grouped calls' writes, which a single-slot
// pool never reaches. Two workers, two kv heads: each worker's contiguous
// head range is exactly one kv group, so both take the grouped path.
func TestAttendGroupedHeads_concurrentWorkers(t *testing.T) {
	arch := syntheticGroupedArch(64)
	const nKeys = 200
	cache, q := syntheticDecodeCache(t, arch, nKeys, 0x511C)
	hd := arch.HeadDim
	qDim := arch.NumHeads * hd
	keys, vals := cache.Keys(0), cache.Vals(0)

	newPool := func() []headWorkerScratch {
		pool := make([]headWorkerScratch, arch.NumKVHeads)
		for i := range pool {
			pool[i] = headWorkerScratch{
				qh: make([]float32, hd), ch: make([]float32, hd),
				scores:      make([]float32, nKeys),
				avAcc:       make([]float64, hd),
				groupScores: make([]float32, nKeys*attnGroupedNEONSize),
				groupCtx:    make([]float32, hd*attnGroupedNEONSize),
				groupAvAcc:  make([]float64, hd*attnGroupedNEONSize),
			}
		}
		return pool
	}

	t.Setenv("GOINFER_ATTN_GROUPED", "0")
	ungrouped := make([]float32, qDim)
	attendBatchedHeads(q, ungrouped, keys, vals, 0, cache, 0, nKeys-1, 1, true, arch, true, newPool())

	t.Setenv("GOINFER_ATTN_GROUPED", "1")
	before := atomic.LoadInt64(&attnGroupedRuns)
	grouped := make([]float32, qDim)
	attendBatchedHeads(q, grouped, keys, vals, 0, cache, 0, nKeys-1, 1, true, arch, true, newPool())
	if got := atomic.LoadInt64(&attnGroupedRuns) - before; got != int64(arch.NumKVHeads) {
		t.Fatalf("concurrent workers: attnGroupedRuns advanced by %d, want %d", got, arch.NumKVHeads)
	}

	for i := range ungrouped {
		if math.Float32bits(ungrouped[i]) != math.Float32bits(grouped[i]) {
			t.Fatalf("idx %d: ungrouped %v vs grouped %v — not bit-identical under concurrent fan-out", i, ungrouped[i], grouped[i])
		}
	}
}

// TestAttendGroupedHeads_wrongGroupSizeStaysUngrouped: a GQA ratio other than
// attnGroupedNEONSize must never take the grouped path, even above the nWin
// gate and with grouping enabled — aikit's NEON port is specialized to
// exactly that group size (see attnGroupedNEONSize's own comment).
func TestAttendGroupedHeads_wrongGroupSizeStaysUngrouped(t *testing.T) {
	arch := &Architecture{NumHeads: 8, NumKVHeads: 2, HeadDim: 64, AttnScale: 1 / math.Sqrt(64)} // group=4
	const nKeys = 200
	cache, q := syntheticDecodeCache(t, arch, nKeys, 0x511B)

	t.Setenv("GOINFER_ATTN_GROUPED", "1")
	before := atomic.LoadInt64(&attnGroupedRuns)
	runDecodeAttend(t, arch, cache, q, nKeys)
	if got := atomic.LoadInt64(&attnGroupedRuns) - before; got != 0 {
		t.Fatalf("group=4 (!= attnGroupedNEONSize=%d): attnGroupedRuns advanced by %d, want 0", attnGroupedNEONSize, got)
	}
}
