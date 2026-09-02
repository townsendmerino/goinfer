package decoder

import (
	"math/rand"
	"testing"
)

// C-04: THE POOL IS SIZED FROM THE GLOBAL KEY COUNT AND THE TILE IS COMPUTED FROM THE PER-LAYER ONE.
//
// forwardLayersN sizes the head-worker pool once, from maxKeys = startPos+K, and its comment states
// the premise outright: "nKeys = startPos+K is the same for every layer in this sweep". It is not.
// A LOCAL (sliding-window) layer whose ring has wrapped assembles a SHORTER window — nKeys = W-1+K
// — and attendOneHead recomputes its row tile from that. attnRowTile is INVERSE in nKeys, so a
// shorter window yields a LARGER tile than the slot's qh was allocated to hold, and the Q gather
// slices past its length: `panic: slice bounds out of range`, in a worker goroutine on the fan-out
// arm and in the Generate goroutine on the serial one. Neither is recovered.
//
// A COLD PREFILL ALWAYS HAS nKeys == maxKeys, which is why every golden and benchmark misses this.
// It needs a WARM session: turn 1 leaves the ring past W, turn 2 appends a long suffix.
func TestAttnTile_ringLayerDoesNotOutgrowThePoolSlot(t *testing.T) {
	// hd tiny keeps the arithmetic cheap; the defect is in the row bookkeeping, not the math.
	// K*maxKeys must exceed attnScoreTileBytes/4 (2Mi) or attnRowTile returns K for both and the
	// two tiles cannot differ — that is the "cold prefill misses it" condition, stated as numbers.
	const nH, nKV, hd, W = 1, 1, 4, 512
	const startPos, K = 600, 2048
	kvDim, qDim := nKV*hd, nH*hd
	maxKeys := startPos + K

	if got := attnRowTile(K, maxKeys); got >= K {
		t.Fatalf("the premise broke: attnRowTile(K=%d, maxKeys=%d) = %d, not tiled — "+
			"K*maxKeys must exceed %d for the two tiles to differ", K, maxKeys, got, attnScoreTileBytes/4)
	}
	arch := ringTestArch(nH, nKV, hd, W, func(int) bool { return false }) // all local

	ring := NewKVCache(1, nKV, hd, W, startPos+K)
	ring.enableRings(W, arch.isGlobalLayer)
	rng := rand.New(rand.NewSource(41))

	// Turn 1: fill the ring past its window, so the assembled window is shorter than maxKeys.
	hist := randVec(rng, startPos*kvDim)
	ring.commitBatch(0, 0, startPos, hist, hist)
	ring.advanceTo(startPos)

	// Turn 2: a long divergent suffix — the batched prefill of K new positions.
	k, v, q := randVec(rng, K*kvDim), randVec(rng, K*kvDim), randVec(rng, K*qDim)
	alk := make([]float32, maxKeys*kvDim)
	alv := make([]float32, maxKeys*kvDim)
	base, nRows := ring.batchReadLocal(0, startPos, K, k, v, alk, alv)
	if nRows >= maxKeys {
		t.Fatalf("the premise broke: the ring window is %d rows, not shorter than maxKeys %d — "+
			"the ring has not wrapped, so this is a cold-prefill shape", nRows, maxKeys)
	}
	if attnRowTile(K, nRows) <= attnRowTile(K, maxKeys) {
		t.Fatalf("the premise broke: per-layer tile %d does not exceed the pool's %d",
			attnRowTile(K, nRows), attnRowTile(K, maxKeys))
	}

	// Sized exactly as forwardLayersN sizes it: from maxKeys, once, for the whole sweep.
	pool := newHeadWorkerPool(1, K, maxKeys, hd)
	ctx := make([]float32, K*qDim)
	attendBatchedHeads(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, ring, 0, startPos, K,
		false, arch, false, pool)

	// NOT MERELY "IT DID NOT PANIC". The clamp changes how many rows a pass handles, and a fix
	// that stopped the crash while changing the ANSWER would satisfy the weaker claim. Tiling
	// splits independent outputs — no key-dimension split — so a different tile must be
	// bit-identical. Reference: a pool sized from this layer's own key count, where no clamp
	// applies and the tile is whatever attnRowTile asked for.
	ref := newHeadWorkerPool(1, K, nRows, hd)
	refCtx := make([]float32, K*qDim)
	attendBatchedHeads(q, refCtx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, ring, 0, startPos, K,
		false, arch, false, ref)
	if attendTileFor(&pool[0], K, nRows, hd) == attendTileFor(&ref[0], K, nRows, hd) {
		t.Fatalf("the premise broke: both pools chose tile %d, so this comparison comes for free",
			attendTileFor(&pool[0], K, nRows, hd))
	}
	for i := range refCtx {
		if ctx[i] != refCtx[i] {
			t.Fatalf("ctx[%d] = %v with the clamped tile, %v with the unclamped one — the row tile "+
				"must be numerically inert (it splits independent outputs)", i, ctx[i], refCtx[i])
		}
	}
}

// The same shape on the acc64 arm, which the audit notes is equally affected: it shares qh.
func TestAttnTile_ringLayerDoesNotOutgrowThePoolSlot_acc64(t *testing.T) {
	const nH, nKV, hd, W = 1, 1, 4, 512
	const startPos, K = 600, 2048
	kvDim, qDim := nKV*hd, nH*hd
	maxKeys := startPos + K
	arch := ringTestArch(nH, nKV, hd, W, func(int) bool { return false })

	ring := NewKVCache(1, nKV, hd, W, startPos+K)
	ring.enableRings(W, arch.isGlobalLayer)
	rng := rand.New(rand.NewSource(42))
	hist := randVec(rng, startPos*kvDim)
	ring.commitBatch(0, 0, startPos, hist, hist)
	ring.advanceTo(startPos)

	k, v, q := randVec(rng, K*kvDim), randVec(rng, K*kvDim), randVec(rng, K*qDim)
	alk := make([]float32, maxKeys*kvDim)
	alv := make([]float32, maxKeys*kvDim)
	base, nRows := ring.batchReadLocal(0, startPos, K, k, v, alk, alv)
	pool := newHeadWorkerPool(1, K, maxKeys, hd)
	ctx := make([]float32, K*qDim)
	attendBatchedHeads(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, ring, 0, startPos, K,
		false, arch, true, pool)
}
