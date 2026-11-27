package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Increment 1 gate for ring-buffer KV storage:
// model-free, bit-exact. A ring (sliding-window) cache must produce attention
// output byte-identical to the append-forever cache it replaces — the ring only
// drops keys provably outside every future query's window, so outputs can't
// move. These run without a checkpoint (the gemma/mellum window goldens, which
// also gate this, need their models). W is kept small so the ring wraps many
// times within the sequence.

func ringTestArch(nH, nKV, hd, window int, globalLayer func(int) bool) *darch.Architecture {
	return &darch.Architecture{
		NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		AttnScale:     1.0 / math.Sqrt(float64(hd)),
		SlidingWindow: window,
		LayerIsGlobal: globalLayer,
	}
}

func randVec(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64()) * 0.3
	}
	return s
}

// TestRing_decodeBitExact: decode (attendQuery, single query) over 4×W positions
// on a 2-layer model (layer 0 local sliding-window, layer 1 global). The ring
// cache's per-token attention context must equal the append-forever cache's.
func TestRing_decodeBitExact(t *testing.T) {
	const nLayers, nH, nKV, hd, W, N = 2, 4, 2, 8, 4, 17
	kvDim, qDim := nKV*hd, nH*hd
	arch := ringTestArch(nH, nKV, hd, W, func(l int) bool { return l == 1 })

	ringC := NewKVCache(nLayers, nKV, hd, W, N+1, nil)
	ringC.c.EnableRings(W, arch.IsGlobalLayer)
	ringC.c.Scr = newDecodeScratch(arch, ringC.c.Knobs)
	fullC := NewKVCache(nLayers, nKV, hd, W, N+1, nil)
	fullC.c.Scr = newDecodeScratch(arch, fullC.c.Knobs)

	rng := rand.New(rand.NewSource(11))
	for pos := range N {
		for l := range nLayers {
			global := arch.IsGlobalLayer(l)
			k, v, q := randVec(rng, kvDim), randVec(rng, kvDim), randVec(rng, qDim)

			rk, rv, rq := slices.Clone(k), slices.Clone(v), slices.Clone(q)
			ringC.Append(l, rk, rv)
			rCtx := make([]float32, qDim)
			attendQuery(rq, rCtx, ringC.c.Scr.ScoresBuf(ringC.c.StoredRows(l, kvDim)), ringC, l, pos, global, arch)

			fullC.Append(l, k, v)
			fCtx := make([]float32, qDim)
			attendQuery(q, fCtx, fullC.c.Scr.ScoresBuf(fullC.c.StoredRows(l, kvDim)), fullC, l, pos, global, arch)

			for j := range fCtx {
				if rCtx[j] != fCtx[j] {
					t.Fatalf("pos %d layer %d: ring ctx[%d]=%v != full %v (NOT bit-exact)", pos, l, j, rCtx[j], fCtx[j])
				}
			}
		}
	}
	// Memory: the local layer holds exactly W rows; the global layer grows with N.
	if got := len(ringC.c.Rings[0].K); got != W*kvDim {
		t.Errorf("local ring storage = %d floats, want W*kvDim = %d (O(W), not O(context))", got, W*kvDim)
	}
	if ringC.c.Rings[1] != nil {
		t.Errorf("global layer 1 must not have a ring")
	}
	if got := len(fullC.c.KeysBuf[0]) / kvDim; got != N {
		t.Errorf("append-forever sanity: full cache stored %d rows, want %d", got, N)
	}
}

// TestRing_batchedBitExact: the batched prefill path (attendBatchedHeads) over a
// single K=N>W batch. The ring assembles a [base, N) window (history + new rows)
// and reads it with a base offset; the result must equal append-forever (base 0,
// full keys). This is the path the doc's naive "s%W" mapping gets wrong.
func TestRing_batchedBitExact(t *testing.T) {
	const nH, nKV, hd, W, K = 4, 2, 8, 4, 19 // K ≫ W: the prefill-eviction case
	kvDim, qDim := nKV*hd, nH*hd
	arch := ringTestArch(nH, nKV, hd, W, func(int) bool { return false }) // all local

	rng := rand.New(rand.NewSource(23))
	q := randVec(rng, K*qDim)
	k := randVec(rng, K*kvDim)
	v := randVec(rng, K*kvDim)

	// Reference: append-forever cache holding all K rows, base 0.
	full := NewKVCache(1, nKV, hd, W, K, nil)
	full.c.KeysBuf[0] = append(full.c.KeysBuf[0], k...)
	full.c.ValsBuf[0] = append(full.c.ValsBuf[0], v...)
	full.c.PosN = K
	bufs := func(maxKeys int) []headWorkerScratch {
		return []headWorkerScratch{{
			QH: make([]float32, K*hd), KH: make([]float32, maxKeys*hd), VT: make([]float32, maxKeys*hd),
			Scores: make([]float32, K*maxKeys), CH: make([]float32, K*hd),
		}}
	}
	fCtx := make([]float32, K*qDim)
	attendBatchedHeads(q, fCtx, full.c.KeysBuf[0], full.c.ValsBuf[0], 0, full, 0, 0, K, false, arch, false, bufs(K))

	// Ring: empty cache (startPos 0), assemble [base, K) from (empty history + new).
	ring := NewKVCache(1, nKV, hd, W, K, nil)
	ring.c.EnableRings(W, arch.IsGlobalLayer)
	alk := make([]float32, K*kvDim)
	alv := make([]float32, K*kvDim)
	base, nRows := ring.c.BatchReadLocal(0, 0, K, k, v, alk, alv)
	rCtx := make([]float32, K*qDim)
	attendBatchedHeads(q, rCtx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, ring, 0, 0, K, false, arch, false, bufs(nRows))

	for j := range fCtx {
		if rCtx[j] != fCtx[j] {
			t.Fatalf("ctx[%d]: ring %v != append-forever %v (batched prefill NOT bit-exact, K=%d>W=%d)", j, rCtx[j], fCtx[j], K, W)
		}
	}
	ring.c.CommitBatch(0, 0, K, k, v)
	if ring.c.Rings[0].Count != K {
		t.Errorf("after commit, ring count = %d, want %d", ring.c.Rings[0].Count, K)
	}
	if len(ring.c.Rings[0].K) != W*kvDim {
		t.Errorf("ring storage = %d, want W*kvDim = %d", len(ring.c.Rings[0].K), W*kvDim)
	}
}

// TestRing_moeDecodeBitExact: the Mellum2 decode path — single-token attention
// routed through attendBatchedHeads (K=1) on a local layer, step by step. Each
// step the ring assembles its [base, pos] window (history + this token) and
// commits; the context must match the append-forever cache fed the same tokens
// through the same K=1 batched kernel. Covers the per-token decode's MoE+ring path.
func TestRing_moeDecodeBitExact(t *testing.T) {
	const nH, nKV, hd, W, N = 4, 2, 8, 4, 17
	kvDim, qDim := nKV*hd, nH*hd
	arch := ringTestArch(nH, nKV, hd, W, func(int) bool { return false }) // all local

	ring := NewKVCache(1, nKV, hd, W, N+1, nil)
	ring.c.EnableRings(W, arch.IsGlobalLayer)
	full := NewKVCache(1, nKV, hd, W, N+1, nil)

	mkbufs := func(nKeys int) []headWorkerScratch {
		return []headWorkerScratch{{
			QH: make([]float32, hd), KH: make([]float32, nKeys*hd), VT: make([]float32, nKeys*hd),
			Scores: make([]float32, nKeys), CH: make([]float32, hd), AvAcc: make([]float64, hd),
		}}
	}
	rng := rand.New(rand.NewSource(31))
	for pos := range N {
		k, v, q := randVec(rng, kvDim), randVec(rng, kvDim), randVec(rng, qDim)

		// Ring: defer write, assemble [base,pos] window, attend, commit.
		rk, rv := slices.Clone(k), slices.Clone(v)
		rows := pos - ring.WindowStart(pos, false) + 1
		lk := make([]float32, rows*kvDim)
		lv := make([]float32, rows*kvDim)
		base, nKeys := ring.c.BatchReadLocal(0, pos, 1, rk, rv, lk, lv)
		rCtx := make([]float32, qDim)
		attendBatchedHeads(q, rCtx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], base, ring, 0, pos, 1, false, arch, true, mkbufs(nKeys))
		ring.c.CommitBatch(0, pos, 1, rk, rv)

		// Reference: append-forever, attend K=1 base 0 over all stored keys.
		full.c.KeysBuf[0] = append(full.c.KeysBuf[0], k...)
		full.c.ValsBuf[0] = append(full.c.ValsBuf[0], v...)
		full.c.PosN = pos + 1
		fCtx := make([]float32, qDim)
		nf := pos + 1
		attendBatchedHeads(q, fCtx, full.c.KeysBuf[0], full.c.ValsBuf[0], 0, full, 0, pos, 1, false, arch, true, mkbufs(nf))

		for j := range fCtx {
			if rCtx[j] != fCtx[j] {
				t.Fatalf("pos %d: MoE-decode ring ctx[%d]=%v != full %v", pos, j, rCtx[j], fCtx[j])
			}
		}
	}
}
