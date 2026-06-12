package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// Increment 1 gate for ring-buffer KV storage (docs/task-kv-ring-eviction.md):
// model-free, bit-exact. A ring (sliding-window) cache must produce attention
// output byte-identical to the append-forever cache it replaces — the ring only
// drops keys provably outside every future query's window, so outputs can't
// move. These run without a checkpoint (the gemma/mellum window goldens, which
// also gate this, need their models). W is kept small so the ring wraps many
// times within the sequence.

func ringTestArch(nH, nKV, hd, window int, globalLayer func(int) bool) *Architecture {
	return &Architecture{
		NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		AttnScale:     1.0 / math.Sqrt(float64(hd)),
		SlidingWindow: window,
		layerIsGlobal: globalLayer,
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

	ringC := NewKVCache(nLayers, nKV, hd, W, N+1)
	ringC.enableRings(W, arch.isGlobalLayer)
	ringC.scr = newDecodeScratch(arch)
	fullC := NewKVCache(nLayers, nKV, hd, W, N+1)
	fullC.scr = newDecodeScratch(arch)

	rng := rand.New(rand.NewSource(11))
	for pos := 0; pos < N; pos++ {
		for l := range nLayers {
			global := arch.isGlobalLayer(l)
			k, v, q := randVec(rng, kvDim), randVec(rng, kvDim), randVec(rng, qDim)

			rk, rv, rq := append([]float32(nil), k...), append([]float32(nil), v...), append([]float32(nil), q...)
			ringC.Append(l, rk, rv)
			rCtx := make([]float32, qDim)
			attendQuery(rq, rCtx, ringC.scr.scoresBuf(ringC.storedRows(l, kvDim)), ringC, l, pos, global, arch)

			fullC.Append(l, k, v)
			fCtx := make([]float32, qDim)
			attendQuery(q, fCtx, fullC.scr.scoresBuf(fullC.storedRows(l, kvDim)), fullC, l, pos, global, arch)

			for j := range fCtx {
				if rCtx[j] != fCtx[j] {
					t.Fatalf("pos %d layer %d: ring ctx[%d]=%v != full %v (NOT bit-exact)", pos, l, j, rCtx[j], fCtx[j])
				}
			}
		}
	}
	// Memory: the local layer holds exactly W rows; the global layer grows with N.
	if got := len(ringC.rings[0].k); got != W*kvDim {
		t.Errorf("local ring storage = %d floats, want W*kvDim = %d (O(W), not O(context))", got, W*kvDim)
	}
	if ringC.rings[1] != nil {
		t.Errorf("global layer 1 must not have a ring")
	}
	if got := len(fullC.keys[0]) / kvDim; got != N {
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
	full := NewKVCache(1, nKV, hd, W, K)
	full.keys[0] = append(full.keys[0], k...)
	full.vals[0] = append(full.vals[0], v...)
	full.pos = K
	bufs := func(maxKeys int) (qh, kh, vt, sc, ch []float32) {
		return make([]float32, K*hd), make([]float32, maxKeys*hd), make([]float32, maxKeys*hd), make([]float32, K*maxKeys), make([]float32, K*hd)
	}
	fCtx := make([]float32, K*qDim)
	qh, kh, vt, sc, ch := bufs(K)
	attendBatchedHeads(q, fCtx, full.keys[0], full.vals[0], 0, full, 0, 0, K, false, arch, false, qh, kh, vt, sc, ch)

	// Ring: empty cache (startPos 0), assemble [base, K) from (empty history + new).
	ring := NewKVCache(1, nKV, hd, W, K)
	ring.enableRings(W, arch.isGlobalLayer)
	alk := make([]float32, K*kvDim)
	alv := make([]float32, K*kvDim)
	base, nRows := ring.batchReadLocal(0, 0, K, k, v, alk, alv)
	rCtx := make([]float32, K*qDim)
	qh, kh, vt, sc, ch = bufs(nRows)
	attendBatchedHeads(q, rCtx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, ring, 0, 0, K, false, arch, false, qh, kh, vt, sc, ch)

	for j := range fCtx {
		if rCtx[j] != fCtx[j] {
			t.Fatalf("ctx[%d]: ring %v != append-forever %v (batched prefill NOT bit-exact, K=%d>W=%d)", j, rCtx[j], fCtx[j], K, W)
		}
	}
	ring.commitBatch(0, 0, K, k, v)
	if ring.rings[0].count != K {
		t.Errorf("after commit, ring count = %d, want %d", ring.rings[0].count, K)
	}
	if len(ring.rings[0].k) != W*kvDim {
		t.Errorf("ring storage = %d, want W*kvDim = %d", len(ring.rings[0].k), W*kvDim)
	}
}

// TestRing_moeDecodeBitExact: the Mellum2 decode path — single-token attention
// routed through attendBatchedHeads (K=1) on a local layer, step by step. Each
// step the ring assembles its [base, pos] window (history + this token) and
// commits; the context must match the append-forever cache fed the same tokens
// through the same K=1 batched kernel. Covers causalAttention's MoE+ring branch.
func TestRing_moeDecodeBitExact(t *testing.T) {
	const nH, nKV, hd, W, N = 4, 2, 8, 4, 17
	kvDim, qDim := nKV*hd, nH*hd
	arch := ringTestArch(nH, nKV, hd, W, func(int) bool { return false }) // all local

	ring := NewKVCache(1, nKV, hd, W, N+1)
	ring.enableRings(W, arch.isGlobalLayer)
	full := NewKVCache(1, nKV, hd, W, N+1)

	mkbufs := func(nKeys int) (qh, kh, vt, sc, ch []float32) {
		return make([]float32, hd), make([]float32, nKeys*hd), make([]float32, nKeys*hd), make([]float32, nKeys), make([]float32, hd)
	}
	rng := rand.New(rand.NewSource(31))
	for pos := 0; pos < N; pos++ {
		k, v, q := randVec(rng, kvDim), randVec(rng, kvDim), randVec(rng, qDim)

		// Ring: defer write, assemble [base,pos] window, attend, commit.
		rk, rv := append([]float32(nil), k...), append([]float32(nil), v...)
		rows := pos - ring.WindowStart(pos, false) + 1
		lk := make([]float32, rows*kvDim)
		lv := make([]float32, rows*kvDim)
		base, nKeys := ring.batchReadLocal(0, pos, 1, rk, rv, lk, lv)
		rCtx := make([]float32, qDim)
		qh, kh, vt, sc, ch := mkbufs(nKeys)
		attendBatchedHeads(q, rCtx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], base, ring, 0, pos, 1, false, arch, true, qh, kh, vt, sc, ch)
		ring.commitBatch(0, pos, 1, rk, rv)

		// Reference: append-forever, attend K=1 base 0 over all stored keys.
		full.keys[0] = append(full.keys[0], k...)
		full.vals[0] = append(full.vals[0], v...)
		full.pos = pos + 1
		fCtx := make([]float32, qDim)
		nf := pos + 1
		qh, kh, vt, sc, ch = mkbufs(nf)
		attendBatchedHeads(q, fCtx, full.keys[0], full.vals[0], 0, full, 0, pos, 1, false, arch, true, qh, kh, vt, sc, ch)

		for j := range fCtx {
			if rCtx[j] != fCtx[j] {
				t.Fatalf("pos %d: MoE-decode ring ctx[%d]=%v != full %v", pos, j, rCtx[j], fCtx[j])
			}
		}
	}
}
