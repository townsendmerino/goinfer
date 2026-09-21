package decoder

import (
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
)

// gemma4OverlapFixture builds a small, real (not synthetic-zero) gemma4MoEWeights: dense
// weights and norms are plain heap f32 (never paged, matches production), and the nExperts
// experts' gate‖up + down tensors are REAL int4-quantized bytes written to a temp file and
// mmap'd -- so a real *expertPager (either mmap+madvise or pread-pool mode) can manage them
// exactly as newExpertPager's gemma4 branch does (one key = &w.expertsGateUp[e], two spans).
func gemma4OverlapFixture(t *testing.T, nExperts, topK int) (w *gemma4MoEWeights, arch *Architecture, path string, mapping []byte) {
	t.Helper()
	// Dims large enough that each expert's quantized span survives MappedSpan's PAGE
	// rounding (mmap.PageAlignedInterior) -- Apple Silicon's 16 KB pages mean a
	// sub-page synthetic tensor round-trips to an EMPTY span (moepaging_test.go's own
	// newRealMmapPager doc comment names this exact limitation), so this fixture must
	// be production-scale per expert, not toy-scale, for the mmap-mode pager to
	// manage it at all.
	const hidden, denseInter, moeInter, group = 1024, 768, 512, 32
	const guRows, guCols = 2 * moeInter, hidden // [1024,1024]: bpr=512, nGroups=32 -> 512 KiB
	const dnRows, dnCols = hidden, moeInter     // [1024,512]:  bpr=256, nGroups=16 -> 256 KiB
	return gemma4BuildFixture(t, nExperts, topK, hidden, denseInter, moeInter, group, guRows, guCols, dnRows, dnCols)
}

// gemma4BuildFixture is gemma4OverlapFixture's dimension-parameterized core, factored out so
// zz_lever3_bench_test.go's production-scale fixture (real ~3 MB/expert) can share it.
func gemma4BuildFixture(t *testing.T, nExperts, topK, hidden, denseInter, moeInter, group, guRows, guCols, dnRows, dnCols int) (w *gemma4MoEWeights, arch *Architecture, path string, mapping []byte) {
	t.Helper()
	guBpr, guGroups := guCols/2, guCols/group
	dnBpr, dnGroups := dnCols/2, dnCols/group
	guBytes := guRows * guBpr
	dnBytes := dnRows * dnBpr

	rng := rand.New(rand.NewSource(7))
	path = filepath.Join(t.TempDir(), "experts.bin")
	buf := make([]byte, nExperts*(guBytes+dnBytes))
	rng.Read(buf)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var err error
	mapping, err = mmap.MapReadOnly(path)
	if err != nil {
		t.Fatalf("mmap fixture: %v", err)
	}
	t.Cleanup(func() { _ = mmap.Unmap(mapping) })

	randF32 := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	randPos := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32() + 0.2
		}
		return s
	}

	expertsGateUp := make([]linalg.WeightMat, nExperts)
	expertsDown := make([]linalg.WeightMat, nExperts)
	off := 0
	for e := 0; e < nExperts; e++ {
		guQ4 := mapping[off : off+guBytes]
		off += guBytes
		dnQ4 := mapping[off : off+dnBytes]
		off += dnBytes
		expertsGateUp[e] = linalg.WrapInt4(guQ4, randPos(guRows*guGroups), guRows, guCols, group)
		expertsDown[e] = linalg.WrapInt4(dnQ4, randPos(dnRows*dnGroups), dnRows, dnCols, group)
	}

	w = &gemma4MoEWeights{
		preFFNNorm:     randPos(hidden),
		postFFNNorm1:   randPos(hidden),
		preFFNNorm2:    randPos(hidden),
		postFFNNorm2:   randPos(hidden),
		postFFNNorm:    randPos(hidden),
		mlpGate:        linalg.WrapF32(randF32(denseInter*hidden), denseInter, hidden),
		mlpUp:          linalg.WrapF32(randF32(denseInter*hidden), denseInter, hidden),
		mlpDown:        linalg.WrapF32(randF32(hidden*denseInter), hidden, denseInter),
		routerProj:     linalg.WrapF32(randF32(nExperts*hidden), nExperts, hidden),
		routerScale:    randPos(hidden),
		perExpertScale: randPos(nExperts),
		expertsGateUp:  expertsGateUp,
		expertsDown:    expertsDown,
		layerScalar:    1.0,
		denseInter:     denseInter,
		moeInter:       moeInter,
		nE:             nExperts,
		topK:           topK,
	}
	arch = &Architecture{HiddenDim: hidden, NormEps: 1e-6}
	return w, arch, path, mapping
}

// gemma4MmapPager mirrors newExpertPager's gemma4 branch exactly (one member per expert, keyed
// by &expertsGateUp[e], spans = [gateUp, down]) without needing a full Weights/LayerWeights tree
// -- matches moepaging_test.go's own newRealMmapPager convention for synthetic fixtures.
func gemma4MmapPager(w *gemma4MoEWeights, mapping []byte, budgetExperts int) *expertPager {
	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))
	type member struct {
		key   unsafe.Pointer
		spans [][]byte
	}
	members := make([]member, len(w.expertsGateUp))
	var perExpert int64
	for e := range w.expertsGateUp {
		guSpan := w.expertsGateUp[e].MappedSpan(base, end)
		dnSpan := w.expertsDown[e].MappedSpan(base, end)
		members[e] = member{key: unsafe.Pointer(&w.expertsGateUp[e]), spans: [][]byte{guSpan, dnSpan}}
		if perExpert == 0 {
			perExpert = int64(len(guSpan) + len(dnSpan))
		}
	}
	cache := mmap.NewSpanCacheWithPolicy[unsafe.Pointer](perExpert*int64(budgetExperts), mmap.EvictLeastRecent)
	for _, m := range members {
		cache.Add(m.key, m.spans)
	}
	return &expertPager{cache: cache, nExperts: len(w.expertsGateUp), total: perExpert * int64(len(w.expertsGateUp))}
}

// gemma4PoolPager builds a real pread pool over the same fixture, mirroring newExpertPager's
// pool-mode construction for the gemma4 branch (buildExpertField over both tensors, one
// poolMember per expert).
func gemma4PoolPager(t *testing.T, w *gemma4MoEWeights, path string, mapping []byte, budgetExperts, minSlots int) *expertPager {
	t.Helper()
	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))
	members := make([]poolMember, len(w.expertsGateUp))
	var perExpert int64
	for e := range w.expertsGateUp {
		guF, ok := buildExpertField(&w.expertsGateUp[e], base, end)
		if !ok {
			t.Fatalf("buildExpertField(gateUp[%d]) rejected", e)
		}
		dnF, ok := buildExpertField(&w.expertsDown[e], base, end)
		if !ok {
			t.Fatalf("buildExpertField(down[%d]) rejected", e)
		}
		members[e] = poolMember{key: unsafe.Pointer(&w.expertsGateUp[e]), fields: []expertField{guF, dnF}}
		if perExpert == 0 {
			perExpert = int64(guF.n + dnF.n)
		}
	}
	pool, err := newExpertBufferPool(path, members, perExpert*int64(budgetExperts), minSlots)
	if err != nil {
		t.Fatalf("newExpertBufferPool: %v", err)
	}
	t.Cleanup(func() { _ = pool.close() })
	return &expertPager{pool: pool, nExperts: len(w.expertsGateUp)}
}

// TestGemma4MoEFFN_overlapBitIdentical is Lever 3's correctness gate: reordering the router
// ahead of the dense branch and overlapping the expert fills with the dense branch's compute
// must not change a single output bit, in EITHER pager mode, against a fully-resident (no
// pager) reference -- across many tokens and a budget small enough to force real eviction (not
// just cold-start misses) in both modes.
func TestGemma4MoEFFN_overlapBitIdentical(t *testing.T) {
	const nExperts, topK, nTokens = 6, 2, 40
	w, arch, path, mapping := gemma4OverlapFixture(t, nExperts, topK)
	be := &cpuBackend{}

	rng := rand.New(rand.NewSource(99))
	toks := make([][]float32, nTokens)
	for i := range toks {
		h := make([]float32, arch.HiddenDim)
		for j := range h {
			h[j] = rng.Float32()*2 - 1
		}
		toks[i] = h
	}

	runAll := func(pager *expertPager) [][]float32 {
		out := make([][]float32, nTokens)
		for i, h := range toks {
			out[i] = gemma4MoEFFN(be, arch, h, w, pager)
		}
		return out
	}

	want := runAll(nil) // fully resident (no pager at all)

	mmapPager := gemma4MmapPager(w, mapping, 3) // 3-of-6 experts resident: forces real eviction
	gotMmap := runAll(mmapPager)
	if hits, misses, evictions := mmapPager.stats(); evictions == 0 {
		t.Fatalf("mmap-mode pager evicted nothing (hits=%d misses=%d) -- budget too large to exercise eviction", hits, misses)
	}

	poolPager := gemma4PoolPager(t, w, path, mapping, 3, topK)
	gotPool := runAll(poolPager)
	if hits, misses, evictions := poolPager.stats(); evictions == 0 {
		t.Fatalf("pool-mode pager evicted nothing (hits=%d misses=%d) -- budget too large to exercise eviction", hits, misses)
	}

	for tok := range toks {
		for i := range want[tok] {
			if gotMmap[tok][i] != want[tok][i] {
				t.Fatalf("mmap mode diverged at token %d elem %d: got %v want %v", tok, i, gotMmap[tok][i], want[tok][i])
			}
			if gotPool[tok][i] != want[tok][i] {
				t.Fatalf("pool mode diverged at token %d elem %d: got %v want %v", tok, i, gotPool[tok][i], want[tok][i])
			}
		}
	}
}

// TestGemma4MoEFFN_overlapConcurrentStreamsNoCorruption stresses the SAME hazard Lever 1b's own
// cross-stream test closes, at the gemma4MoEFFN call-site level instead of the pool level: many
// goroutines (simulated concurrent decode streams) repeatedly calling gemma4MoEFFN against a
// SHARED pool-mode pager deliberately smaller than the expert count, asserting every result
// matches a known-correct reference. A pool slot reused mid-computation by another stream would
// surface here as silent wrong output, not necessarily a -race failure.
//
// The reference values are precomputed SEQUENTIALLY, with no pager at all, before any goroutine
// starts and before the pool pager (which repoints w.expertsGateUp/expertsDown in place) is even
// built -- computing them concurrently WITH the pool-mode calls would itself race on those same
// shared WeightMat fields (a nil-pager call reads them as they currently stand; it does not
// wait for or protect against a concurrent pool-mode repoint). A small fixed set of tokens, not
// random-per-goroutine, so every goroutine's iterations check against the same known answers.
func TestGemma4MoEFFN_overlapConcurrentStreamsNoCorruption(t *testing.T) {
	const nExperts, topK, nToks = 8, 2, 5
	w, arch, path, mapping := gemma4OverlapFixture(t, nExperts, topK)
	be := &cpuBackend{}

	rng := rand.New(rand.NewSource(500))
	toks := make([][]float32, nToks)
	want := make([][]float32, nToks)
	for i := range toks {
		h := make([]float32, arch.HiddenDim)
		for j := range h {
			h[j] = rng.Float32()*2 - 1
		}
		toks[i] = h
		want[i] = gemma4MoEFFN(be, arch, append([]float32(nil), h...), w, nil) // sequential, no pager yet
	}

	poolPager := gemma4PoolPager(t, w, path, mapping, 2, topK) // deliberately tight: forces eviction under contention

	const goroutines, itersEach = 8, 30
	var wg sync.WaitGroup
	fails := make(chan string, goroutines*itersEach)
	for g := 0; g < goroutines; g++ {
		wg.Go(func() {
			rng := rand.New(rand.NewSource(int64(g)))
			for iter := 0; iter < itersEach; iter++ {
				i := rng.Intn(nToks)
				got := gemma4MoEFFN(be, arch, append([]float32(nil), toks[i]...), w, poolPager)
				for j := range want[i] {
					if got[j] != want[i][j] {
						fails <- "mismatch"
						return
					}
				}
			}
		})
	}
	wg.Wait()
	close(fails)
	if n := len(fails); n > 0 {
		t.Fatalf("%d/%d concurrent calls observed wrong output (cross-stream slot-reuse race)", n, goroutines*itersEach)
	}
	if _, _, evictions := poolPager.stats(); evictions == 0 {
		t.Fatal("no evictions occurred -- test didn't actually exercise slot reuse under contention")
	}
}
