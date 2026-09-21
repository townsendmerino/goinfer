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

// int4PoolFixture is a real, file-backed (not heap) bank of nExperts synthetic int4 experts,
// each rows x cols with the given group size -- real random nibble bytes on disk, mmap'd, so
// buildExpertField's containment check and the pool's own pread both exercise genuine I/O
// against the same underlying file (mirroring a real .giw: nibbles mmap-aliased, scales a heap
// copy). Returns the file path (pread source), the mmap (kept alive via t.Cleanup), and one
// "canonical" WeightMat per expert built directly over the mmap -- the ground truth every
// pool-refilled WeightMat must match byte-for-byte and compute-identically.
func int4PoolFixture(t *testing.T, nExperts int) (path string, mapping []byte, canonical []linalg.WeightMat) {
	t.Helper()
	const rows, cols, group = 4, 64, 32
	const nGroups, bpr = 2, 32 // groupsFor(64, 32)
	const perExpert = rows * bpr

	path = filepath.Join(t.TempDir(), "experts.bin")
	rng := rand.New(rand.NewSource(1))
	buf := make([]byte, nExperts*perExpert)
	rng.Read(buf)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	mapping, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatalf("mmap fixture: %v", err)
	}
	t.Cleanup(func() { _ = mmap.Unmap(mapping) })

	canonical = make([]linalg.WeightMat, nExperts)
	for i := range nExperts {
		scales := make([]float32, rows*nGroups)
		for j := range scales {
			scales[j] = rng.Float32() + 0.1
		}
		q4 := mapping[i*perExpert : (i+1)*perExpert]
		canonical[i] = linalg.WrapInt4(q4, scales, rows, cols, group)
	}
	return path, mapping, canonical
}

// buildPoolMembers runs each canonical WeightMat through buildExpertField (the real
// production resolution path newExpertPager uses) against its own mmap window, returning the
// poolMembers plus a parallel slice of the mutable WeightMat copies the pool will repoint --
// analogous to lw.Experts[e].Gate being the single shared, in-place-repointed instance in
// production.
func buildPoolMembers(t *testing.T, canonical []linalg.WeightMat, mapping []byte) ([]poolMember, []linalg.WeightMat, []unsafe.Pointer) {
	t.Helper()
	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))
	live := make([]linalg.WeightMat, len(canonical))
	copy(live, canonical)
	keys := make([]unsafe.Pointer, len(canonical))
	members := make([]poolMember, len(canonical))
	for i := range live {
		keys[i] = unsafe.Pointer(&live[i])
		f, ok := buildExpertField(&live[i], base, end)
		if !ok {
			t.Fatalf("buildExpertField rejected expert %d (expected mmap-contained int4 canonical)", i)
		}
		members[i] = poolMember{key: keys[i], fields: []expertField{f}}
	}
	return members, live, keys
}

func matmulOut(wm *linalg.WeightMat, x []float32) []float32 {
	out := make([]float32, wm.Rows())
	wm.MatmulBT(x, out, 1)
	return out
}

// TestExpertBufferPool_refillIsByteExact is Lever 1b's bit-exactness gate at the pool level:
// a pread-refilled WeightMat must reproduce the same packed nibbles a direct mmap alias of the
// same file region holds, byte for byte and (belt-and-suspenders) through an actual matmul.
func TestExpertBufferPool_refillIsByteExact(t *testing.T) {
	path, mapping, canonical := int4PoolFixture(t, 3)
	members, live, keys := buildPoolMembers(t, canonical, mapping)

	pool, err := newExpertBufferPool(path, members, int64(len(members))*int64(1<<20), 1)
	if err != nil {
		t.Fatalf("newExpertBufferPool: %v", err)
	}
	t.Cleanup(func() { _ = pool.close() })

	x := make([]float32, canonical[0].Cols())
	rng := rand.New(rand.NewSource(2))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	for i, k := range keys {
		if err := pool.ensure(k); err != nil {
			t.Fatalf("ensure(%d): %v", i, err)
		}
		wantQ4, wantScales, wantGroup, ok := canonical[i].Int4()
		if !ok {
			t.Fatalf("canonical[%d] not int4", i)
		}
		gotQ4, gotScales, gotGroup, ok := live[i].Int4()
		if !ok {
			t.Fatalf("live[%d] not int4 after refill", i)
		}
		if gotGroup != wantGroup || len(gotQ4) != len(wantQ4) {
			t.Fatalf("expert %d shape mismatch: got group=%d len=%d, want group=%d len=%d", i, gotGroup, len(gotQ4), wantGroup, len(wantQ4))
		}
		for j := range wantQ4 {
			if gotQ4[j] != wantQ4[j] {
				t.Fatalf("expert %d nibble byte %d: got %02x want %02x (pread refill diverged from mmap source)", i, j, gotQ4[j], wantQ4[j])
			}
		}
		for j := range wantScales {
			if gotScales[j] != wantScales[j] {
				t.Fatalf("expert %d scale %d: got %v want %v", i, j, gotScales[j], wantScales[j])
			}
		}
		wantOut := matmulOut(&canonical[i], x)
		gotOut := matmulOut(&live[i], x)
		for j := range wantOut {
			if gotOut[j] != wantOut[j] {
				t.Fatalf("expert %d matmul output row %d: got %v want %v (bit-exactness broken through compute, not just raw bytes)", i, j, gotOut[j], wantOut[j])
			}
		}
	}
	if hits, misses, evictions := pool.stats(); hits != 0 || misses != int64(len(keys)) || evictions != 0 {
		t.Fatalf("stats: hits=%d misses=%d evictions=%d, want hits=0 misses=%d evictions=0", hits, misses, evictions, len(keys))
	}
}

// TestExpertBufferPool_topKNeverSelfEvicts is the minSlots guard's gate: newExpertBufferPool
// raises the slot count to at least minSlots (the model's real top-k), so touching that many
// DISTINCT never-before-seen experts in one batch -- exactly what one token's routed top-k
// does -- must never evict any of them before all are read (see expertBufferPool's doc comment
// on why moeMLP/gemma4MoEFFN's "touch every idx member, then compute" ordering depends on this).
func TestExpertBufferPool_topKNeverSelfEvicts(t *testing.T) {
	const topK = 4
	path, mapping, canonical := int4PoolFixture(t, topK)
	members, _, keys := buildPoolMembers(t, canonical, mapping)

	// A budget of exactly ONE expert's worth would normally clamp to 1 slot -- minSlots=topK
	// must override that up to topK.
	pool, err := newExpertBufferPool(path, members, int64(members[0].fields[0].n), topK)
	if err != nil {
		t.Fatalf("newExpertBufferPool: %v", err)
	}
	t.Cleanup(func() { _ = pool.close() })
	if got := len(pool.slotBuf); got != topK {
		t.Fatalf("slot count = %d, want %d (minSlots clamp)", got, topK)
	}

	for _, k := range keys {
		if err := pool.ensure(k); err != nil {
			t.Fatalf("ensure: %v", err)
		}
	}
	if hits, misses, evictions := pool.stats(); evictions != 0 {
		t.Fatalf("stats: hits=%d misses=%d evictions=%d, want evictions=0 (topK distinct experts must all fit)", hits, misses, evictions)
	}
}

// TestExpertBufferPool_evictsLeastRecent is the LRU accounting gate beyond the topK floor: a
// 5th distinct expert touched against a topK=4-sized pool must evict the least-recently-used
// of the first four, and stats() must reflect exactly that.
func TestExpertBufferPool_evictsLeastRecent(t *testing.T) {
	path, mapping, canonical := int4PoolFixture(t, 5)
	members, _, keys := buildPoolMembers(t, canonical, mapping)

	pool, err := newExpertBufferPool(path, members, int64(members[0].fields[0].n)*4, 4)
	if err != nil {
		t.Fatalf("newExpertBufferPool: %v", err)
	}
	t.Cleanup(func() { _ = pool.close() })

	for _, k := range keys[:4] {
		if err := pool.ensure(k); err != nil {
			t.Fatalf("ensure: %v", err)
		}
	}
	if _, ok := pool.where[keys[0]]; !ok {
		t.Fatal("expert 0 should still be resident before the 5th touch")
	}
	if err := pool.ensure(keys[4]); err != nil {
		t.Fatalf("ensure(4): %v", err)
	}
	if _, ok := pool.where[keys[0]]; ok {
		t.Fatal("expert 0 (least-recently-used) should have been evicted by the 5th distinct touch")
	}
	if hits, misses, evictions := pool.stats(); hits != 0 || misses != 5 || evictions != 1 {
		t.Fatalf("stats: hits=%d misses=%d evictions=%d, want hits=0 misses=5 evictions=1", hits, misses, evictions)
	}
	// Re-touching expert 0 now must be a genuine miss that correctly re-fetches it (not stale
	// data from whatever slot it lands in).
	if err := pool.ensure(keys[0]); err != nil {
		t.Fatalf("ensure(0) again: %v", err)
	}
	wantOut := matmulOut(&canonical[0], make([]float32, canonical[0].Cols()))
	live0 := (*linalg.WeightMat)(keys[0])
	gotOut := matmulOut(live0, make([]float32, canonical[0].Cols()))
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("re-fetched expert 0 output row %d: got %v want %v", i, gotOut[i], wantOut[i])
		}
	}
}

// TestExpertPager_poolModeLockPreventsCrossStreamCorruption is the concurrency-hazard
// regression this design exists to close (see expertBufferPool's doc comment and the Lever 1a
// research this session did before building Lever 1b): with owned buffers, a slot's bytes ARE
// mutable storage a competing miss can overwrite mid-read. Many goroutines (simulated
// concurrent decode streams) repeatedly Lock, touch a RANDOM key, verify the resulting
// WeightMat's matmul output matches THAT key's known-correct output, then Unlock -- over a
// pool deliberately smaller than the key count, so eviction (and therefore slot reuse) is
// guaranteed to happen constantly during the run. Any window where Lock/Unlock failed to
// protect a read would surface here as a wrong-expert (not-necessarily-a-Go-race) output
// mismatch, which -race alone cannot catch -- this asserts correctness directly, not just the
// absence of an unsynchronized access.
func TestExpertPager_poolModeLockPreventsCrossStreamCorruption(t *testing.T) {
	const nExperts = 6
	const nSlots = 2 // deliberately << nExperts: forces constant eviction under concurrency
	path, mapping, canonical := int4PoolFixture(t, nExperts)
	members, _, keys := buildPoolMembers(t, canonical, mapping)

	pool, err := newExpertBufferPool(path, members, int64(members[0].fields[0].n)*nSlots, nSlots)
	if err != nil {
		t.Fatalf("newExpertBufferPool: %v", err)
	}
	t.Cleanup(func() { _ = pool.close() })
	pager := &expertPager{pool: pool, nExperts: nExperts}

	x := make([]float32, canonical[0].Cols())
	wantOuts := make([][]float32, nExperts)
	for i := range wantOuts {
		wantOuts[i] = matmulOut(&canonical[i], x)
	}

	const goroutines = 8
	const itersEach = 200
	var wg sync.WaitGroup
	errs := make(chan string, goroutines*itersEach)
	for g := 0; g < goroutines; g++ {
		wg.Go(func() {
			rng := rand.New(rand.NewSource(int64(g) + 100))
			for iter := 0; iter < itersEach; iter++ {
				i := rng.Intn(nExperts)
				pager.Lock()
				pager.touch(keys[i])
				live := (*linalg.WeightMat)(keys[i])
				got := matmulOut(live, x)
				pager.Unlock()
				for j := range got {
					if got[j] != wantOuts[i][j] {
						errs <- ""
						return
					}
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	if n := len(errs); n > 0 {
		t.Fatalf("%d/%d reads observed wrong-expert data under concurrent Lock/touch/read (cross-stream slot-reuse race)", n, goroutines*itersEach)
	}
	if _, _, evictions := pool.stats(); evictions == 0 {
		t.Fatal("no evictions occurred -- test didn't actually exercise slot reuse under contention")
	}
}
