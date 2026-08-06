package decoder

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// TestPagers_concurrent is the C-30 gate: the layer and expert pagers live on *Model and StreamWeights
// supports concurrent decode streams, so their shared paging state must be race-free. Run with -race;
// before the mutex, concurrent enterLayer/touch fire the detector on state[]/counters and the LRU list.
func TestLayerPager_concurrentNoRace(t *testing.T) {
	const n = 48
	spans := make([][][]byte, n)
	for i := range spans {
		spans[i] = [][]byte{make([]byte, 4096)} // non-empty so state[] actually toggles (Advise no-ops on heap mem)
	}
	p := &layerPager{spans: spans, window: 4, ahead: 1, state: make([]bool, n)}
	var wg sync.WaitGroup
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iter := 0; iter < 3; iter++ {
				for l := 0; l < n; l++ {
					p.enterLayer(l)
				}
				p.finishLayers()
				_, _ = p.stats()
			}
		}()
	}
	wg.Wait()
}

func TestExpertPager_concurrentNoRace(t *testing.T) {
	const nExp = 16
	cache := mmap.NewSpanCacheWithPolicy[unsafe.Pointer](4*4096, mmap.EvictLeastRecent) // holds ~4 → forces eviction
	keys := make([]unsafe.Pointer, nExp)
	for i := range keys {
		keys[i] = unsafe.Pointer(new([8]byte))
		cache.Add(keys[i], [][]byte{make([]byte, 4096)})
	}
	p := &expertPager{cache: cache, nExperts: nExp, total: nExp * 4096}
	var wg sync.WaitGroup
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func(off int) {
			defer wg.Done()
			for iter := 0; iter < 50; iter++ {
				p.touch(keys[(iter+off)%nExp])
				_, _, _ = p.stats()
			}
		}(g)
	}
	wg.Wait()
}
