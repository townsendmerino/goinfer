//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestP20ExpertLocality is the "measure the split before building" step docs/queue-performance.md's
// P20 entry asks for, before any expert-major kernel is written: expert-major batching only pays off
// if a whole M-row prefill chunk touches FEW ENOUGH distinct experts per layer to stage them all on
// the device at once — if it touches MORE than the box can hold, "fetch each distinct expert once per
// chunk" is not achievable at that chunk width regardless of kernel design, and the item needs a
// smaller batch width (or is dead on this box), not a kernel.
//
// Uses routeRecord (cuda/resident.go), a test-only hook set on the already-loaded resident: for every
// loadRoutedExperts call during ONE prefill, records (layer, routed expert ids) and, per layer,
// accumulates the SET of distinct ids seen across the WHOLE chunk. Reports, per layer: distinct
// experts touched by the chunk, vs topK*M (the row-at-a-time total, what the current sequential path
// fetches with full row-to-row duplication), vs the C′ cache's own slot count (what the box can
// actually hold resident at once).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_P20_MODEL=gemma4-26b-int4.giw GOINFER_P20_M=512 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestP20ExpertLocality -v -timeout 20m
func TestP20ExpertLocality(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 26B model)")
	}
	name := os.Getenv("GOINFER_P20_MODEL")
	if name == "" {
		name = "gemma4-26b-int4.giw"
	}
	path := modelPath(name)
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	M := 512
	if v := os.Getenv("GOINFER_P20_M"); v != "" {
		fmt.Sscanf(v, "%d", &M)
	}
	t0 := time.Now()
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", ResidentContext: 8192, MoECacheExperts: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	t.Logf("loaded %s in %s", name, time.Since(t0).Round(time.Second))
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if !r.cacheExperts {
		t.Fatal("C′ expert staging is OFF — this test measures the constrained-VRAM regime specifically")
	}
	nLayers := len(r.layers)
	seen := make([]map[uint32]bool, nLayers)
	rows := make([]int, nLayers)
	for l := range seen {
		seen[l] = map[uint32]bool{}
	}
	r.routeRecord = func(layer int, ids []uint32) {
		for _, id := range ids {
			seen[layer][id] = true
		}
		rows[layer]++
	}
	defer func() { r.routeRecord = nil }()

	_, _, _, _, _, _, vocab := mc.Dims()
	embs := make([][]float32, M)
	var s uint32 = 999331
	for i := range embs {
		s = s*1664525 + 1013904223
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
	}
	if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}

	fmt.Printf("=== P20 EXPERT LOCALITY: %s, M=%d, cacheSlots=%d, nE=%d, topK=%d ===\n", name, M, r.cacheSlots, r.nE, r.topK)
	fmt.Printf("%-6s %8s %8s %10s %10s %12s\n", "layer", "rows", "distinct", "row*topK", "dupFactor", "fitsInSlots?")
	var maxDistinct, moeLayers int
	var sumRatio float64
	for l := 0; l < nLayers; l++ {
		if rows[l] == 0 {
			continue // dense/non-MoE layer (gemma4's dense‖MoE branch skips routing here)
		}
		moeLayers++
		d := len(seen[l])
		total := rows[l] * r.topK
		ratio := float64(total) / float64(max(d, 1))
		sumRatio += ratio
		if d > maxDistinct {
			maxDistinct = d
		}
		fits := "NO"
		if d <= r.cacheSlots {
			fits = "yes"
		}
		fmt.Printf("%-6d %8d %8d %10d %10.2f %12s\n", l, rows[l], d, total, ratio, fits)
	}
	fmt.Printf("MoE layers=%d, worst-case distinct/layer=%d (cacheSlots=%d), mean row-to-distinct dup factor=%.2f\n",
		moeLayers, maxDistinct, r.cacheSlots, sumRatio/float64(max(moeLayers, 1)))
	if hits, misses := r.CacheStatsForTest(); hits+misses > 0 {
		fmt.Printf("EXISTING per-row LRU cache, this same prefill: %d hits / %d misses (%.1f%% hit rate) — the headroom expert-major would have to beat\n",
			hits, misses, 100*float64(hits)/float64(hits+misses))
	}

	// A histogram of distinct counts, so "does it fit" isn't read off one worst layer.
	var counts []int
	for l := 0; l < nLayers; l++ {
		if rows[l] > 0 {
			counts = append(counts, len(seen[l]))
		}
	}
	sort.Ints(counts)
	if len(counts) > 0 {
		fmt.Printf("distinct-per-layer distribution: min=%d p25=%d median=%d p75=%d max=%d\n",
			counts[0], counts[len(counts)/4], counts[len(counts)/2], counts[3*len(counts)/4], counts[len(counts)-1])
	}
}
