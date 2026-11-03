//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEPerLayerHitRate is Lead 4's own "first step" (docs/tasks/task-freetoken-techniques.md):
// "instrument per-layer hit rate at a fixed total budget... and see whether the hit-rate
// distribution across layers is actually uneven before building anything." A global pool only
// pays if layers differ in hotness; if every layer sees roughly the same demand, a fixed per-layer
// depth already gives each layer its fair share and pooling buys nothing.
//
// Real 26B, C′ on at whatever slot depth ships today (GOINFER_MOE_CACHE_SLOTS unset → auto-capped
// to free VRAM, same as production) — "a fixed total budget" per the lead's own wording, not a
// number chosen to make a point. Synthetic embeddings, same reasoning as
// TestMoEStreamingDecodeProfile: a hit-rate measurement needs real routing, not real tokens, and
// the model's own trained router is what decides which experts, from any in-vocab embedding.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestMoEPerLayerHitRate -v -timeout 20m
func TestMoEPerLayerHitRate(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 26B decode")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the 26B with C′ on")
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("C′ expert staging is OFF")
	}
	t.Logf("loaded 26B .giw in %s, cacheSlots=%d (per layer, auto-capped to free VRAM)", time.Since(t0).Round(time.Second), r.cacheSlots)

	// Long enough that the LRU has settled and per-layer rates aren't dominated by cold start
	// (every layer's first ~cacheSlots distinct experts are misses by construction).
	const nNew = 200
	var s uint32 = 24601
	next := func() int {
		s = s*1664525 + 1013904223
		return int(s>>8) % 100000
	}
	pos := 0
	id := next()
	genStart := time.Now()
	for i := range nNew {
		emb := m.EmbedResidentForTest(id)
		nextID, err := r.ForwardArgmax(emb, pos)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		id = nextID
		pos++
	}
	genDur := time.Since(genStart)
	t.Logf("generated %d tokens in %s (%.2f tok/s)", nNew, genDur.Round(time.Millisecond), float64(nNew)/genDur.Seconds())

	hits, misses := r.PerLayerCacheStatsForTest()
	type row struct {
		layer      int
		hits, miss uint64
		rate       float64
	}
	var moeRows []row
	for l := range hits {
		if hits[l]+misses[l] == 0 {
			continue // dense layer, no expert cache
		}
		moeRows = append(moeRows, row{l, hits[l], misses[l], float64(hits[l]) / float64(hits[l]+misses[l])})
	}
	if len(moeRows) == 0 {
		t.Fatal("no MoE layer reported any cache traffic — instrument or routing is broken")
	}
	sort.Slice(moeRows, func(i, j int) bool { return moeRows[i].rate < moeRows[j].rate })

	var sumRate, sumHits, sumMiss float64
	for _, rw := range moeRows {
		sumRate += rw.rate
		sumHits += float64(rw.hits)
		sumMiss += float64(rw.miss)
	}
	mean := sumRate / float64(len(moeRows))
	var variance float64
	for _, rw := range moeRows {
		d := rw.rate - mean
		variance += d * d
	}
	stdev := 0.0
	if len(moeRows) > 1 {
		stdev = math.Sqrt(variance / float64(len(moeRows)-1))
	}
	overall := sumHits / (sumHits + sumMiss)

	t.Logf("per-layer hit rate over %d MoE layers, %d slots/layer:", len(moeRows), r.cacheSlots)
	for _, rw := range moeRows {
		t.Logf("  layer %2d: %6.2f%% (%d hits / %d misses)", rw.layer, rw.rate*100, rw.hits, rw.miss)
	}
	t.Logf("overall (pooled) hit rate: %.2f%% | per-layer mean %.2f%% stdev %.2f pp | spread [min %.2f%%, max %.2f%%]",
		overall*100, mean*100, stdev*100, moeRows[0].rate*100, moeRows[len(moeRows)-1].rate*100)

	// Spread alone (min vs max) overstates unevenness driven by one or two outlier layers rather
	// than a real gradient across all of them — report it, but the bounded-upside estimate in
	// docs/measurements/lead4-perlayer-hitrate-2026-09-22.md (what pooling could actually save, in
	// misses) is the number the decision is made on, not this ratio.
	spread := (moeRows[len(moeRows)-1].rate - moeRows[0].rate) * 100
	t.Logf("spread (min to max) %.2f pp — see the measurement record for the bounded-upside estimate this alone does not give.", spread)
}

// TestKV11TradeCurve is KV11's KV-share trade curve on CUDA (docs/measurements/kv11-trade-curve-2026-11/README.md, registered before
// the run): the real 26B with C′ at each ResidentContext, in a mirrored order, records the context the build resolved, the expert
// slots per layer, the pooled hit rate and the decode rate over 200 tokens, with each cell's start load, available memory and free
// VRAM. A timed run: hold the exclusive timing lock.
func TestKV11TradeCurve(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: real 26B loads")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	readLoad := func() string {
		b, _ := os.ReadFile("/proc/loadavg")
		f := strings.Fields(string(b))
		if len(f) == 0 {
			return "?"
		}
		return f[0]
	}
	readAvail := func() float64 {
		b, _ := os.ReadFile("/proc/meminfo")
		for _, l := range strings.Split(string(b), "\n") {
			if f := strings.Fields(l); len(f) >= 2 && f[0] == "MemAvailable:" {
				kb, _ := strconv.ParseFloat(f[1], 64)
				return kb / (1 << 20)
			}
		}
		return -1
	}
	for i, ctx := range []int{4096, 8192, 16384, 32768, 32768, 16384, 8192, 4096} {
		free, _ := decoder.FreeBytesFor("cuda")
		start := fmt.Sprintf("load %s, MemAvailable %.1f GB, free VRAM %d MiB", readLoad(), readAvail(), free>>20)
		t0 := time.Now()
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", ResidentContext: ctx})
		if err != nil {
			fmt.Fprintf(os.Stderr, "KV11 cell %d ctx %d: %s; load error: %v\n", i+1, ctx, start, err)
			continue
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || !r.cacheExperts {
			fmt.Fprintf(os.Stderr, "KV11 cell %d ctx %d: %s; not resident with C′: %s\n", i+1, ctx, start, m.ResidentDecline())
			m.Close()
			continue
		}
		load := time.Since(t0)
		var s uint32 = 24601
		next := func() int { s = s*1664525 + 1013904223; return int(s>>8) % 100000 }
		id := next()
		const nNew = 200
		g0 := time.Now()
		for pos := range nNew {
			if id, err = r.ForwardArgmax(m.EmbedResidentForTest(id), pos); err != nil {
				t.Fatalf("cell %d ctx %d step %d: %v", i+1, ctx, pos, err)
			}
		}
		rate := float64(nNew) / time.Since(g0).Seconds()
		hits, misses := r.PerLayerCacheStatsForTest()
		var h, ms uint64
		for l := range hits {
			h, ms = h+hits[l], ms+misses[l]
		}
		hit := 0.0
		if h+ms > 0 {
			hit = 100 * float64(h) / float64(h+ms)
		}
		fmt.Fprintf(os.Stderr, "KV11 cell %d ctx %d: %s; resolved ctx %d, %d expert slots a layer, hit rate %.2f%%, decode %.2f tok/s, load %.0f s\n",
			i+1, ctx, start, r.ctxCap, r.cacheSlots, hit, rate, load.Seconds())
		m.Close()
	}
}
