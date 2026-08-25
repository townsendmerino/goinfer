package decoder

import (
	"os"
	"testing"
)

// TestRow4GiwKind_gemma4_identicalToCanonical is the kind-4 acceptance gate's real-scale
// correctness proof: the gemma4-26b-int4-row4.giw built by re-prequanting the existing
// kind-3 bundle (docs/task-w4a8-neon-bandwidth.md's ".giw kind 4" section) must decode
// byte-identical to the kind-3 original it was built from. Bit-identical dispatch was
// already proven at 0.5B fixture scale (w4a8_row4_giwkind_test.go); this is the same
// proof on the real 26B-A4B checkpoint. Both .giw paths default to the sidecar-naming
// convention used by the acceptance run and can be overridden via env for other boxes.
func TestRow4GiwKind_gemma4_identicalToCanonical(t *testing.T) {
	requireHeavyModel(t)
	kind3 := envOr("GOINFER_GEMMA4_26B_GIW", "~/models/gemma4-26b-int4.giw")
	kind4 := envOr("GOINFER_GEMMA4_26B_GIW_ROW4", "~/models/gemma4-26b-int4-row4.giw")
	kind3, kind4 = expandHome(t, kind3), expandHome(t, kind4)
	if _, err := os.Stat(kind3); err != nil {
		t.Skipf("no kind-3 gemma4 .giw at %s: %v", kind3, err)
	}
	if _, err := os.Stat(kind4); err != nil {
		t.Skipf("no kind-4 gemma4 .giw at %s: %v", kind4, err)
	}

	m3, err := Load(kind3, Options{})
	if err != nil {
		t.Fatalf("load kind-3 %s: %v", kind3, err)
	}
	defer m3.Close()
	m4, err := Load(kind4, Options{})
	if err != nil {
		t.Fatalf("load kind-4 %s: %v", kind4, err)
	}
	defer m4.Close()

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, m3, prompt, 24)
	b := greedyN(t, m4, prompt, 24)
	if len(a) == 0 {
		t.Fatal("no tokens generated")
	}
	if !slicesEqualInt(a, b) {
		t.Fatalf("kind-4 upgrade changed the decode:\n kind3: %v\n kind4: %v", a, b)
	}
	t.Logf("gemma4-26b kind-3 vs kind-4: byte-identical over %d tokens", len(a))
}

// TestRow4GiwKind_gemma4_pagedEviction is TestExpertPaging_bitExact's shape, run
// specifically against the kind-4 gemma4 bundle: full-resident vs a small forced-eviction
// budget must decode identically, and the pager must actually evict (not just cold-fault).
func TestRow4GiwKind_gemma4_pagedEviction(t *testing.T) {
	requireHeavyModel(t)
	kind4 := expandHome(t, envOr("GOINFER_GEMMA4_26B_GIW_ROW4", "~/models/gemma4-26b-int4-row4.giw"))
	if _, err := os.Stat(kind4); err != nil {
		t.Skipf("no kind-4 gemma4 .giw at %s: %v", kind4, err)
	}

	full, err := Load(kind4, Options{})
	if err != nil {
		t.Fatalf("load full: %v", err)
	}
	defer full.Close()

	const budget = 1 << 30 // 1 GB, a small fraction of ~11.4 GB of experts -- forces real eviction
	paged, err := Load(kind4, Options{StreamWeights: true, WeightCacheBytes: budget})
	if err != nil {
		t.Fatalf("load paged: %v", err)
	}
	defer paged.Close()
	if paged.pager == nil {
		t.Fatal("paged load built no pager")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, full, prompt, 24)
	b := greedyN(t, paged, prompt, 24)
	if !slicesEqualInt(a, b) {
		t.Fatalf("paging the kind-4 bundle changed the decode:\n full:  %v\n paged: %v", a, b)
	}
	hits, misses, evictions := paged.pager.stats()
	if evictions == 0 {
		t.Fatalf("pager evicted nothing (hits=%d misses=%d) -- budget too large to exercise eviction", hits, misses)
	}
	assertAdvisedBytesSane(t, paged.pager, misses)
	t.Logf("gemma4-26b kind-4 paged decode byte-identical over %d tokens at %d MB budget; hits=%d misses=%d evictions=%d",
		len(a), budget>>20, hits, misses, evictions)
}

// TestRow4GiwKind_qwen35_pagedEviction is the same eviction-exercising proof for the
// real 35B-A3B kind-4 bundle -- the campaign's other acceptance target.
func TestRow4GiwKind_qwen35_pagedEviction(t *testing.T) {
	requireHeavyModel(t)
	kind4 := expandHome(t, envOr("GOINFER_QWEN35_GIW_ROW4", "~/models/qwen3.5-35b-a3b-int4-row4.giw"))
	if _, err := os.Stat(kind4); err != nil {
		t.Skipf("no kind-4 qwen3.5-35b .giw at %s: %v", kind4, err)
	}

	full, err := Load(kind4, Options{})
	if err != nil {
		t.Fatalf("load full: %v", err)
	}
	defer full.Close()

	// Post-fix, a kind-4 bundle registers only ONE span per expert (row4 when usable,
	// canonical otherwise) -- the pageable total is back to matching kind-3's ~15.6 GB,
	// not the pre-fix doubled ~31.2 GB. 3 GB (~19% residency) forces real eviction over
	// a 40-token run without landing in the near-zero-residency, all-miss regime.
	const budget = 3 << 30
	paged, err := Load(kind4, Options{StreamWeights: true, WeightCacheBytes: budget})
	if err != nil {
		t.Fatalf("load paged: %v", err)
	}
	defer paged.Close()
	if paged.pager == nil {
		t.Fatal("paged load built no pager")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, full, prompt, 40)
	b := greedyN(t, paged, prompt, 40)
	if !slicesEqualInt(a, b) {
		t.Fatalf("paging the kind-4 bundle changed the decode:\n full:  %v\n paged: %v", a, b)
	}
	hits, misses, evictions := paged.pager.stats()
	if evictions == 0 {
		t.Fatalf("pager evicted nothing (hits=%d misses=%d) -- budget too large to exercise eviction", hits, misses)
	}
	assertAdvisedBytesSane(t, paged.pager, misses)
	t.Logf("qwen3.5-35b-a3b kind-4 paged decode byte-identical over %d tokens at %d MB budget; hits=%d misses=%d evictions=%d",
		len(a), budget>>20, hits, misses, evictions)
}

// assertAdvisedBytesSane is the durable, contamination-proof I/O-waste check the
// campaign's original acceptance run had no way to make (docs/task-zeno-compare.md's
// "At-scale acceptance run"/"Pass 1"): bytes actually passed to WILLNEED per miss
// should track the pager's own average per-expert size, not exceed it by a wide
// margin -- a member registering redundant spans (the exact kind-4 double-WILLNEED
// bug this exists to catch) shows up here directly as roughly double, regardless of
// what else the machine's disk is doing (aikit's SpanCache.AdvisedBytes counts what
// THIS cache requested, not physical reads shared with every other process).
func assertAdvisedBytesSane(t *testing.T, p *expertPager, misses int64) {
	t.Helper()
	if misses == 0 {
		t.Fatal("no misses recorded -- cannot check bytes-per-miss")
	}
	advised := p.advisedBytes()
	avgExpert := p.total / int64(p.nExperts)
	perMiss := advised / misses
	// Generous ceiling (1.5x the average expert size): eviction thrash and uneven
	// expert sizes both add real variance above the mean, but a genuine double-
	// registration bug inflates this by ~2x, well outside this margin.
	if ceiling := avgExpert * 3 / 2; perMiss > ceiling {
		t.Fatalf("advised bytes/miss = %d, want <= %d (1.5x the %d-byte average expert size) -- "+
			"looks like a member is registering redundant spans again", perMiss, ceiling, avgExpert)
	}
	t.Logf("advised bytes: %.2f GB over %d misses (%.1f MB/miss avg expert %.1f MB)",
		float64(advised)/1e9, misses, float64(perMiss)/1e6, float64(avgExpert)/1e6)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func expandHome(t *testing.T, p string) string {
	t.Helper()
	if len(p) >= 2 && p[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Fatalf("home dir: %v", err)
		}
		return home + p[1:]
	}
	return p
}
