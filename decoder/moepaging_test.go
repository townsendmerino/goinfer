package decoder

import (
	"context"
	"os"
	"testing"
)

// The pager's generic span-residency cache (eviction over budget, WILLNEED/DONTNEED,
// stats) now lives in aikit/mmap.SpanCache and is gated there; the expert pager runs
// it with the frequency-aware policy (TestSpanCache_evictsLeastRecentWithPolicy),
// distinct from the ANN paths' scan-resistant default. The page-granular re-fault
// safety the whole pager rests on is gated by aikit's TestMadvise_dontneedRefaultsIntact.
// This file keeps only the goinfer-specific end-to-end gate.

// TestExpertPaging_bitExact is the end-to-end correctness gate for idea #2: paging
// experts out of the read-only mapping must not change a single output token. It
// loads a real large-expert MoE .giw (GOINFER_MOE_GIW) twice — fully resident and
// with a small expert-cache budget (forcing the LRU to evict and the decode to
// re-fault evicted experts) — and asserts the greedy decodes are byte-identical and
// that eviction actually happened (evictions > 0, not just cold-start misses).
// Skipped without the asset (no small MoE .giw round-trips on this box — the tiny
// hybrid fixture's DeltaNet weights aren't .giw-serializable and its experts are
// sub-page anyway).
func TestExpertPaging_bitExact(t *testing.T) {
	giwPath := os.Getenv("GOINFER_MOE_GIW")
	if giwPath == "" {
		t.Skip("set GOINFER_MOE_GIW=<large-expert MoE .giw> for the end-to-end paging gate")
	}

	full, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load full .giw: %v", err)
	}
	defer full.Close()

	// A 512 MB expert cache is a small fraction of a multi-GB MoE, so the LRU evicts
	// across tokens and cold experts re-fault — exercising the path under test —
	// while still holding roughly a token's working set, so the decode stays fast
	// enough to run many tokens. (Correctness doesn't depend on the budget; this is
	// chosen to force eviction at a sane speed.)
	const budget = 512 << 20
	paged, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: budget})
	if err != nil {
		t.Fatalf("load paged .giw: %v", err)
	}
	defer paged.Close()
	if paged.pager == nil {
		t.Fatal("paged load built no pager (expected mmap-backed MoE experts)")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, full, prompt, 24)
	b := greedyN(t, paged, prompt, 24)
	if len(a) == 0 {
		t.Fatal("no tokens generated")
	}
	if !slicesEqualInt(a, b) {
		t.Fatalf("paging changed the decode:\n full:  %v\n paged: %v", a, b)
	}
	hits, misses, evictions := paged.pager.stats()
	if evictions == 0 {
		t.Fatalf("pager evicted nothing (hits=%d misses=%d) — budget too large to exercise eviction", hits, misses)
	}
	t.Logf("paged decode byte-identical over %d tokens at %d MB budget; pager hits=%d misses=%d evictions=%d (evict/re-fault proven lossless on a real %s MoE)",
		len(a), budget>>20, hits, misses, evictions, "35B-class")
}

func greedyN(t *testing.T, m *Model, prompt []int, n int) []int {
	t.Helper()
	out, gen := m.Generate(context.Background(), prompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return got
}

func slicesEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
