package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
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

// greedyN generates n tokens and, per-token, writes a live progress line straight to
// stderr (not t.Logf, which the testing package buffers until the whole test function
// returns — useless for a heavy-model test that can run tens of minutes with zero
// visibility otherwise). One line per token, cheap relative to the multi-second-per-token
// cost these paged/resident real-checkpoint tests already pay.
func greedyN(t *testing.T, m *Model, prompt []int, n int) []int {
	t.Helper()
	start := time.Now()
	out, gen := m.Generate(context.Background(), prompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
		fmt.Fprintf(os.Stderr, "  [%s] token %d/%d (id %d)\n", time.Since(start).Round(time.Second), len(got), n, tok)
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

// newRealMmapPager builds a real, working *expertPager over n keys, each backed by a
// genuinely mmap'd (file-backed) span of spanBytes — bypassing newExpertPager's own
// tensor-introspection (WeightMat.MappedSpan requires quantized storage AND a span that
// survives page-rounding, neither of which a tiny synthetic fixture's few-KB expert
// tensors can satisfy; see M-34's own investigation). SpanCache.Touch issues real
// MADV_WILLNEED/DONTNEED against these spans, so they must be real mmap'd, file-backed
// memory — not a heap slice, which DONTNEED would corrupt.
func newRealMmapPager(t *testing.T, keys []unsafe.Pointer, spanBytes int64) *expertPager {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pager.bin")
	if err := os.WriteFile(path, make([]byte, spanBytes*int64(len(keys))), 0o644); err != nil {
		t.Fatalf("write backing file: %v", err)
	}
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	t.Cleanup(func() { _ = mmap.Unmap(data) })
	// budget = one span: the smallest budget that can ever be resident, maximizing the
	// chance a touch is actually exercised (not just present).
	cache := mmap.NewSpanCacheWithPolicy[unsafe.Pointer](spanBytes, mmap.EvictLeastRecent)
	for i, k := range keys {
		cache.Add(k, [][]byte{data[int64(i)*spanBytes : int64(i+1)*spanBytes]})
	}
	return &expertPager{cache: cache, nExperts: len(keys), total: spanBytes * int64(len(keys))}
}

// TestExpertPaging_touchesEveryFamilyThatBuildsAPager is M-34's gate (audit-2026-09-10): the
// pager and its budget banner are built generically for ANY mmap-backed MoE (newExpertPager
// registers every w.Layers[li].Experts entry regardless of architecture), but gpt-oss, Llama 4
// and Nemotron 3 Nano's own forward loops indexed lw.Experts directly with no pager.touch call
// at all — so -stream-weights enforced no RAM bound for these three families even though the
// banner claimed one. This drives each family's own real MoE forward function directly
// (hand-built minimal Architecture/LayerWeights, same discipline as
// TestNemotron3NanoMoE_forward) with m.pager wired to a real mmap-backed pager keyed by
// exactly the addresses (&lw.Experts[e]) the fixed code now touches, and asserts
// pager.stats() shows nonzero hits+misses afterward — which it did NOT before this fix
// (misses would stay 0 forever, since the mapped experts were never faulted through the
// cache at all).
func TestExpertPaging_touchesEveryFamilyThatBuildsAPager(t *testing.T) {
	const hidden, inter = 2, 2
	newExperts := func() []expertWeights {
		return []expertWeights{
			{
				Gate: linalg.WrapF32([]float32{1, 0, 0, 1}, inter, hidden),
				Up:   linalg.WrapF32([]float32{1, 0, 0, 1}, inter, hidden),
				Down: linalg.WrapF32([]float32{1, 1, 1, 1}, hidden, inter),
			},
			{
				Gate: linalg.WrapF32([]float32{2, 0, 0, 2}, inter, hidden),
				Up:   linalg.WrapF32([]float32{2, 0, 0, 2}, inter, hidden),
				Down: linalg.WrapF32([]float32{1, 0, 0, 1}, hidden, inter),
			},
		}
	}
	keysOf := func(exps []expertWeights) []unsafe.Pointer {
		ks := make([]unsafe.Pointer, len(exps))
		for i := range exps {
			ks[i] = unsafe.Pointer(&exps[i])
		}
		return ks
	}

	t.Run("gpt-oss", func(t *testing.T) {
		experts := newExperts()
		arch := &Architecture{
			HiddenDim: hidden,
			MoE:       &MoEConfig{NumExperts: 2, TopK: 2, IntermediateDim: inter},
			gptoss:    &gptOssParams{SwigluAlpha: 1.702, SwigluLimit: 7.0},
		}
		lw := &LayerWeights{
			Router:  linalg.WrapF32([]float32{2, 0, 0, 1}, 2, hidden),
			Experts: experts,
		}
		pager := newRealMmapPager(t, keysOf(experts), 16384)
		m := &Model{be: &cpuBackend{}, w: &Weights{arch: arch}, pager: pager}
		if _, err := m.gptOssMoE([]float32{1, 1}, lw, arch); err != nil {
			t.Fatalf("gptOssMoE: %v", err)
		}
		hits, misses, _ := pager.stats()
		if hits+misses == 0 {
			t.Fatal("pager.stats() = 0 touches after gptOssMoE — gpt-oss still bypasses the pager (M-34)")
		}
		t.Logf("gpt-oss: hits=%d misses=%d", hits, misses)
	})

	t.Run("llama4", func(t *testing.T) {
		experts := newExperts()
		arch := &Architecture{
			HiddenDim: hidden,
			MoE:       &MoEConfig{NumExperts: 2, TopK: 1, IntermediateDim: inter, SharedIntermediateDim: inter},
		}
		lw := &LayerWeights{
			Router:  linalg.WrapF32([]float32{2, 0, 0, 1}, 2, hidden),
			Experts: experts,
			SharedExpert: expertWeights{
				Gate: linalg.WrapF32([]float32{1, 1, 1, 1}, inter, hidden),
				Up:   linalg.WrapF32([]float32{1, 1, 1, 1}, inter, hidden),
				Down: linalg.WrapF32([]float32{1, 0, 0, 1}, hidden, inter),
			},
		}
		pager := newRealMmapPager(t, keysOf(experts), 16384)
		m := &Model{be: &cpuBackend{}, pager: pager}
		out := make([]float32, hidden)
		m.llama4MoE([]float32{1, 1}, out, lw, arch)
		hits, misses, _ := pager.stats()
		if hits+misses == 0 {
			t.Fatal("pager.stats() = 0 touches after llama4MoE — Llama 4 still bypasses the pager (M-34)")
		}
		t.Logf("llama4: hits=%d misses=%d", hits, misses)
	})

	t.Run("nemotron3nano", func(t *testing.T) {
		experts := newExperts()
		arch := &Architecture{
			HiddenDim: hidden,
			MoE: &MoEConfig{
				NumExperts: 2, TopK: 2, NormTopKProb: true, IntermediateDim: inter,
				RouterSigmoid: true, RoutedScale: 1.0,
			},
		}
		// nemotronExpertFFN reads only Up/Down (no Gate) — matches nemotronMoE's real shape.
		experts[0].Gate, experts[1].Gate = linalg.WeightMat{}, linalg.WeightMat{}
		lw := &LayerWeights{
			Router:     linalg.WrapF32([]float32{2, 0, 0, 1}, 2, hidden),
			RouterBias: []float32{0, 0},
			Experts:    experts,
		}
		pager := newRealMmapPager(t, keysOf(experts), 16384)
		m := &Model{be: &cpuBackend{}, pager: pager}
		_ = m.nemotronMoE([]float32{1, 1}, lw, arch, hidden)
		hits, misses, _ := pager.stats()
		if hits+misses == 0 {
			t.Fatal("pager.stats() = 0 touches after nemotronMoE — Nemotron 3 Nano still bypasses the pager (M-34)")
		}
		t.Logf("nemotron3nano: hits=%d misses=%d", hits, misses)
	})
}
