package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestLayerPaging_bitExact is the end-to-end gate for idea #4 (dense weight
// streaming): paging a dense model's layers out of the read-only mapping under a
// small window must not change a single output token. It prequantizes the dense
// fixture to a .giw, loads it fully resident and with a 1-byte budget (clamped to
// the minimum window, so the layer loop evicts and re-faults most layers every
// token), and asserts the greedy decodes are byte-identical and that streaming
// actually ran (prefetches and evictions > 0). The re-fault safety it relies on is
// proven model-free by TestMadvise_dontneedRefaultsIntact.
func TestLayerPaging_bitExact(t *testing.T) {
	path := prequantGGUF(t) // a dense .gguf (qwen2.5-coder-0.5b by default)
	m0, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load gguf: %v", err)
	}
	blob, err := SerializeWeights(m0.w, "layer-paging")
	m0.Close()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	giwPath := filepath.Join(t.TempDir(), "dense.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}

	full, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load full: %v", err)
	}
	defer full.Close()

	// WeightCacheBytes 1 clamps the window to its floor, so nearly every layer is
	// evicted behind and re-faulted ahead each token — the streaming path under test.
	paged, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: 1})
	if err != nil {
		t.Fatalf("load paged: %v", err)
	}
	defer paged.Close()
	if paged.layerPager == nil {
		t.Fatal("paged load built no layer pager (expected an mmap-backed dense model)")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, full, prompt, 16)
	b := greedyN(t, paged, prompt, 16)
	if len(a) == 0 {
		t.Fatal("no tokens generated")
	}
	if !slicesEqualInt(a, b) {
		t.Fatalf("dense streaming changed the decode:\n full:  %v\n paged: %v", a, b)
	}
	pf, ev := paged.layerPager.stats()
	if pf == 0 || ev == 0 {
		t.Fatalf("streaming did not run (prefetches=%d evictions=%d)", pf, ev)
	}
	t.Logf("dense streaming byte-identical over %d tokens at window=%d; prefetches=%d evictions=%d",
		len(a), paged.layerPager.window, pf, ev)
}

// TestLoad_resolvesAutoWeightCacheBudgetFromLiveProbe is S4 item 2's WIRING gate
// (task-never-swap-2026-09.md): TestResolveWeightCacheBudget (fitguard_test.go) proves the
// arithmetic in isolation; this proves decoder.Load actually calls it before newLayerPager sees
// WeightCacheBytes, through a REAL .giw load. Injecting hostRAMAvailable to two different live
// figures must move newLayerPager's own window arithmetic in the direction each figure implies: a
// modest available produces a budget smaller than this fixture's total resident weight bytes
// (window pinned well under its layer count, so a REAL pager is built) while a generous available
// makes the whole model "fit" (window >= n, so newLayerPager returns nil — no streaming needed).
// Reverting the model.go wiring (the `opts.WeightCacheBytes = resolveWeightCacheBudget(...)` line)
// makes BOTH cases build off aikit's own AutoBudget() instead, indistinguishable from each other
// since the live probe would never be consulted.
//
// Needs the 1.5B fixture, not the registry's default 0.5B one: at 0.5B (0.59 GB resident), every
// available figure large enough to clear guardGIWFit's own ~1 GB KV+scratch floor (S4 item 1)
// already halves to a budget bigger than the whole model — there is no available figure where
// BOTH guards are exercised on that fixture. 1.5B (1.66 GB resident) leaves a real gap between the
// two floors (empirically probed: 2.5 GB available streams at window=27 of 28 layers; 3 GB+ fits
// outright).
func TestLoad_resolvesAutoWeightCacheBudgetFromLiveProbe(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	t.Setenv("GOINFER_PREQUANT_GGUF", filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	path := prequantGGUF(t)
	m0, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load gguf: %v", err)
	}
	blob, err := SerializeWeights(m0.w, "layer-paging")
	m0.Close()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	giwPath := filepath.Join(t.TempDir(), "dense.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}

	t.Run("modest live probe forces real streaming", func(t *testing.T) {
		defer injectHostRAMAvailable(t, int64(2.5*(1<<30)))() // 2.5 GB available -> 1.25 GB budget
		paged, err := Load(giwPath, Options{StreamWeights: true})
		if err != nil {
			t.Fatalf("load paged: %v", err)
		}
		defer paged.Close()
		if paged.layerPager == nil {
			t.Fatal("no layer pager built with a modest live-available figure — " +
				"resolveWeightCacheBudget is not wired into Load")
		}
		t.Logf("window=%d", paged.layerPager.window)
	})
	t.Run("generous live probe means the model fits, no pager needed", func(t *testing.T) {
		defer injectHostRAMAvailable(t, 10<<30)() // 10 GB available -> 5 GB budget, well past 1.66 GB total
		paged, err := Load(giwPath, Options{StreamWeights: true})
		if err != nil {
			t.Fatalf("load paged: %v", err)
		}
		defer paged.Close()
		if paged.layerPager != nil {
			t.Fatal("layer pager built even though the live-available figure implies the " +
				"whole model fits comfortably")
		}
	})
}

// TestLayerPager_residentBytesWithinBudget is the P-12 (secondary claim) gate:
// enterLayer prefetches layer l+ahead before releasing layer l-window, so the
// resident set at steady state is `window+ahead` layers, not `window` — the
// finishLayers doc comment's own claim ("stays bounded by `window`") is false
// against that mechanism. Drives enterLayer directly (bypassing a real decode)
// across every layer and asserts the peak resident byte total — summed from the
// pager's own spans, the same bytes newLayerPager derived `window` from — never
// exceeds the WeightCacheBytes budget that was requested.
func TestLayerPager_residentBytesWithinBudget(t *testing.T) {
	path := prequantGGUF(t)
	m0, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load gguf: %v", err)
	}
	blob, err := SerializeWeights(m0.w, "layer-paging")
	m0.Close()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	giwPath := filepath.Join(t.TempDir(), "dense.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}

	// A budget comfortably above the ahead+2 floor and comfortably below "whole
	// model fits" (newLayerPager returns nil once window >= n), so the fix's
	// effect on window shows up distinctly from either edge case.
	const budget = 128 * 1024 * 1024
	paged, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: budget})
	if err != nil {
		t.Fatalf("load paged: %v", err)
	}
	defer paged.Close()
	p := paged.layerPager
	if p == nil {
		t.Fatal("paged load built no layer pager")
	}
	layerBytes := make([]int64, len(p.spans))
	var maxLayer int64
	for l, spans := range p.spans {
		for _, s := range spans {
			layerBytes[l] += int64(len(s))
		}
		if layerBytes[l] > maxLayer {
			maxLayer = layerBytes[l]
		}
	}
	t.Logf("budget=%d maxLayer=%d window=%d ahead=%d", budget, maxLayer, p.window, p.ahead)
	if p.window <= p.ahead+2 {
		t.Skipf("budget=%d (maxLayer=%d) clamped window=%d to the correctness floor (ahead+2=%d) — pick a larger budget to exercise the budget-derived case", budget, maxLayer, p.window, p.ahead+2)
	}
	residentBytes := func() int64 {
		var b int64
		for l, resident := range p.state {
			if resident {
				b += layerBytes[l]
			}
		}
		return b
	}

	var peak int64
	for l := range p.spans {
		p.enterLayer(l)
		if b := residentBytes(); b > peak {
			peak = b
		}
	}
	t.Logf("budget=%d window=%d ahead=%d peak resident=%d", budget, p.window, p.ahead, peak)
	if peak > budget {
		t.Errorf("peak resident bytes %d exceeds requested budget %d (window=%d, ahead=%d — resident set peaks at window+ahead layers)", peak, budget, p.window, p.ahead)
	}
}
