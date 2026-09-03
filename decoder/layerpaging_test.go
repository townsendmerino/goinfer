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
