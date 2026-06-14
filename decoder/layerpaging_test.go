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
