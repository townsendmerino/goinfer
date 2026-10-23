package decoder

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// gbWriteGIW writes m's weights to path as a weights-only .giw, with islands as f32 arrays when f32Form (the form before kind 9).
func gbWriteGIW(t *testing.T, m *Model, path string, f32Form bool) {
	t.Helper()
	prev := giwIslandsF32ForTest
	giwIslandsF32ForTest = f32Form
	defer func() { giwIslandsF32ForTest = prev }()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	werr := giw.WriteStream(f, nil, func(w io.Writer) (int64, error) {
		return SerializeWeightsToForTarget(w, m.Weights(), "mem11-giw-bf16", GIWTargetNone)
	})
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		t.Fatalf("write %s: %v", path, werr)
	}
}

// gbConvs is m's short-conv mixers, in layer order.
func gbConvs(m *Model) []*shortConvWeights {
	var out []*shortConvWeights
	for l := range m.w.Layers {
		if sc := m.w.Layers[l].shortConv; sc != nil {
			out = append(out, sc)
		}
	}
	return out
}

// gbRoundIslands replaces m's short-conv projections with their bf16-rounded values held as the bf16 kind (B16-G1's fixture),
// and returns the islands' bf16 bytes.
func gbRoundIslands(t *testing.T, m *Model) int64 {
	t.Helper()
	var bytes int64
	for _, sc := range gbConvs(m) {
		for _, w := range []*linalg.WeightMat{&sc.inProj, &sc.outProj} {
			f := slices.Clone(islandF32(w))
			for i, x := range f {
				f[i] = linalg.BF16ToF32(linalg.F32ToBF16(x))
			}
			*w = islandMat(f, w.Rows(), w.Cols())
			if w.Kind() != "bf16" {
				t.Fatalf("a rounded island is %s", w.Kind())
			}
			bytes += 2 * int64(len(f))
		}
	}
	return bytes
}

// gbIslands reports the kind every island of m holds ("" if they differ) and how many of their bytes alias the mapping.
func gbIslands(m *Model) (kind string, aliased int64) {
	for i, sc := range gbConvs(m) {
		for _, w := range []*linalg.WeightMat{&sc.inProj, &sc.outProj} {
			if i == 0 && kind == "" {
				kind = w.Kind()
			} else if w.Kind() != kind {
				kind = "mixed"
			}
			aliased += mmapAliasedBytes(m, w)
		}
	}
	return kind, aliased
}

// TestMEM11_giwBF16Islands is GB-G1, GB-G2's tier 0 and GB-G3 (docs/tasks/task-memory-campaign-2026-10.md, "MEM11-giw-bf16
// gates"): lfm2-tiny's bf16 islands written as kind 9 load aliased from the mapping, bit-identical, off the heap; the committed
// fixture's f32 islands round-trip as f32; a bundle in the f32 form (and, with GOINFER_GB_MAIN_BUNDLE, one written by main's
// build) still loads, with its islands bf16 on the heap.
func TestMEM11_giwBF16Islands(t *testing.T) {
	const dir = "../testdata/lfm2-tiny"
	tmp := t.TempDir()
	load := func(path string) *Model {
		t.Helper()
		m, err := Load(path, Options{})
		if err != nil {
			t.Fatalf("Load(%s): %v", path, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}

	// The committed fixture: its islands are f32 (not bf16-exact) and round-trip as f32 arrays.
	m0 := load(dir)
	want0 := mem11Logits(t, m0)
	p0 := filepath.Join(tmp, "f32.giw")
	gbWriteGIW(t, m0, p0, false)
	g0 := load(p0)
	if k, a := gbIslands(g0); k != "f32" || a != 0 {
		t.Errorf("the committed fixture's islands load as %s with %d aliased bytes, want f32, 0", k, a)
	}
	if d := mem11SameBits(want0, mem11Logits(t, g0)); d != 0 {
		t.Errorf("the committed fixture: %d logits differ after the round trip", d)
	}

	// The rounded islands: kind 9, aliased, bit-identical, off the heap.
	m := load(dir)
	islandBytes := gbRoundIslands(t, m)
	want := mem11Logits(t, m)
	p9, pf := filepath.Join(tmp, "bf16.giw"), filepath.Join(tmp, "bf16-f32form.giw")
	gbWriteGIW(t, m, p9, false)
	gbWriteGIW(t, m, pf, true)
	g9, gf := load(p9), load(pf)
	if k, a := gbIslands(g9); k != "bf16" || a != islandBytes {
		t.Errorf("kind 9: islands load as %s with %d aliased bytes, want bf16, %d", k, a, islandBytes)
	}
	if k, a := gbIslands(gf); k != "bf16" || a != 0 {
		t.Errorf("the f32 form: islands load as %s with %d aliased bytes, want bf16 on the heap, 0", k, a)
	}
	d9, df := mem11SameBits(want, mem11Logits(t, g9)), mem11SameBits(want, mem11Logits(t, gf))
	if d9 != 0 || df != 0 {
		t.Errorf("logits differ from the direct load: kind 9 %d, the f32 form %d", d9, df)
	}
	rw9, rwf := g9.ResidentWeightBytes(), gf.ResidentWeightBytes()
	hc9, hcf := g9.ResidentHostCopyBytes(0), gf.ResidentHostCopyBytes(0)
	if rw9 != rwf || hcf-hc9 != islandBytes {
		t.Errorf("ResidentWeightBytes %d vs %d (want equal); ResidentHostCopyBytes %d vs %d, want lower by %d", rw9, rwf, hc9, hcf, islandBytes)
	}
	fmt.Fprintf(os.Stderr, "[GB-G1/G2/G3] lfm2-tiny: %d conv layers, islands %d B bf16; logits differ: kind 9 %d, f32 form %d, f32 fixture 0 expected; ResidentWeightBytes %d / %d; host copy %d -> %d\n",
		len(gbConvs(g9)), islandBytes, d9, df, rwf, rw9, hcf, hc9)

	if p := os.Getenv("GOINFER_GB_MAIN_BUNDLE"); p != "" {
		gm := load(p)
		if k, a := gbIslands(gm); k != "bf16" || a != 0 {
			t.Errorf("main's bundle: islands load as %s with %d aliased bytes, want bf16 on the heap, 0", k, a)
		}
		d := mem11SameBits(want, mem11Logits(t, gm))
		if d != 0 {
			t.Errorf("main's bundle: %d logits differ", d)
		}
		fmt.Fprintf(os.Stderr, "[GB-G3] main's bundle %s: loads, %d logits differ\n", p, d)
	}
}

// TestMEM11_giwBF16HeapLFM2 is GB-G2's tier 1: LFM2.5-2.6B from ~/models at int4, written as a .giw in each form and loaded
// mapped one at a time, Go heap in use after GC with only that model live, and the two loads' logits. Not timed.
//
//	GOINFER_MEM11_LFM2=1 go test -count=1 -run '^TestMEM11_giwBF16HeapLFM2$' -timeout 20m -v ./decoder/
func TestMEM11_giwBF16HeapLFM2(t *testing.T) {
	if os.Getenv("GOINFER_MEM11_LFM2") == "" {
		t.Skip("set GOINFER_MEM11_LFM2=1 (loads LFM2.5-2.6B from ~/models and writes two .giw files of a few GB)")
	}
	home, _ := os.UserHomeDir()
	tmp := t.TempDir()
	heap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapInuse
	}
	src, err := Load(filepath.Join(home, "models", "lfm2.5-2.6b"), Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var islandBytes int64
	for _, sc := range gbConvs(src) {
		if sc.inProj.Kind() != "bf16" {
			t.Fatalf("the direct load's island is %s, want bf16 (MEM11-bf16)", sc.inProj.Kind())
		}
		islandBytes += wmBytes(&sc.inProj) + wmBytes(&sc.outProj)
	}
	p9, pf := filepath.Join(tmp, "kind9.giw"), filepath.Join(tmp, "f32form.giw")
	gbWriteGIW(t, src, p9, false)
	gbWriteGIW(t, src, pf, true)
	src.Close()
	src = nil
	base := heap()
	read := func(path string) (uint64, [][]float32) {
		m, err := Load(path, Options{})
		if err != nil {
			t.Fatalf("Load(%s): %v", path, err)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		cache := m.NewCache(16)
		var lg [][]float32
		for i := range 8 {
			l, err := m.forward((i*37+11)%vocab, cache)
			if err != nil {
				t.Fatal(err)
			}
			lg = append(lg, append([]float32(nil), l...))
		}
		cache = nil
		h := heap()
		runtime.KeepAlive(m)
		m.Close()
		return h, lg
	}
	hf, lf := read(pf)
	h9, l9 := read(p9)
	d := mem11SameBits(lf, l9)
	fall := int64(hf) - int64(h9)
	fmt.Fprintf(os.Stderr, "[GB-G2] LFM2.5-2.6B int4 .giw: islands %.3f GB bf16; Go heap in use after GC (base %.3f GB): f32 form %.3f GB, kind 9 %.3f GB, fall %.3f GB (%.1f%% of the islands); logits over 8 tokens differ: %d\n",
		float64(islandBytes)/1e9, float64(base)/1e9, float64(hf)/1e9, float64(h9)/1e9, float64(fall)/1e9, 100*float64(fall)/float64(islandBytes), d)
	if d != 0 {
		t.Errorf("%d logits differ between the two forms", d)
	}
	if r := float64(fall) / float64(islandBytes); r < 0.95 || r > 1.05 {
		t.Errorf("the heap fell by %d B, want the islands' %d B within 5%%", fall, islandBytes)
	}
}
