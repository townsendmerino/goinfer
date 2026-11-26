package decoder

import (
	"context"
	"errors"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// rowPadArms runs fn with m's row pad on (at alignment align, in bytes) or off, and restores the default alignment after. Both arms of
// f1Arms share m's Architecture, so the setting reaches either.
func rowPadArms[T any](t *testing.T, m *Model, on bool, align int, fn func() T) T {
	t.Helper()
	m.w.Arch.KVRowPadAlign = 0
	if on {
		m.w.Arch.KVRowPadAlign = align
	}
	defer func() { m.w.Arch.KVRowPadAlign = kvRowPadAlignDefault }()
	return fn()
}

// TestKVRowPad_padOnlyAtAnAlignedRow pins the pad rule: a row of elems elements of elemBytes bytes is padded by one 64 B line when its
// bytes are a multiple of the alignment (4 KiB by default), and not otherwise.
func TestKVRowPad_padOnlyAtAnAlignedRow(t *testing.T) {
	for _, c := range []struct{ elems, elemBytes, want int }{
		{3072, 4, 16}, // Phi-3 mini f32: 12 KiB
		{3072, 2, 0},  // Phi-3 mini f16: 6 KiB
		{1024, 4, 16}, // 4 KiB
		{2048, 2, 32}, // f16, 4 KiB
		{4096, 2, 32}, // f16, 8 KiB
		{128, 4, 0},   // Qwen2.5-0.5B f32: 512 B
		{1000, 4, 0},
	} {
		if got := archparts.KVRowPadElems(c.elems, c.elemBytes, kvRowPadAlignDefault); got != c.want {
			t.Errorf("kvRowPadElems(%d, %d) = %d, want %d", c.elems, c.elemBytes, got, c.want)
		}
	}
}

// rowPadTestAlign is the alignment the tests pad at: 32 B divides every dense fixture's f32 and f16 rows (qwen3moe-tiny's f16 row is
// 32 B), so their caches take the pad.
const rowPadTestAlign = 32

// rowPadFixtures are the tiny fixtures the pad cells run on, and whether the pad applies to each: dense families with uniform
// attention layers take it; a family with a sliding window (rings) is declined it.
var rowPadFixtures = []struct {
	dir    string
	n      int
	padded bool
}{
	{"llama-tiny", 40, true},
	{"qwen25vl-tiny", 24, true},
	{"mixtral-tiny", 32, true},
	{"qwen3moe-tiny", 32, true},
	{"glm-tiny", 32, true},
	{"phi3-tiny", 32, true},
	{"llama4-tiny", 32, true}, // its attention runs attendQuery (chunked RoPE layers), not attendToken
	{"gemma2-tiny", 30, false},
}

// rowPadCells runs every reader of the CPU KV layout once on model m, through its real caller, and returns each cell's output:
// decode, batched prefill, prefill then decode, the GPU-upload bridge's rows, rewind then decode, a session snapshot's restore,
// a prefix copy, and the multi-sequence walker. It also returns the prefill cache, so a caller can check whether it was padded.
func rowPadCells(t *testing.T, m *Model, n int) (map[string][][]float32, *KVCache) {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(n, vocab)
	step := func(c *KVCache, id int) []float32 {
		lg, err := m.forward(id, c)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	out := map[string][][]float32{}
	out["decode"] = f1Decode(t, m, ids)
	lg, pc := f1Prefill(t, m, ids[:n-4], nil)
	out["prefill"] = lg
	for _, id := range ids[n-4:] {
		out["prefill+decode"] = append(out["prefill+decode"], step(pc, id))
	}
	for l := range pc.numLayers {
		k, v, _ := pc.LayerKV(l)
		out["bridge"] = append(out["bridge"], k, v)
	}
	pc.TruncateTo(n - 6)
	for _, id := range ids[n-6 : n-2] {
		out["rewind"] = append(out["rewind"], step(pc, id))
	}
	s := m.NewSession(n + 8)
	if _, err := m.forwardN(context.Background(), ids, s.cache); err != nil {
		t.Fatal(err)
	}
	s.tokens = append([]int(nil), ids...)
	next := (ids[n-1] + 1) % vocab
	if blob := s.Snapshot("id"); blob != nil {
		back, err := m.LoadSession(blob, "id")
		if err != nil {
			t.Fatalf("LoadSession: %v", err)
		}
		out["snapshot"] = [][]float32{step(back.cache, next)}
	} else {
		t.Fatalf("Snapshot refused a %d-row cache", n)
	}
	cp := m.NewCache(n + 8)
	if err := cp.copyPrefixFrom(s.cache, n); errors.Is(err, errCopyLayout) {
		out["copy refused"] = [][]float32{{1}} // a layout copyPrefixFrom declines (both arms must)
	} else if err != nil {
		t.Fatalf("copyPrefixFrom: %v", err)
	} else {
		out["copy"] = [][]float32{step(cp, next)}
	}
	a, b := m.NewCache(n+8), m.NewCache(n+8)
	for i := range n / 2 {
		lgs, err := m.decodeMultiStep([]int{ids[i], ids[n-1-i]}, []*KVCache{a, b})
		if err != nil && i == 0 {
			out["multi refused"] = [][]float32{{1}} // the walker declines this family (both arms must)
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		out["multi"] = append(out["multi"], lgs...)
	}
	return out, pc
}

// TestKVRowPad_everyReaderBitIdentical holds every reader of the CPU KV layout bit-identical with the row pad on and off, at f32 and
// f16, on the dense fixtures (rowPadTestAlign, so their small rows are padded) and on a ring family the pad declines.
func TestKVRowPad_everyReaderBitIdentical(t *testing.T) {
	for _, fx := range rowPadFixtures {
		m := f1Load(t, filepath.Join("..", "testdata", fx.dir))
		f16, f32 := f1Arms(m)
		for _, arm := range []struct {
			name string
			m    *Model
		}{{"f32", f32}, {"f16", f16}} {
			var rc *KVCache
			ref := rowPadArms(t, arm.m, false, 0, func() map[string][][]float32 {
				o, c := rowPadCells(t, arm.m, fx.n)
				rc = c
				return o
			})
			var pc *KVCache
			got := rowPadArms(t, arm.m, true, rowPadTestAlign, func() map[string][][]float32 {
				o, c := rowPadCells(t, arm.m, fx.n)
				pc = c
				return o
			})
			if (pc.rowPad > 0) != fx.padded {
				t.Fatalf("%s %s: the pad-on prefill cache has rowPad %d, want padded=%v", fx.dir, arm.name, pc.rowPad, fx.padded)
			}
			cells := make([]string, 0, len(ref))
			for c := range ref {
				cells = append(cells, c)
			}
			slices.Sort(cells)
			t.Logf("%s %s: rowPad %d, cells %v", fx.dir, arm.name, pc.rowPad, cells)
			if rc.rowPad != 0 {
				t.Fatalf("%s %s: the pad-off cache is padded (%d)", fx.dir, arm.name, rc.rowPad)
			}
			for cell := range got {
				if _, ok := ref[cell]; !ok {
					t.Errorf("%s %s: cell %q ran only with the pad on", fx.dir, arm.name, cell)
				}
			}
			for cell, want := range ref {
				if d := f1Diff(got[cell], want); d != 0 || len(got[cell]) != len(want) {
					t.Errorf("%s %s %s: padded differs from unpadded in %d values (%d against %d rows)", fx.dir, arm.name, cell, d, len(got[cell]), len(want))
				}
			}
		}
	}
}

// TestKVRowPad_padAppliesWhereEligible checks the pad is on for the dense fixtures' caches and off for the ring family's, so the
// bit-identity test above is not comparing two unpadded runs.
func TestKVRowPad_padAppliesWhereEligible(t *testing.T) {
	for _, fx := range rowPadFixtures {
		m := f1Load(t, filepath.Join("..", "testdata", fx.dir))
		f16, f32 := f1Arms(m)
		for _, mm := range []*Model{f32, f16} {
			c := rowPadArms(t, mm, true, rowPadTestAlign, func() *KVCache { return mm.NewCache(16) })
			if (c.rowPad > 0) != fx.padded {
				t.Errorf("%s: rowPad %d, want padded=%v", fx.dir, c.rowPad, fx.padded)
			}
		}
	}
}

// TestKVRowPad_snapshotCrossesLayouts restores a padded session into an unpadded cache and the reverse: the snapshot format holds
// compact rows, so the next step matches either way.
func TestKVRowPad_snapshotCrossesLayouts(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	_, f32 := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(24, vocab)
	next := (ids[len(ids)-1] + 1) % vocab
	snap := func(padded bool) []byte {
		return rowPadArms(t, f32, padded, rowPadTestAlign, func() []byte {
			s := f32.NewSession(64)
			if _, err := f32.forwardN(context.Background(), ids, s.cache); err != nil {
				t.Fatal(err)
			}
			s.tokens = append([]int(nil), ids...)
			return s.Snapshot("id")
		})
	}
	restoreStep := func(blob []byte, padded bool) []float32 {
		return rowPadArms(t, f32, padded, rowPadTestAlign, func() []float32 {
			back, err := f32.LoadSession(blob, "id")
			if err != nil {
				t.Fatalf("LoadSession: %v", err)
			}
			if (back.cache.rowPad > 0) != padded {
				t.Fatalf("restored cache rowPad %d, want padded=%v", back.cache.rowPad, padded)
			}
			lg, err := f32.forward(next, back.cache)
			if err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), lg...)
		})
	}
	want := restoreStep(snap(false), false)
	for _, c := range []struct{ from, to bool }{{true, false}, {false, true}, {true, true}} {
		if d := f1Diff([][]float32{restoreStep(snap(c.from), c.to)}, [][]float32{want}); d != 0 {
			t.Errorf("snapshot padded=%v restored padded=%v: the next step differs in %d values", c.from, c.to, d)
		}
	}
}

// TestKVRowPad_planPricesThePad holds the CPU KV price to what the cache holds with the pad on, on every fixture at both
// precisions: a padded fixture's priced bytes equal its global stores' bytes after ctx positions and exceed the unpadded price; a
// declined one prices as unpadded.
func TestKVRowPad_planPricesThePad(t *testing.T) {
	const ctx = 24
	for _, fx := range rowPadFixtures {
		m := f1Load(t, filepath.Join("..", "testdata", fx.dir))
		f16, f32 := f1Arms(m)
		_, _, _, _, _, _, vocab := m.Dims()
		ids := f1IDs(ctx, vocab)
		for _, arm := range []struct {
			name string
			m    *Model
			f16  bool
		}{{"f32", f32, false}, {"f16", f16, true}} {
			unpadded := rowPadArms(t, arm.m, false, 0, func() int64 { return kvBytesForCtx(arm.m.w.Arch, ctx, arm.f16, false) })
			rowPadArms(t, arm.m, true, rowPadTestAlign, func() int {
				c := arm.m.NewCache(ctx)
				for _, id := range ids {
					if _, err := arm.m.forward(id, c); err != nil {
						t.Fatal(err)
					}
				}
				priced := kvBytesForCtx(arm.m.w.Arch, ctx, arm.f16, false)
				switch {
				case c.rowPad == 0 && priced != unpadded:
					t.Errorf("%s %s: unpadded cache, but the plan prices %d against the unpadded %d", fx.dir, arm.name, priced, unpadded)
				case c.rowPad > 0 && (priced != c.rowPadBytesHeld() || priced <= unpadded):
					t.Errorf("%s %s: the padded cache holds %d bytes at %d positions; the plan prices %d (unpadded %d)", fx.dir, arm.name, c.rowPadBytesHeld(), ctx, priced, unpadded)
				}
				return 0
			})
		}
	}
}

// TestKVRowPad_keysRefusesAPaddedCache checks the unstrided accessors refuse a padded cache: a reader that indexes Keys/Vals by kvDim
// would read the pad as data.
func TestKVRowPad_keysRefusesAPaddedCache(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	_, f32 := f1Arms(m)
	c := rowPadArms(t, f32, true, rowPadTestAlign, func() *KVCache { return f32.NewCache(16) })
	for name, fn := range map[string]func(){"Keys": func() { c.Keys(0) }, "Vals": func() { c.Vals(0) }} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s accepted a padded cache", name)
				}
			}()
			fn()
		}()
	}
}

// kvAccessorCall matches a call of a Keys or Vals method on anything but the standard library's maps package.
var kvAccessorCall = regexp.MustCompile(`\b(?:[A-Za-z_][A-Za-z0-9_]*)\.(?:Keys|Vals)\(`)

// TestKVRowPad_keysReadersAreDeclinedFamilies is the reader census: every production call of KVCache.Keys or Vals in decoder/ is in a
// family loop whose caches the pad declines (rings, hybrid state or per-layer widths), so a new reader that indexes the unstrided store
// shows up here instead of reading the pad as data.
func TestKVRowPad_keysReadersAreDeclinedFamilies(t *testing.T) {
	allowed := map[string]bool{
		"forward_gptoss.go":       true, // sliding-window layers: rings
		"forward_qwen35.go":       true, // hybrid: manualPos
		"forward_gemma4.go":       true, // per-layer widths, KV sharing: manualPos
		"program_cpu_n_gemma4.go": true,
		"program_cpu_n_qwen35.go": true,
		"kvcache.go":              true, // the accessors themselves
		"testhooks.go":            true,
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || allowed[f] {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.HasPrefix(strings.TrimSpace(line), "//") {
				continue
			}
			if slices.ContainsFunc(kvAccessorCall.FindAllString(line, -1), func(m string) bool { return !strings.HasPrefix(m, "maps.") }) {
				t.Errorf("%s:%d reads KVCache.Keys/Vals, which a padded cache refuses; read layerRows (and its stride) or decline the pad for this family: %s", f, i+1, strings.TrimSpace(line))
			}
		}
	}
}
