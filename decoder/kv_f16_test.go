package decoder

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Gate F1 of docs/tasks/task-cpu-kv-f16-2026-10.md (registered before any code): an f16 CPU cache gives the logits, bit for
// bit, of the f32 path with every stored K and V row rounded to f16 by hand (kvRoundF16ForTest), on every path that stores
// or reads the cache. Zero differing bits in every cell; no tolerance.

// f1Arms returns m's two arms: the f16 cache, and the f32 reference with the hand rounding. The reference hook is global,
// so the caller runs the arms one after the other through f1Run.
func f1Arms(m *Model) (f16, ref *Model) {
	a, b := *m, *m
	a.kvH, b.kvH = true, false
	return &a, &b
}

// f1Run runs fn with the hand rounding on when ref is set.
func f1Run[T any](ref bool, fn func() T) T {
	kvRoundF16ForTest = ref
	defer func() { kvRoundF16ForTest = false }()
	return fn()
}

// f1Diff counts differing bits between two logit sets.
func f1Diff(a, b [][]float32) int {
	n := 0
	for i := range a {
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				n++
			}
		}
	}
	return n
}

func f1Load(t *testing.T, dir string) *Model {
	t.Helper()
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func f1IDs(n, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (7 + 37*i) % vocab
	}
	return ids
}

// f1Decode is the decode cell: n tokens one at a time from a fresh cache, every step's logits.
func f1Decode(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	c := m.NewCache(len(ids))
	var out [][]float32
	for _, id := range ids {
		lg, err := m.forward(id, c)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// f1Prefill is the batched cell: every row's logits of one forwardN over ids, after setup(cache) (an image block, m-RoPE).
func f1Prefill(t *testing.T, m *Model, ids []int, setup func(*KVCache)) ([][]float32, *KVCache) {
	t.Helper()
	c := m.NewCache(len(ids) + 8)
	if setup != nil {
		setup(c)
	}
	lg, err := m.forwardN(context.Background(), ids, c)
	if err != nil {
		t.Fatal(err)
	}
	return lg, c
}

// f1Cells runs every F1 cell on one fixture and returns the differing bits per cell.
func f1Cells(t *testing.T, m *Model, n int, setup func(*KVCache)) map[string]int {
	t.Helper()
	f16, ref := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(n, vocab)
	got := map[string]int{}
	got["decode"] = f1Diff(f1Run(false, func() [][]float32 { return f1Decode(t, f16, ids) }), f1Run(true, func() [][]float32 { return f1Decode(t, ref, ids) }))
	type pf struct {
		lg [][]float32
		c  *KVCache
	}
	pa := f1Run(false, func() pf { lg, c := f1Prefill(t, f16, ids, setup); return pf{lg, c} })
	pb := f1Run(true, func() pf { lg, c := f1Prefill(t, ref, ids, setup); return pf{lg, c} })
	if pa.c.quant != kvqF16 || pb.c.quant != kvqF32 {
		t.Fatalf("arms' caches are %d and %d, want f16 and f32", pa.c.quant, pb.c.quant)
	}
	got["prefill"] = f1Diff(pa.lg, pb.lg)
	// The GPU-upload bridge: every layer's K and V as the bridge reads them.
	bridge := 0
	for l := range pa.c.numLayers {
		ka, va, _ := pa.c.LayerKV(l)
		kb, vb, _ := pb.c.LayerKV(l)
		bridge += f1Diff([][]float32{ka, va}, [][]float32{kb, vb})
	}
	got["bridge"] = bridge
	// Rewind: back 5 positions, then 5 decode steps.
	rewind := func(mm *Model, c *KVCache) [][]float32 {
		c.TruncateTo(len(ids) - 5)
		var out [][]float32
		for _, id := range ids[len(ids)-5:] {
			lg, err := mm.forward(id, c)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	got["rewind"] = f1Diff(f1Run(false, func() [][]float32 { return rewind(f16, pa.c) }), f1Run(true, func() [][]float32 { return rewind(ref, pb.c) }))
	return got
}

// TestKVF16_F1Exactness is gate F1's tiny-fixture cells: decode, batched prefill under 512 rows, the GPU-upload bridge and
// rewind, on llama-tiny (dense), gemma2-tiny (a sliding window, so the ring) and qwen25vl-tiny (m-RoPE positions and an image
// block). Every cell must show zero differing bits.
func TestKVF16_F1Exactness(t *testing.T) {
	vlSetup := func(c *KVCache) {
		c.SetImageBlocks([][2]int{{4, 12}})
		pos := make([][3]int, 64)
		for i := range pos {
			pos[i] = [3]int{i, i, i}
			if i >= 4 && i < 12 { // an image's (t, h, w) grid
				pos[i] = [3]int{4, 4 + (i-4)/4, 4 + (i-4)%4}
			}
		}
		c.mropePos = pos
	}
	for _, fx := range []struct {
		dir   string
		n     int
		setup func(*KVCache)
	}{
		{"llama-tiny", 40, nil},
		{"gemma2-tiny", 30, nil},
		{"qwen25vl-tiny", 24, vlSetup},
	} {
		m := f1Load(t, filepath.Join("..", "testdata", fx.dir))
		if !m.w.arch.kvF16OK() {
			t.Fatalf("%s: kvF16OK is false", fx.dir)
		}
		for cell, d := range f1Cells(t, m, fx.n, fx.setup) {
			t.Logf("%s %s: %d differing bits", fx.dir, cell, d)
			if d != 0 {
				t.Errorf("%s %s: the f16 cache differs from the hand-rounded f32 path in %d values", fx.dir, cell, d)
			}
		}
	}
}

// TestKVF16_F1PrefillEqualsDecode: on the f16 path, a batched forward over N tokens gives the logits of N-1 batched and one
// decode step, on a global layer (llama-tiny) and a ring layer (gemma2-tiny).
func TestKVF16_F1PrefillEqualsDecode(t *testing.T) {
	for _, dir := range []string{"llama-tiny", "gemma2-tiny"} {
		m := f1Load(t, filepath.Join("..", "testdata", dir))
		f16, _ := f1Arms(m)
		_, _, _, _, _, _, vocab := m.Dims()
		ids := f1IDs(20, vocab)
		whole, _ := f1Prefill(t, f16, ids, nil)
		_, c := f1Prefill(t, f16, ids[:len(ids)-1], nil)
		last, err := f16.forward(ids[len(ids)-1], c)
		if err != nil {
			t.Fatal(err)
		}
		if d := f1Diff([][]float32{whole[len(ids)-1]}, [][]float32{last}); d != 0 {
			t.Errorf("%s: the batched forward's last row differs from N-1 batched plus one decode step in %d values", dir, d)
		}
	}
}

// TestKVF16_F1SnapshotAndCopy: an f16 session survives a snapshot round trip and KV2's session copy bit for bit (the next
// decode step is identical), and the planted defect of a snapshot whose storage byte reads f32 is refused.
func TestKVF16_F1SnapshotAndCopy(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	f16, _ := f1Arms(m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := f1IDs(24, vocab)
	s := f16.NewSession(64)
	if _, err := f16.forwardN(context.Background(), ids, s.cache); err != nil {
		t.Fatal(err)
	}
	s.tokens = append([]int(nil), ids...)
	next := (ids[len(ids)-1] + 1) % vocab
	step := func(c *KVCache) []float32 {
		lg, err := f16.forward(next, c)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	blob := s.Snapshot("id")
	back, err := f16.LoadSession(blob, "id")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	cp := f16.NewCache(64)
	if err := cp.copyPrefixFrom(s.cache, len(ids)); err != nil {
		t.Fatalf("copyPrefixFrom: %v", err)
	}
	want := step(s.cache)
	if d := f1Diff([][]float32{step(back.cache)}, [][]float32{want}); d != 0 {
		t.Errorf("the restored snapshot's next step differs in %d values", d)
	}
	if d := f1Diff([][]float32{step(cp)}, [][]float32{want}); d != 0 {
		t.Errorf("the copied session's next step differs in %d values", d)
	}
	kvSnapF16ByteF32ForTest = true
	planted := s.Snapshot("id")
	kvSnapF16ByteF32ForTest = false
	if _, err := f16.LoadSession(planted, "id"); err == nil {
		t.Error("a snapshot whose storage byte reads f32 restored into an f16 cache: the plant is not refused")
	}
}

// TestKVF16_F1Fast512: the batched prefill at 512 rows or more (the fast path's per-head gather), on Qwen2.5-Coder-0.5B, the
// smallest model whose window holds that many rows. Zero differing bits.
func TestKVF16_F1Fast512(t *testing.T) {
	path := filepath.Join(os.Getenv("HOME"), "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.canBatchN(600) {
		t.Skip("no batched prefill at 600 rows")
	}
	f16, ref := f1Arms(m)
	ids := f1IDs(600, 20000)
	a := f1Run(false, func() [][]float32 { lg, _ := f1Prefill(t, f16, ids, nil); return lg })
	b := f1Run(true, func() [][]float32 { lg, _ := f1Prefill(t, ref, ids, nil); return lg })
	d := f1Diff(a, b)
	t.Logf("600-row prefill: %d differing values", d)
	if d != 0 {
		t.Errorf("the 600-row prefill on the f16 cache differs from the hand-rounded f32 path in %d values", d)
	}
}

// TestKVF16_F1Plants: each planted defect turns its cell red on gemma2-tiny's ring: a prefill's new rows attended at f32, and
// the ring window read one slot late.
func TestKVF16_F1Plants(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "gemma2-tiny"))
	for _, p := range []struct {
		name string
		on   *bool
	}{{"new prefill rows at f32", &kvF16NewRowsF32ForTest}, {"ring read one slot late", &kvF16RingOffsetForTest}} {
		*p.on = true
		cells := f1Cells(t, m, 30, nil)
		*p.on = false
		red := cells["prefill"] + cells["decode"] + cells["rewind"]
		t.Logf("plant %q: %d differing values (prefill %d, decode %d, rewind %d)", p.name, red, cells["prefill"], cells["decode"], cells["rewind"])
		if red == 0 {
			t.Errorf("plant %q: F1 does not see it", p.name)
		}
	}
}

// TestKVF16_excludedFamilyKeepsF32: an f16 request on a family outside kvF16OK (mixtral-tiny, MoE) builds an f32 cache and
// says why.
func TestKVF16_excludedFamilyKeepsF32(t *testing.T) {
	m, err := Load(filepath.Join("..", "testdata", "mixtral-tiny"), Options{KVQuant: "f16"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if c := m.NewCache(8); c.quant != kvqF32 {
		t.Errorf("mixtral-tiny's cache is %d, want f32", c.quant)
	}
	if p, why := m.CPUKVPrecision(), m.CPUKVF16DeclineReason(); p != "f32" || why == "" {
		t.Errorf("CPUKVPrecision %q, reason %q: want f32 and a reason", p, why)
	}
}

// TestKVF16_F5LibraryDefaultIsF32: decoder.Load with zero options builds an f32 CPU cache, the guard against the library
// default following the command lines (gate F5).
func TestKVF16_F5LibraryDefaultIsF32(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	if c := m.NewCache(8); c.quant != kvqF32 || m.CPUKVPrecision() != "f32" {
		t.Errorf("zero options built a %d cache (CPUKVPrecision %q), want f32", c.quant, m.CPUKVPrecision())
	}
}
