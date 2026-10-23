package decoder

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
)

// v1Prompt is a 60-token prompt with two image spans: one longer than a 7-row chunk and crossing the 16-row cut, and one a cut
// would split at 16 rows. feats is every span's rows; mrope gives each position an m-RoPE triple (text diagonal, the image a 3×4
// grid then a 2×2), deep two DeepStack sets.
func v1Prompt(m *Model) (ids []int, spans []ImageSpan, feats []float32, mrope [][3]int, deep [][]float32) {
	_, _, _, _, _, _, vocab := m.Dims()
	hidden := m.w.arch.HiddenDim
	ids = make([]int, 60)
	for i := range ids {
		ids[i] = (i*37 + 11) % vocab
	}
	spans = []ImageSpan{{Pos: 5, Len: 12, Hash: 1}, {Pos: 30, Len: 4, Hash: 2}}
	r := rand.New(rand.NewSource(5))
	rows := func(n int) []float32 {
		f := make([]float32, n*hidden)
		for i := range f {
			f[i] = float32(r.NormFloat64()) * 0.5
		}
		return f
	}
	feats = rows(16)
	deep = [][]float32{rows(16), rows(16)}
	mrope = make([][3]int, len(ids))
	next := 0
	for i := 0; i < len(ids); {
		switch i {
		case 5, 30:
			h, w := 3, 4
			if i == 30 {
				h, w = 2, 2
			}
			for y := range h {
				for x := range w {
					mrope[i] = [3]int{next, next + y, next + x}
					i++
				}
			}
			next += max(h, w)
		default:
			mrope[i] = [3]int{next, next, next}
			next++
			i++
		}
	}
	return ids, spans, feats, mrope, deep
}

// v1Run is one image prefill of ids on a fresh cache with prefillChunkRows at chunk: every position's rows, the last logits and
// four greedy decode steps after, and the passes it took.
func v1Run(t *testing.T, m *Model, route string, chunk int) (rows []float32, logits [][]float32, passes int64) {
	t.Helper()
	prev := prefillChunkRows
	prefillChunkRows = chunk
	defer func() { prefillChunkRows = prev }()
	ids, spans, feats, mrope, deep := v1Prompt(m)
	ctx := context.Background()
	p0 := atomic.LoadInt64(&prefillPasses)
	var lg []float32
	var err error
	switch route {
	case "bidirectional", "causal":
		causal := route == "causal"
		if rows, err = m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)+8), causal); err != nil {
			t.Fatal(err)
		}
		cache := m.NewCache(len(ids) + 8)
		p0 = atomic.LoadInt64(&prefillPasses)
		lg, err = m.prefillLogitsVLSpans(ctx, ids, spans, feats, cache, causal)
		if err != nil {
			t.Fatal(err)
		}
		logits = v1Decode(t, m, cache, lg)
	case "mrope", "deepstack":
		c := m.NewCache(len(ids) + 8)
		c.mropePos, c.mropeDelta = mrope, mropeDelta(mrope, len(ids))
		if route == "deepstack" {
			c.deepstack = &deepstackRows{spans: spans, rows: deep}
		}
		if rows, err = m.prefillSpliced(ctx, ids, spans, feats, c, true); err != nil {
			t.Fatal(err)
		}
		cache := m.NewCache(len(ids) + 8)
		if route == "deepstack" {
			cache.deepstack = &deepstackRows{spans: spans, rows: deep}
		}
		p0 = atomic.LoadInt64(&prefillPasses)
		lg, err = m.prefillLogitsQwenVLSpans(ctx, ids, spans, feats, mrope, cache)
		cache.deepstack = nil
		if err != nil {
			t.Fatal(err)
		}
		logits = v1Decode(t, m, cache, lg)
	}
	return rows, logits, atomic.LoadInt64(&prefillPasses) - p0
}

func v1Decode(t *testing.T, m *Model, cache *KVCache, lg []float32) [][]float32 {
	t.Helper()
	out := [][]float32{append([]float32(nil), lg...)}
	for range 4 {
		best := 0
		for j := range lg {
			if lg[j] > lg[best] {
				best = j
			}
		}
		var err error
		if lg, err = m.forward(best, cache); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

func v1Differ(a, b []float32) int {
	n := 0
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			n++
		}
	}
	return n
}

// TestMEM5VL_sharedRouteChunked is V1-G1 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL (1) gates"): the shared batched
// route's image prefill run in 7- and 16-row passes against one pass, on every family it serves: every position's rows, the last
// logits and four decode steps after are byte-equal, and each chunked run really took several passes.
func TestMEM5VL_sharedRouteChunked(t *testing.T) {
	cases := []struct{ dir, route string }{
		{"../testdata/gemma3-vl-tiny", "bidirectional"},
		{"../testdata/ministral3-tiny", "causal"},
		{"../testdata/qwen25vl-tiny", "mrope"},
		{"../testdata/glm-ocr-tiny", "mrope"},
		{"../testdata/qwen3vl-tiny", "deepstack"},
		{"../testdata/qwen3vlmoe-tiny", "deepstack"},
	}
	ran := 0
	for _, c := range cases {
		for _, q := range []string{"", "int4"} {
			name := fmt.Sprintf("%s/%s/%q", c.dir[len("../testdata/"):], c.route, q)
			if _, err := os.Stat(c.dir + "/model.safetensors"); err != nil {
				t.Logf("%s: no fixture", name)
				continue
			}
			m, err := Load(c.dir, Options{Quant: q})
			if err != nil {
				t.Fatalf("%s: load: %v", name, err)
			}
			wantRows, want, p1 := v1Run(t, m, c.route, 1<<20)
			if p1 != 1 {
				t.Errorf("%s: the single pass took %d passes", name, p1)
			}
			line := fmt.Sprintf("[V1-G1] %-40s", name)
			for _, chunk := range []int{7, 16} {
				rows, got, passes := v1Run(t, m, c.route, chunk)
				dr, dl := v1Differ(wantRows, rows), 0
				for s := range want {
					dl += v1Differ(want[s], got[s])
				}
				line += fmt.Sprintf(" | C=%d passes=%d rows differ %d, logits differ %d", chunk, passes, dr, dl)
				if dr != 0 || dl != 0 || passes < 2 {
					t.Errorf("%s C=%d: %d rows and %d logits differ from one pass, in %d passes", name, chunk, dr, dl, passes)
				}
			}
			fmt.Fprintln(os.Stderr, line)
			m.Close()
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no fixture ran")
	}
}

// v1RowBytes is the bytes of a pass working set's row-sized buffers, by capacity, and of its key-sized attention buffers (alk,
// alv: sized by the whole prefill's keys in every pass, G40's amendment).
func v1RowBytes(ps *prefillPassScratch) (rows, keys int64) {
	var n int
	for _, b := range [][]float32{ps.h, ps.norm, ps.q, ps.k, ps.v, ps.ctx, ps.att, ps.gate, ps.up, ps.mlpOut} {
		n += cap(b)
	}
	return 4 * int64(n), 4 * int64(cap(ps.alk)+cap(ps.alv)+cap(ps.gbuf))
}

// TestMEM5VL_sharedRouteBytes is V1-G2: a 200-row image prefill whose first image span (20 rows) is longer than a 16-row chunk.
// The pass's row buffers hold the longest pass's rows (that block, where it is bidirectional; a chunk otherwise), not the prompt's, in exact proportion to the single pass's,
// and its key-sized attention buffers are the single pass's; every chunk after the first allocates at most 1 MiB.
func TestMEM5VL_sharedRouteBytes(t *testing.T) {
	for _, c := range []struct{ dir, route string }{{"../testdata/gemma3-vl-tiny", "bidirectional"}, {"../testdata/qwen25vl-tiny", "mrope"}} {
		m, err := Load(c.dir, Options{Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		hidden := m.w.arch.HiddenDim
		const n, chunk, block = 200, 16, 20
		ids := make([]int, n)
		mrope := make([][3]int, n)
		for i := range ids {
			ids[i] = (i*37 + 11) % vocab
			mrope[i] = [3]int{i, i, i}
		}
		spans := []ImageSpan{{Pos: 50, Len: block, Hash: 1}, {Pos: 120, Len: 6, Hash: 2}}
		feats := make([]float32, (block+6)*hidden)
		for i := range feats {
			feats[i] = float32(i%13) * 0.01
		}
		run := func(C int) (int64, int64, []uint64) {
			prev, prevSeen, prevDone := prefillChunkRows, prefillScratchSeen, prefillChunkDone
			defer func() { prefillChunkRows, prefillScratchSeen, prefillChunkDone = prev, prevSeen, prevDone }()
			prefillChunkRows = C
			var bytes, keys int64
			prefillScratchSeen = func(ps *prefillPassScratch) { bytes, keys = v1RowBytes(ps) }
			var allocs []uint64
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			last := ms.TotalAlloc
			prefillChunkDone = func(i int) {
				runtime.ReadMemStats(&ms)
				if i >= 0 {
					allocs = append(allocs, ms.TotalAlloc-last)
				}
				last = ms.TotalAlloc
			}
			cache := m.NewCache(n + 8)
			var err error
			if c.route == "mrope" {
				_, err = m.prefillLogitsQwenVLSpans(context.Background(), ids, spans, feats, mrope, cache)
			} else {
				_, err = m.prefillLogitsVLSpans(context.Background(), ids, spans, feats, cache, false)
			}
			if err != nil {
				t.Fatal(err)
			}
			return bytes, keys, allocs
		}
		whole, wholeKeys, _ := run(1 << 20)
		chunked, chunkedKeys, allocs := run(chunk)
		// The bidirectional route's passes at 16 rows: [0,16) [16,32) [32,48) [48,50) cut back to the block, [50,70) the block,
		// then 16-row passes. Qwen's image rows are causal (no block), so its passes are all 16 rows or fewer.
		longest := chunk
		if c.route == "bidirectional" {
			longest = block
		}
		fmt.Fprintf(os.Stderr, "[V1-G2] %s: row buffers %d B one pass (%d rows) -> %d B chunked (longest pass %d rows); key-sized %d / %d B; TotalAlloc per chunk after the first: %v\n",
			c.dir[len("../testdata/"):], whole, n, chunked, longest, wholeKeys, chunkedKeys, allocs[1:])
		if chunkedKeys != wholeKeys {
			t.Errorf("%s: key-sized buffers %d B chunked, %d B in one pass", c.dir, chunkedKeys, wholeKeys)
		}
		if chunked*n != whole*int64(longest) {
			t.Errorf("%s: chunked row buffers %d B, want the single pass's %d B scaled to %d of %d rows", c.dir, chunked, whole, longest, n)
		}
		for i, a := range allocs[1:] {
			if a > 1<<20 {
				t.Errorf("%s: chunk %d allocated %d B", c.dir, i+1, a)
			}
		}
		m.Close()
	}
}
