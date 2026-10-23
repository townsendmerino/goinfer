package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
)

const qwen35VLDir = "../testdata/qwen35vl-tiny"

// v3Run is one Qwen3.5 image prefill of route 1's prompt on a fresh cache, with prefillChunkRows at chunk: the last logits and four
// greedy decode steps after, and the passes it took.
func v3Run(t *testing.T, m *Model, chunk int) ([][]float32, int64) {
	t.Helper()
	prev := prefillChunkRows
	prefillChunkRows = chunk
	defer func() { prefillChunkRows = prev }()
	ids, spans, feats, mrope, _ := v1Prompt(m)
	cache := m.NewCache(len(ids) + 8)
	p0 := atomic.LoadInt64(&prefillPasses)
	lg, err := m.prefillLogitsQwenVLSpans(context.Background(), ids, spans, feats, mrope, cache)
	if err != nil {
		t.Fatal(err)
	}
	passes := atomic.LoadInt64(&prefillPasses) - p0
	return v1Decode(t, m, cache, lg), passes
}

// TestMEM5VL_qwen35Chunked is V3-G1 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL (3) gates"): Qwen3.5's image prefill in
// 7-, 16- and 59-row passes (the last a one-row pass, which takes the per-token DeltaNet core) against one pass; the last logits and
// four decode steps after are byte-equal.
func TestMEM5VL_qwen35Chunked(t *testing.T) {
	for _, q := range []string{"", "int4"} {
		m, err := Load(qwen35VLDir, Options{Quant: q})
		if err != nil {
			t.Fatal(err)
		}
		if !m.qwen35BatchNAnyPos(60, m.NewCache(68)) {
			t.Fatalf("%q: the fixture does not take the batched Qwen3.5 prefill", q)
		}
		want, p1 := v3Run(t, m, 1<<20)
		if p1 != 1 {
			t.Errorf("%q: the single pass took %d passes", q, p1)
		}
		line := fmt.Sprintf("[V3-G1] qwen35vl-tiny %-6q", q)
		for _, chunk := range []int{7, 16, 59} {
			got, passes := v3Run(t, m, chunk)
			d := 0
			for s := range want {
				d += v1Differ(want[s], got[s])
			}
			line += fmt.Sprintf(" | C=%d passes=%d logits differ %d", chunk, passes, d)
			if d != 0 || passes < 2 {
				t.Errorf("%q C=%d: %d logits differ from one pass, in %d passes", q, chunk, d, passes)
			}
		}
		fmt.Fprintln(os.Stderr, line)
		m.Close()
	}
}

// TestMEM5VL_qwen35Bytes is V3-G2: a 200-row Qwen3.5 image prefill in 16-row passes. The largest pass allocates no more than the
// single pass's allocation scaled to its rows, plus its key-sized attention scratch and 64 KiB; the KV arrays a pass grows are left
// out of its reading.
func TestMEM5VL_qwen35Bytes(t *testing.T) {
	m, err := Load(qwen35VLDir, Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	arch := m.w.arch
	hidden := arch.HiddenDim
	const n, chunk = 200, 16
	ids := make([]int, n)
	mrope := make([][3]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % vocab
		mrope[i] = [3]int{i, i, i}
	}
	spans := []ImageSpan{{Pos: 50, Len: 20, Hash: 1}, {Pos: 120, Len: 6, Hash: 2}}
	feats := make([]float32, 26*hidden)
	for i := range feats {
		feats[i] = float32(i%13) * 0.01
	}
	run := func(C int) []uint64 {
		prev, prevDone := prefillChunkRows, prefillChunkDone
		defer func() { prefillChunkRows, prefillChunkDone = prev, prevDone }()
		prefillChunkRows = C
		var allocs []uint64
		cache := m.NewCache(n + 8)
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		last, caps := ms.TotalAlloc, v2KVCaps(cache)
		prefillChunkDone = func(i int) {
			runtime.ReadMemStats(&ms)
			now := v2KVCaps(cache)
			if i >= 0 {
				allocs = append(allocs, ms.TotalAlloc-last-v2KVGrown(caps, now))
			}
			last, caps = ms.TotalAlloc, now
		}
		if _, err := m.prefillLogitsQwenVLSpans(context.Background(), ids, spans, feats, mrope, cache); err != nil {
			t.Fatal(err)
		}
		return allocs
	}
	whole := run(1 << 20)
	passes := run(chunk)
	// The key-sized scratch is the full-attention layers' worker pool at the last pass's keys (newHeadWorkerPoolK: per worker, keys
	// and values of nKeys×hd and a row tile of scores), plus the widened K/V copies when the cache stores them narrower.
	hd, nKeys := arch.HeadDim, n
	workers := min(max(prefillAttnWorkersK(m.knobs, chunk, nKeys, hd, arch.NumHeads), 1), maxAttnWorkers)
	tile := attnRowTileK(m.knobs, chunk, nKeys)
	keyScratch := uint64(4 * workers * (2*nKeys*hd + tile*nKeys + 2*tile*hd + 2*hd))
	keyScratch += uint64(4 * 2 * nKeys * arch.NumKVHeads * hd)
	var largest uint64
	for _, a := range passes {
		largest = max(largest, a)
	}
	bound := whole[0]*chunk/n + keyScratch + 64<<10
	fmt.Fprintf(os.Stderr, "[V3-G2] qwen35vl-tiny: one pass allocates %d B for %d rows; %d passes, the largest %d B (bound %d B: %d rows' share, key scratch %d B, 64 KiB); each pass %v\n",
		whole[0], n, len(passes), largest, bound, chunk, keyScratch, passes)
	if largest > bound {
		t.Errorf("the largest pass allocated %d B, bound %d B", largest, bound)
	}
}
