package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
)

const gemma4BidirDir = "../testdata/gemma4-vl-bidir-tiny"

// needGemma4Bidir skips when the gitignored bidirectional fixture is absent (scripts/pin_gemma4_vl_bidir_tiny.py makes it).
func needGemma4Bidir(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(gemma4BidirDir + "/model.safetensors"); err != nil {
		t.Skipf("no checkpoint at %s — run scripts/pin_gemma4_vl_bidir_tiny.py", gemma4BidirDir)
	}
}

// v2KVCaps is the byte capacity of each of c's KV arrays, in a fixed order.
func v2KVCaps(c *KVCache) []int {
	var out []int
	for l := range c.numLayers {
		out = append(out, 4*cap(c.keys[l]), 4*cap(c.vals[l]))
		if c.keysH != nil {
			out = append(out, 2*cap(c.keysH[l]), 2*cap(c.valsH[l]))
		}
		if c.keysQ != nil {
			out = append(out, cap(c.keysQ[l]), cap(c.valsQ[l]), 4*cap(c.keyScale[l]), 4*cap(c.valScale[l]))
		}
	}
	return out
}

// v2KVGrown is the bytes the arrays that were reallocated between two v2KVCaps readings now hold: the KV's own allocations,
// which a pass's working set does not include.
func v2KVGrown(before, after []int) uint64 {
	var n uint64
	for i := range after {
		if after[i] != before[i] {
			n += uint64(after[i])
		}
	}
	return n
}

// v2Run is one bidirectional Gemma 4 image prefill of route 1's prompt on a fresh cache, with prefillChunkRows at chunk: the last
// logits and four greedy decode steps after, and the passes it took.
func v2Run(t *testing.T, m *Model, chunk int) ([][]float32, int64) {
	t.Helper()
	prev := prefillChunkRows
	prefillChunkRows = chunk
	defer func() { prefillChunkRows = prev }()
	ids, spans, feats, _, _ := v1Prompt(m)
	cache := m.NewCache(len(ids) + 8)
	p0 := atomic.LoadInt64(&prefillPasses)
	lg, err := m.prefillLogitsGemma4VLBidirectionalSpans(context.Background(), ids, feats, spans, cache)
	if err != nil {
		t.Fatal(err)
	}
	passes := atomic.LoadInt64(&prefillPasses) - p0
	return v1Decode(t, m, cache, lg), passes
}

// TestMEM5VL_gemma4BidirChunked is V2-G1 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL (2) gates"): the bidirectional
// image prefill in 7- and 16-row passes against one pass; the last logits and four decode steps after are byte-equal.
func TestMEM5VL_gemma4BidirChunked(t *testing.T) {
	needGemma4Bidir(t)
	for _, q := range []string{"", "int4"} {
		m, err := Load(gemma4BidirDir, Options{Quant: q})
		if err != nil {
			t.Fatal(err)
		}
		want, p1 := v2Run(t, m, 1<<20)
		if p1 != 1 {
			t.Errorf("%q: the single pass took %d passes", q, p1)
		}
		line := fmt.Sprintf("[V2-G1] gemma4-vl-bidir-tiny %-6q", q)
		for _, chunk := range []int{7, 16} {
			got, passes := v2Run(t, m, chunk)
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

// TestMEM5VL_gemma4BidirBytes is V2-G2: a 200-row bidirectional prefill with a 20-row image block, in 16-row passes. The largest
// pass allocates no more than the single pass's allocation scaled to its rows, plus its key-sized attention scratch and 64 KiB.
// Each pass's reading leaves out the KV arrays it grew, which are the cache's storage, not the pass's.
func TestMEM5VL_gemma4BidirBytes(t *testing.T) {
	needGemma4Bidir(t)
	m, err := Load(gemma4BidirDir, Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	arch := m.w.arch
	hidden := arch.HiddenDim
	const n, chunk, block = 200, 16, 20
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % vocab
	}
	spans := []ImageSpan{{Pos: 50, Len: block, Hash: 1}, {Pos: 120, Len: 6, Hash: 2}}
	feats := make([]float32, (block+6)*hidden)
	for i := range feats {
		feats[i] = float32(i%13) * 0.01
	}
	run := func(C int) []uint64 {
		prev, prevDone := prefillChunkRows, prefillChunkDone
		defer func() { prefillChunkRows, prefillChunkDone = prev, prevDone }()
		prefillChunkRows = C
		var allocs []uint64
		cache := m.NewCache(n + 8) // before the baseline: the cache is not the pass's
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		last, caps := ms.TotalAlloc, v2KVCaps(cache)
		prefillChunkDone = func(i int) {
			runtime.ReadMemStats(&ms)
			now := v2KVCaps(cache)
			if i >= 0 { // -1 is the KV reserve, read apart from the passes
				allocs = append(allocs, ms.TotalAlloc-last-v2KVGrown(caps, now))
			}
			last, caps = ms.TotalAlloc, now
		}
		if _, err := m.prefillLogitsGemma4VLBidirectionalSpans(context.Background(), ids, feats, spans, cache); err != nil {
			t.Fatal(err)
		}
		return allocs
	}
	whole := run(1 << 20)
	passes := run(chunk)
	maxHd := max(arch.gemma4.GlobalHeadDim, arch.HeadDim)
	keyScratch := uint64(4 * (2*n*maxHd + arch.NumHeads*n))
	var largest uint64
	for _, a := range passes {
		largest = max(largest, a)
	}
	bound := whole[0]*block/n + keyScratch + 64<<10
	fmt.Fprintf(os.Stderr, "[V2-G2] gemma4-vl-bidir-tiny: one pass allocates %d B for %d rows; %d passes, the largest %d B (bound %d B: %d rows' share, key scratch %d B, 64 KiB)\n",
		whole[0], n, len(passes), largest, bound, block, keyScratch)
	fmt.Fprintf(os.Stderr, "[V2-G2] each pass: %v\n", passes)
	if largest > bound {
		t.Errorf("the largest pass allocated %d B, bound %d B", largest, bound)
	}
}
