package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"testing"
)

// G40-T0a and G40-T0b (docs/tasks/task-cpu-prefill-memory-2026-10.md, "T0a demoted to a diagnostic"): a chunked prefill
// allocates its working set once. T0b, graded: its live heap stays within the prompt's KV, one chunk's working set and
// the attention pool. T0a counts each later chunk's allocations of 4 KiB or more, with its total bytes and allocation
// count beside them. On the fixture it asserts none, which keeps goinfer's own buffers from coming back; on the 0.5B it
// only reports, because aikit's W4A8 scale-buffer pool, emptied at each collection, allocates there.

// bigMallocs is the count of heap allocations of 4 KiB or more so far, from the runtime's allocation size histogram
// (/gc/heap/allocs-by-size:bytes), whose buckets cover every size, large objects included.
func bigMallocs() uint64 {
	sample := []metrics.Sample{{Name: "/gc/heap/allocs-by-size:bytes"}}
	metrics.Read(sample)
	h := sample[0].Value.Float64Histogram()
	var n uint64
	for i, c := range h.Counts {
		if h.Buckets[i] >= 4096 { // the bucket's lower bound
			n += c
		}
	}
	return n
}

// chunkWorkingSet is one pass's activation buffers for C rows, by the arch's geometry (the buffers runLayersFromEmbedNFloor
// takes), in bytes.
func chunkWorkingSet(m *Model, C int) int64 {
	a := m.w.arch
	maxQ, kvDim := a.maxHeads()*a.HeadDim, a.NumKVHeads*a.HeadDim
	return int64(C) * int64(5*a.HiddenDim+2*maxQ+2*kvDim+2*a.IntermediateDim) * 4
}

// chunkAllocs prefills prompt into a fresh cache at chunk size C and returns the bytes allocated by each chunk's pass
// (chunk 0's includes the first-use allocations; the KV reserve is read apart and excluded), and the live heap after a
// forced collection at every boundary, minus the live heap before the prefill.
func chunkAllocs(t *testing.T, m *Model, prompt []int, C int, live bool) (perChunk, bigPerChunk []int64, peakLive int64) {
	t.Helper()
	prevC, prevHook := prefillChunkRows, prefillChunkDone
	defer func() { prefillChunkRows, prefillChunkDone = prevC, prevHook }()
	prefillChunkRows = C
	var mallocs, lastMallocs, big, lastBig uint64
	read := func() (total, heap uint64) {
		if live {
			runtime.GC()
		}
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		mallocs, big = ms.Mallocs, bigMallocs()
		return ms.TotalAlloc, ms.HeapAlloc
	}
	cache := m.NewCache(0)
	t0, base := read()
	last := t0
	lastMallocs, lastBig = mallocs, big
	prefillChunkDone = func(chunk int) {
		total, heap := read()
		if chunk >= 0 {
			perChunk = append(perChunk, int64(total-last))
			bigPerChunk = append(bigPerChunk, int64(big-lastBig))
			n := mallocs - lastMallocs
			fmt.Fprintf(os.Stderr, "[g40-t0a] chunk %d: %d allocations of 4 KiB or more ; %d allocations, %.2f MiB in all (reported)\n",
				chunk+1, big-lastBig, n, float64(total-last)/(1<<20))
		}
		last, lastMallocs, lastBig = total, mallocs, big
		peakLive = max(peakLive, int64(heap)-int64(base))
	}
	if _, err := m.prefillLogits(context.Background(), prompt, cache); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	return perChunk, bigPerChunk, peakLive
}

func checkNoChunkAllocs(t *testing.T, m *Model, prompt []int, C int, graded bool) {
	t.Helper()
	_, big, _ := chunkAllocs(t, m, prompt, C, false)
	want := (len(prompt) + C - 1) / C
	if len(big) != want {
		t.Fatalf("%d chunk readings, want %d: the prefill did not chunk as expected", len(big), want)
	}
	for i, n := range big {
		if i > 0 && n != 0 && graded {
			t.Errorf("chunk %d made %d allocations of 4 KiB or more: a chunk after the first allocated working buffers", i+1, n)
		}
	}
}

func TestG40T0a_noAllocsAfterFirstChunk(t *testing.T) {
	m, err := Load(filepath.Join("..", "testdata", "llama-attnfa-tiny"), Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	checkNoChunkAllocs(t, m, chunkPrompt(1000, vocab, 1), 256, true)
}

func load0p5B(t *testing.T) *Model {
	t.Helper()
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s: %v", p, err)
	}
	m, err := Load(p, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return m
}

// TestG40T0a_noAllocsAfterFirstChunkReal0p5B is a diagnostic: it reports and never fails.
func TestG40T0a_noAllocsAfterFirstChunkReal0p5B(t *testing.T) {
	m := load0p5B(t)
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	checkNoChunkAllocs(t, m, chunkPrompt(7000, vocab, 1), 2048, false)
}

func TestG40T0b_peakLiveHeapReal0p5B(t *testing.T) {
	m := load0p5B(t)
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	const n, C = 7000, 2048
	_, _, peak := chunkAllocs(t, m, chunkPrompt(n, vocab, 1), C, true)
	a := m.w.arch
	kv := int64(n) * int64(a.NumLayers) * 2 * int64(a.NumKVHeads*a.HeadDim) * 4
	bound := kv + chunkWorkingSet(m, C) + prefillAttnScratchBudget + 128<<20
	fmt.Fprintf(os.Stderr, "[g40-t0b] peak live heap over the post-load baseline %.1f MiB; bound %.1f MiB (KV %.1f + one chunk %.1f + attention pool %.0f + margin 128)\n",
		float64(peak)/(1<<20), float64(bound)/(1<<20), float64(kv)/(1<<20), float64(chunkWorkingSet(m, C))/(1<<20), float64(prefillAttnScratchBudget)/(1<<20))
	if peak > bound {
		t.Errorf("peak live heap %d bytes over the baseline, over the %d-byte bound", peak, bound)
	}
}
