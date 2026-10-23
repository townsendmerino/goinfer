package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestMEM5VL_peakTier1 is V1-G4 and V3-G4 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL" gates): a real checkpoint's image
// prefill of about 8K tokens with a 1,024-row image, in one pass (main's shape: V1-G1 and V3-G1 show the passes are byte-equal) and
// in prefillChunkRows passes. Two readings per arm: the live heap after a collection at every pass boundary (the registered one,
// which sees a pass's working set only where the pass still holds it), and heap in use sampled every 20 ms without a collection
// (the transient a pass drops before the boundary, as on Qwen3.5's per-layer buffers). Not timed.
//
//	GOINFER_MEM5VL_PEAK=qwen25vl-3b-instruct go test -count=1 -run '^TestMEM5VL_peakTier1$' -timeout 30m -v ./decoder/
//	GOINFER_MEM5VL_PEAK=qwen3.5-0.8b ...
func TestMEM5VL_peakTier1(t *testing.T) {
	name := os.Getenv("GOINFER_MEM5VL_PEAK")
	if name == "" {
		t.Skip("set GOINFER_MEM5VL_PEAK to a model directory under ~/models")
	}
	home, _ := os.UserHomeDir()
	m, err := Load(filepath.Join(home, "models", name), Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	hidden := m.w.arch.HiddenDim
	const n, img, at = 8192, 1024, 512
	ids := make([]int, n)
	mrope := make([][3]int, n)
	for i := range ids {
		ids[i] = (i*7919 + 13) % vocab
		mrope[i] = [3]int{i, i, i}
	}
	spans := []ImageSpan{{Pos: at, Len: img, Hash: 1}}
	feats := make([]float32, img*hidden)
	for i := range feats {
		feats[i] = float32(i%97)*0.001 - 0.05
	}
	heap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapInuse
	}
	arm := func(label string, C int) (uint64, uint64) {
		prev, prevDone := prefillChunkRows, prefillChunkDone
		defer func() { prefillChunkRows, prefillChunkDone = prev, prevDone }()
		prefillChunkRows = C
		cache := m.NewCache(n + 8)
		base := heap()
		peak := base
		prefillChunkDone = func(int) { peak = max(peak, heap()) }
		done, sampled := make(chan struct{}), make(chan uint64)
		go func() {
			var ms runtime.MemStats
			top := uint64(0)
			tick := time.NewTicker(20 * time.Millisecond)
			defer tick.Stop()
			for {
				runtime.ReadMemStats(&ms)
				top = max(top, ms.HeapInuse)
				select {
				case <-done:
					sampled <- top
					return
				case <-tick.C:
				}
			}
		}()
		_, err := m.prefillLogitsQwenVLSpans(context.Background(), ids, spans, feats, mrope, cache)
		close(done)
		top := <-sampled
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		end := heap()
		runtime.KeepAlive(cache)
		fmt.Fprintf(os.Stderr, "[MEM5-VL peak] %s %s: live heap base %.1f MiB, at a pass boundary up to %.1f MiB (+%.1f), sampled heap in use up to %.1f MiB (+%.1f), after %.1f MiB\n",
			name, label, float64(base)/(1<<20), float64(peak)/(1<<20), float64(peak-base)/(1<<20), float64(top)/(1<<20), float64(int64(top)-int64(base))/(1<<20), float64(end)/(1<<20))
		return peak - base, top - min(top, base)
	}
	oneB, oneS := arm("one pass", 1<<30)
	chB, chS := arm(fmt.Sprintf("passes of %d", prefillChunkRows), prefillChunkRows)
	fmt.Fprintf(os.Stderr, "[MEM5-VL peak] %s: over base, at a pass boundary %.1f -> %.1f MiB; sampled %.1f -> %.1f MiB\n", name,
		float64(oneB)/(1<<20), float64(chB)/(1<<20), float64(oneS)/(1<<20), float64(chS)/(1<<20))
}
