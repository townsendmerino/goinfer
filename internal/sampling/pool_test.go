package sampling

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"testing"
)

// TestSamplerScratch_pooled is SCR-G3 (docs/tasks/task-memory-campaign-2026-10.md, MEM12-scratch gates): ten requests, each a
// sampler that observes 64 ids, draws 8 tokens with top-k, top-p, min-p and a repetition penalty, and is released. With the
// collector off, requests 2-10 each allocate at most 5% of request 1's bytes, which built the vocabulary-sized scratch.
func TestSamplerScratch_pooled(t *testing.T) {
	const vocab = 151936
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i) * 0.37))
	}
	ids := make([]int, 64)
	for i := range ids {
		ids[i] = i * 97 % vocab
	}
	if raceEnabled {
		t.Skip("the race detector drops a share of sync.Pool puts on purpose")
	}
	// One P: a sync.Pool caches per P, so a request on another P than the last release's would miss it.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	var bytes [10]uint64
	for r := range bytes {
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		s := NewSampler(SamplingParams{Temperature: 0.8, TopK: 1000, TopP: 0.9, MinP: 0.01, RepeatPenalty: 1.1, Seed: int64(r)})
		s.Observe(ids...)
		for range 8 {
			if _, err := s.Sample(logits); err != nil {
				t.Fatal(err)
			}
		}
		s.Release()
		runtime.ReadMemStats(&b)
		bytes[r] = b.TotalAlloc - a.TotalAlloc
	}
	fmt.Fprintf(os.Stderr, "[SCR-G3] bytes per request: %v\n", bytes)
	for r := 1; r < len(bytes); r++ {
		if bytes[r] > bytes[0]/20 {
			t.Errorf("request %d allocated %d bytes, over 5%% of request 1's %d", r+1, bytes[r], bytes[0])
		}
	}
}
