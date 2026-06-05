package decoder

import (
	"os"
	"sync"
	"testing"
)

// benchModel loads the 0.5B once and caches it — the go test harness calls a
// benchmark function several times (growing b.N), and reloading (a full GGUF
// dequant+requant) each time would swamp the CPU profile with load cost instead
// of decode. Returns "" path → skip.
var (
	benchOnce  sync.Once
	benchModel *Model
	benchErr   error
)

func loadBenchModel() (*Model, error) {
	benchOnce.Do(func() {
		path := os.Getenv("GINFER_PREQUANT_GGUF")
		if path == "" {
			path = "../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
		}
		if _, err := os.Stat(path); err != nil {
			benchErr = err
			return
		}
		benchModel, benchErr = Load(path, Options{Quant: "int8int8"})
	})
	return benchModel, benchErr
}

// BenchmarkDecode times steady-state single-token decode (the chat case:
// batch=1, greedy). It loads the 0.5B at int8int8, prefills a short fixed
// prompt, then runs b.N forward+sample steps in the timed loop — the same
// forward/runLayers + LM head + sampler path Generate drives. Reports tok/s.
//
// It is the perf campaign's regression guard; run with profiles:
//
//	go test ./decoder -run '^$' -bench BenchmarkDecode -benchmem \
//	  -cpuprofile cpu.out -memprofile mem.out -benchtime 5s
//
// Skips cleanly without the model asset (set GINFER_PREQUANT_GGUF or drop the
// gguf in testdata), like the other model-dependent tests.
func BenchmarkDecode(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}

	// A short fixed prompt; greedy so the decode is deterministic.
	prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25}
	cache := m.NewCache(len(prompt) + b.N + 8)
	for _, id := range prompt[:len(prompt)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			b.Fatalf("prefill: %v", err)
		}
	}
	sampler := NewSampler(SamplingParams{Temperature: 0})
	logits, err := m.forward(prompt[len(prompt)-1], cache)
	if err != nil {
		b.Fatalf("seed forward: %v", err)
	}
	next, err := sampler.Sample(logits)
	if err != nil {
		b.Fatalf("seed sample: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logits, err := m.forward(next, cache)
		if err != nil {
			b.Fatalf("forward: %v", err)
		}
		if next, err = sampler.Sample(logits); err != nil {
			b.Fatalf("sample: %v", err)
		}
	}
	b.StopTimer()

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}
