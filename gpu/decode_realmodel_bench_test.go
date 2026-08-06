//go:build gpu

package gpu

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDecodeRealModel_throughput measures STEADY-STATE decode tok/s of a real GGUF
// model loaded GPU-resident (int8), on actual weights — the bench the synthetic
// TestDecodeToken_throughput can't be: it exercises whatever glue the model's
// architecture actually dispatches (q/k/v bias for Qwen2.5, qk-norm for Qwen3, …),
// so a fusion that only fires for those families can be measured here.
//
// Default model is the Qwen2.5-coder-1.5B GGUF (has qkv bias → the target for the
// Increment-2 bias→GEMV-epilogue fusion). Override with GOINFER_DECODE_GGUF.
//
//	go test -tags gpu -run TestDecodeRealModel_throughput -v
//
// Rate is the inter-token rate (total − TTFT)/(n−1), so prefill/first-token pipeline
// compilation is excluded; reported as the BEST of several rounds (the steady-state
// floor, least perturbed by scheduler/thermal noise).
func TestDecodeRealModel_throughput(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("real-model decode bench")
	}
	path := os.Getenv("GOINFER_DECODE_GGUF")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s (set GOINFER_DECODE_GGUF)", path)
	}
	newOrSkipHW(t).Close() // real-HW gate

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	promptIDs, _ := tk.Encode("Write a short note about recursion in programming.\n", true)

	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skipf("model did not go GPU-resident (too big / ineligible) — bench needs the resident decode path")
	}

	greedy := decoder.SamplingParams{Temperature: 0}
	// inter-token rate of one Generate call (excludes TTFT/prefill).
	rate := func(maxTokens int) (tps float64, n int) {
		ch, _ := m.Generate(context.Background(), promptIDs, maxTokens, greedy)
		if ch == nil {
			t.Fatal("nil generate channel")
		}
		var ttft, total time.Duration
		t0 := time.Now()
		for range ch {
			if n == 0 {
				ttft = time.Since(t0)
			}
			n++
		}
		total = time.Since(t0)
		if n < 2 {
			t.Fatalf("model produced %d tokens (need ≥2 for an inter-token rate)", n)
		}
		return float64(n-1) / (total - ttft).Seconds(), n
	}

	rate(8) // warm: compile pipelines, fill the cache; discarded
	const rounds, gen = 6, 48
	best := 0.0
	for range rounds {
		if tps, _ := rate(gen); tps > best {
			best = tps
		}
	}
	t.Logf("REAL-MODEL DECODE: %s", path)
	t.Logf("  steady-state %.1f tok/s = %.2f ms/token (best of %d × %d-tok greedy, resident int8)",
		best, 1000/best, rounds, gen)
}
