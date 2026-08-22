//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func loadOptFwdBenchModel(b *testing.B) *decoder.Model {
	b.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		b.Skip("heavy-checkpoint benchmark: set GOINFER_HEAVY_TESTS=1")
	}
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		b.Skipf("model not present: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "cuda"})
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	if !m.ResidentActive() {
		b.Skip("resident backend not active -- optFwd never engages without it")
	}
	return m
}

// BenchmarkOptFwd measures real end-to-end decode with and without the feature at three
// temperatures, on the same prompt and seed — the whole-generation check, not optFwdStep's isolated
// cost. Ported from metal/optfwd_bench_test.go so the two backends' numbers are comparable.
//
// The Metal run (qwen2.5-coder-0.5b, two independent runs) reported ~8-15% faster at T=0.2/T=0.7 and
// ~6-7% SLOWER at T=1.0 — the last being bounded gate-warmup cost, expected rather than a bug, since
// the gate must observe some misses before it can turn itself off.
func BenchmarkOptFwd(b *testing.B) {
	m := loadOptFwdBenchModel(b)
	defer m.Close()
	ctx := context.Background()
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	const n = 60

	run := func(temp float64, disable bool) {
		if disable {
			b.Setenv("GOINFER_NO_OPTFWD", "1")
		}
		sp := decoder.SamplingParams{Temperature: temp, TopP: 0.9, Seed: 99}
		ch, g := m.Generate(ctx, prompt, n, sp)
		for range ch {
		}
		if err := g.Err(); err != nil {
			b.Fatalf("temp=%.1f disable=%v: %v", temp, disable, err)
		}
	}

	cases := []struct {
		name string
		temp float64
	}{
		{"T0.2", 0.2},
		{"T0.7", 0.7},
		{"T1.0", 1.0},
	}
	for _, tc := range cases {
		b.Run(tc.name+"_on", func(b *testing.B) {
			for range b.N {
				run(tc.temp, false)
			}
		})
		b.Run(tc.name+"_off", func(b *testing.B) {
			for range b.N {
				run(tc.temp, true)
			}
		})
	}
}
