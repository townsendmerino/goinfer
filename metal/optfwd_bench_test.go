//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func loadOptFwdBenchModel(tb testing.TB) *decoder.Model {
	tb.Helper()
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("model not present: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal"})
	if err != nil {
		tb.Fatalf("load: %v", err)
	}
	if !m.ResidentActive() {
		tb.Skip("resident backend not active on this box")
	}
	return m
}

// BenchmarkOptFwd measures real end-to-end tok/s at a few temperatures, with and without the
// feature (GOINFER_NO_OPTFWD), on the same prompt/seed — the whole-generation "election" check
// this session's report method calls for, not just optFwdStep's isolated cost.
func BenchmarkOptFwd(b *testing.B) {
	m := loadOptFwdBenchModel(b)
	ctx := context.Background()
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	const n = 60

	run := func(temp float64, disable bool) {
		if disable {
			os.Setenv("GOINFER_NO_OPTFWD", "1")
			defer os.Unsetenv("GOINFER_NO_OPTFWD")
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
