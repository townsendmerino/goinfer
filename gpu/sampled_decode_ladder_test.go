//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSampledDecodeLadder measures WebGPU decode tok/s for greedy versus temperature-only sampling with the device
// sampler on and off (GOINFER_NO_SAMPLE_FASTPATH=1, the do-nothing arm), arms interleaved round by round with a
// rotating start, decode rate excluding prefill, paired ratio to greedy within a round. Same method as
// cuda/sampled_decode_ladder_test.go.
func TestSampledDecodeLadder(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint measurement: set GOINFER_HEAVY_TESTS=1")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if v := os.Getenv("GOINFER_GUMBEL_MODELS"); v != "" {
		path = v
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", err)
	}
	tok, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.Encode("The lighthouse keeper had lived alone on the rock for eleven winters, and in that time he "+
		"had learned to read the sea the way other men read faces. On the morning the storm finally broke, "+
		"the water lay flat and pewter-colored under a sky the same shade. He wrote", true)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("not resident")
	}
	const reps, nTok, minTokens = 12, 128, 48
	arms := []struct {
		name  string
		sp    decoder.SamplingParams
		noSmp bool
	}{
		{"greedy", decoder.SamplingParams{}, false},
		{"T1.0 device-sample off", decoder.SamplingParams{Temperature: 1.0, Seed: 7}, true},
		{"T1.0", decoder.SamplingParams{Temperature: 1.0, Seed: 7}, false},
	}
	one := func(sp decoder.SamplingParams, seed int64, noSmp bool) (float64, int) {
		if noSmp {
			t.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "1")
		} else {
			t.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "")
		}
		if sp.Temperature > 0 {
			sp.Seed = seed
		}
		ch, g := m.Generate(context.Background(), prompt, nTok, sp)
		var first, last time.Time
		n := 0
		for range ch {
			now := time.Now()
			if n == 0 {
				first = now
			}
			last = now
			n++
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		if n < 2 {
			return 0, n
		}
		return float64(n-1) / last.Sub(first).Seconds(), n
	}
	for _, a := range arms { // warm-up, discarded
		one(a.sp, 1, a.noSmp)
	}
	rates := make([][]float64, len(arms))
	ratios := make([][]float64, len(arms))
	for r := 0; r < reps; r++ {
		round := make([]float64, len(arms))
		ok := make([]bool, len(arms))
		for k := range arms {
			i := (k + r) % len(arms)
			rate, n := one(arms[i].sp, int64(100+r), arms[i].noSmp)
			if n < minTokens {
				continue
			}
			round[i], ok[i] = rate, true
			rates[i] = append(rates[i], rate)
		}
		if ok[0] {
			for i := 1; i < len(arms); i++ {
				if ok[i] {
					ratios[i] = append(ratios[i], round[i]/round[0])
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[webgpu ladder] round %d/%d\n", r+1, reps)
	}
	med := func(x []float64) float64 {
		y := append([]float64(nil), x...)
		sort.Float64s(y)
		if len(y) == 0 {
			return 0
		}
		return y[len(y)/2]
	}
	for i, a := range arms {
		t.Logf("%-24s n=%2d median %.1f tok/s", a.name, len(rates[i]), med(rates[i]))
	}
	for i := 1; i < len(arms); i++ {
		t.Logf("PAIRED %-24s / greedy: n=%2d median %.3f", arms[i].name, len(ratios[i]), med(ratios[i]))
	}
}
