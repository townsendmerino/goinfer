//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSampledDecodeLadder is R7's measurement instrument (docs/tasks/red-october.md): decode tok/s
// for greedy versus sampled configs on ONE binary in ONE session, arms interleaved round by round,
// on a realistic prose prompt. It reports the PAIRED ratio sampled/greedy per round, because pooled
// means carry between-round variance that swamps the effect (CLAUDE.md, measurement discipline).
//
// Decode rate excludes prefill: it is (tokens-1)/(last-first token arrival). A run that stops before
// minTokens (early EOS at temperature 1.0) is dropped and counted, not averaged in.
//
// Run: GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestSampledDecodeLadder -v -timeout 30m ./cuda/
// Env: GOINFER_CUDA_MODEL (default ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf), GOINFER_LADDER_REPS (15),
// GOINFER_LADDER_TOKENS (160).
func TestSampledDecodeLadder(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint measurement: set GOINFER_HEAVY_TESTS=1")
	}
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", err)
	}
	reps := ladderEnvInt("GOINFER_LADDER_REPS", 15)
	nTok := ladderEnvInt("GOINFER_LADDER_TOKENS", 160)
	const minTokens = 64

	tok, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	const prose = "The lighthouse keeper had lived alone on the rock for eleven winters, and in that time he " +
		"had learned to read the sea the way other men read faces. On the morning the storm finally broke, " +
		"the water lay flat and pewter-colored under a sky the same shade, and he climbed the spiral stair " +
		"with his logbook to record what he had seen through the night. He wrote"
	// GOINFER_LADDER_PROMPT_FILE swaps the story for another raw text (e.g. scripts/prompts.json's depth-128
	// filler), because how often the top-K row must fall back to the full logits depends on how flat the
	// next-token distribution is, and that depends on the prompt.
	//
	// The file may hold several prompts separated by a line "===": round r uses prompt r%N for EVERY arm, so
	// pairing is preserved and the per-prompt fallback counts are logged. GOINFER_LADDER_CHAT=1 wraps each
	// in the model's chat template the way `serve` does, instead of feeding raw text.
	texts := []string{prose}
	if f := os.Getenv("GOINFER_LADDER_PROMPT_FILE"); f != "" {
		b, rerr := os.ReadFile(f)
		if rerr != nil {
			t.Fatalf("GOINFER_LADDER_PROMPT_FILE: %v", rerr)
		}
		texts = strings.Split(strings.TrimSpace(string(b)), "\n===\n")
	}
	var prompts [][]int
	for _, text := range texts {
		var ids []int
		if os.Getenv("GOINFER_LADDER_CHAT") != "" {
			ids, err = decoder.EncodeChatForTest(tok, text)
		} else {
			ids, err = tok.Encode(text, false)
		}
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		prompts = append(prompts, ids)
	}
	prompt := prompts[0]

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "cuda"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("resident CUDA backend not active")
	}

	arms := []struct {
		name  string
		sp    decoder.SamplingParams
		noTop bool // GOINFER_NO_TOPK_FASTPATH=1: the device top-K path forced off (the do-nothing arm)
		noSmp bool // GOINFER_NO_SAMPLE_FASTPATH=1: the device Gumbel sampler forced off (the do-nothing arm)
	}{
		{"greedy", decoder.SamplingParams{}, false, false},
		{"T1.0 device-sample off", decoder.SamplingParams{Temperature: 1.0, Seed: 7}, false, true},
		{"T1.0", decoder.SamplingParams{Temperature: 1.0, Seed: 7}, false, false},
		{"T1.3", decoder.SamplingParams{Temperature: 1.3, Seed: 7}, false, false},
		{"T0.8+p0.95 topk-off", decoder.SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 7}, true, false},
		{"T0.8+p0.95", decoder.SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 7}, false, false},
		{"T0.7+k40", decoder.SamplingParams{Temperature: 0.7, TopK: 40, Seed: 7}, false, false},
		{"T1.0+minp0.05", decoder.SamplingParams{Temperature: 1.0, MinP: 0.05, Seed: 7}, false, false},
	}

	var served, fallbacks [16]int
	ppServed := make([][16]int, len(prompts)) // per-prompt copies, to show which prompts force fallbacks
	ppFallbacks := make([][16]int, len(prompts))
	promptIdx := 0
	armIdx := 0
	one := func(sp decoder.SamplingParams, seed int64, noTop, noSmp bool) (float64, int, error) {
		if noSmp {
			decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_SAMPLE_FASTPATH", "1")
		} else {
			decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_SAMPLE_FASTPATH", "")
		}
		if noTop {
			decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_TOPK_FASTPATH", "1")
		} else {
			decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_TOPK_FASTPATH", "")
		}
		if sp.Temperature > 0 {
			sp.Seed = seed
		}
		ch, g := m.Generate(context.Background(), prompts[promptIdx], nTok, sp)
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
			return 0, n, err
		}
		served[armIdx] += g.TopKServed
		fallbacks[armIdx] += g.TopKFallbacks
		ppServed[promptIdx][armIdx] += g.TopKServed
		ppFallbacks[promptIdx][armIdx] += g.TopKFallbacks
		if n < 2 {
			return 0, n, nil
		}
		return float64(n-1) / last.Sub(first).Seconds(), n, nil
	}

	// Warm every arm once (JIT, caches) and discard.
	for _, a := range arms {
		if _, _, err := one(a.sp, 1, a.noTop, a.noSmp); err != nil {
			t.Fatalf("warmup %s: %v", a.name, err)
		}
	}

	served, fallbacks = [16]int{}, [16]int{} // the warm-ups above are not part of the measurement
	for i := range ppServed {
		ppServed[i], ppFallbacks[i] = [16]int{}, [16]int{}
	}
	rates := make([][]float64, len(arms))
	dropped := make([]int, len(arms))
	ratios := make([][]float64, len(arms))
	start := time.Now()
	for r := 0; r < reps; r++ {
		promptIdx = r % len(prompts)
		round := make([]float64, len(arms))
		ok := make([]bool, len(arms))
		for k := range arms {
			i := (k + r) % len(arms) // rotate the starting arm so no arm always runs first
			armIdx = i
			rate, n, err := one(arms[i].sp, int64(100+r), arms[i].noTop, arms[i].noSmp)
			if err != nil {
				t.Fatalf("round %d %s: %v", r, arms[i].name, err)
			}
			if n < minTokens {
				dropped[i]++
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
		fmt.Fprintf(os.Stderr, "[ladder] round %d/%d elapsed %s\n", r+1, reps, time.Since(start).Round(time.Second))
	}

	if len(prompts) > 1 {
		for pi := range prompts {
			for ai, a := range arms {
				if a.sp.TopP > 0 && !a.noTop {
					t.Logf("prompt %d (%d tokens) %-12s top-K served=%d fallbacks=%d", pi, len(prompts[pi]), a.name, ppServed[pi][ai], ppFallbacks[pi][ai])
				}
			}
		}
	}
	t.Logf("model=%s prompt=%d tokens gen<=%d reps=%d (decode tok/s, prefill excluded)", path, len(prompt), nTok, reps)
	for i, a := range arms {
		m, sd := ladderMeanSD(rates[i])
		t.Logf("%-12s n=%2d dropped=%d  median %.1f  mean %.1f  sd %.1f  top-K served=%d fallbacks=%d", a.name, len(rates[i]), dropped[i], ladderMedian(rates[i]), m, sd, served[i], fallbacks[i])
	}
	for i := 1; i < len(arms); i++ {
		m, sd := ladderMeanSD(ratios[i])
		t.Logf("PAIRED %-10s / greedy: n=%2d median %.3f  mean %.3f  sd %.3f", arms[i].name, len(ratios[i]), ladderMedian(ratios[i]), m, sd)
	}
}

func ladderEnvInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func ladderMedian(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	if len(y)%2 == 1 {
		return y[len(y)/2]
	}
	return (y[len(y)/2-1] + y[len(y)/2]) / 2
}

func ladderMeanSD(x []float64) (float64, float64) {
	if len(x) == 0 {
		return 0, 0
	}
	var s float64
	for _, v := range x {
		s += v
	}
	m := s / float64(len(x))
	var v float64
	for _, e := range x {
		v += (e - m) * (e - m)
	}
	if len(x) > 1 {
		v /= float64(len(x) - 1)
	}
	return m, math.Sqrt(v)
}
