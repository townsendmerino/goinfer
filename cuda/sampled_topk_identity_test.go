//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSampledTopKStreamIdentity is R7's end-to-end gate (docs/tasks/red-october.md): with a fixed seed,
// the token stream sampled from the device top-K row is IDENTICAL to the stream from the full-row path
// (GOINFER_NO_TOPK_FASTPATH=1), on real checkpoints, for four filter configurations, 1,000 tokens each.
// It also records how many steps the K candidates served and how many fell back to the full row.
//
// The one place the two paths could legitimately differ is top-p: its cutoff compares a cumulative mass
// with topP·Z, and the device's Z differs from the host's chunked f64 sum by rounding, so a draw could
// move if the cumulative landed inside that rounding of the cut (sampler_topk.go). The test is strict
// anyway — any divergence fails — because that event is expected around 1e-7 per token.
//
// Run: GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestSampledTopKStreamIdentity -v -timeout 30m ./cuda/
// Env: GOINFER_TOPK_MODELS (comma-separated paths; default: the 0.5B, gemma3-1b and llama-3.2-1b in ~/models),
// GOINFER_TOPK_TOKENS (1000).
func TestSampledTopKStreamIdentity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint measurement: set GOINFER_HEAVY_TESTS=1")
	}
	models := []string{
		os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		os.ExpandEnv("$HOME/models/gemma3-1b-q4_k_m.gguf"),
		os.ExpandEnv("$HOME/models/llama-3.2-1b-instruct-q4_k_m.gguf"),
	}
	if v := os.Getenv("GOINFER_TOPK_MODELS"); v != "" {
		models = strings.Split(v, ",")
	}
	nTok := ladderEnvInt("GOINFER_TOPK_TOKENS", 1000)

	type cfg struct {
		name string
		sp   decoder.SamplingParams
		topP bool
	}
	cfgs := []cfg{
		{"T0.8+top_p0.95", decoder.SamplingParams{Temperature: 0.8, TopP: 0.95, Seed: 42}, true},
		{"T0.7+top_k40", decoder.SamplingParams{Temperature: 0.7, TopK: 40, Seed: 42}, false},
		{"T1.0+min_p0.05", decoder.SamplingParams{Temperature: 1.0, MinP: 0.05, Seed: 42}, false},
		{"T0.8+top_k40+top_p0.9", decoder.SamplingParams{Temperature: 0.8, TopK: 40, TopP: 0.9, Seed: 42}, true},
	}
	const prose = "The lighthouse keeper had lived alone on the rock for eleven winters, and in that time he " +
		"had learned to read the sea the way other men read faces. On the morning the storm finally broke, " +
		"the water lay flat and pewter-colored under a sky the same shade. He wrote"

	tested := 0
	for _, path := range models {
		if _, err := os.Stat(path); err != nil {
			t.Logf("skip %s: %v", path, err)
			continue
		}
		tok, err := tokenizer.LoadGGUF(path)
		if err != nil {
			t.Logf("skip %s: tokenizer: %v", path, err)
			continue
		}
		prompt, err := tok.Encode(prose, true)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "cuda"})
		if err != nil {
			t.Logf("skip %s: load: %v", path, err)
			continue
		}
		if !m.ResidentActive() {
			m.Close()
			t.Logf("skip %s: not resident", path)
			continue
		}
		run := func(sp decoder.SamplingParams, disable bool) ([]int, *decoder.Generation) {
			if disable {
				decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_TOPK_FASTPATH", "1")
			} else {
				decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_TOPK_FASTPATH", "")
			}
			ch, g := m.Generate(context.Background(), prompt, nTok, sp)
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			return ids, g
		}
		for _, c := range cfgs {
			full, _ := run(c.sp, true)
			fast, g := run(c.sp, false)
			diff, firstDiff := 0, -1
			n := min(len(full), len(fast))
			for i := 0; i < n; i++ {
				if full[i] != fast[i] {
					if firstDiff < 0 {
						firstDiff = i
					}
					diff++
				}
			}
			served, fb := g.TopKServed, g.TopKFallbacks
			t.Logf("%-38s %-24s tokens full=%d fast=%d  served=%d fallbacks=%d (%.2f%%)  stream diverges at %d (%d positions differ after)",
				pathBase(path), c.name, len(full), len(fast), served, fb, pct(fb, served+fb), firstDiff, diff)
			if served+fb == 0 {
				t.Errorf("%s %s: the device top-K path never engaged — the test would pass vacuously", path, c.name)
			}
			if len(full) != len(fast) && firstDiff < 0 {
				t.Errorf("%s %s: stream lengths differ (%d vs %d) with an identical common prefix", path, c.name, len(full), len(fast))
			}
			// Strict: ANY divergence fails. Once one token differs the streams desynchronise, so the count of
			// differing positions after it means nothing; the first divergence index is the datum. The
			// expected rate of a rounding-boundary event under top-p is ~1e-7 per token, so a divergence is
			// far more likely a real bug than that — investigate it (perturb Z as in
			// decoder.TestSampleFromTopK_deviceRoundedZ) before ever relaxing this.
			if firstDiff >= 0 {
				t.Errorf("%s %s: streams diverge at token %d (top-p config: %v)", path, c.name, firstDiff, c.topP)
			}
		}
		m.Close()
		tested++
	}
	if tested == 0 {
		t.Skip("no model available")
	}
}

func pathBase(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}
