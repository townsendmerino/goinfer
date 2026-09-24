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

// TestSampledGumbelStreamIdentity is R7b's end-to-end gate (docs/tasks/red-october.md): with a fixed seed, the token
// stream drawn on-device by Gumbel-max is IDENTICAL to the stream the host draws (GOINFER_NO_SAMPLE_FASTPATH=1) —
// the host and the device implement one algorithm over one Philox counter stream, so they may differ only where two
// candidates' scores are within an f32 rounding (measured ~1e-6 per token, and 0 in 15,840 kernel-level draws).
// Real checkpoints, three temperatures, 1,000 tokens each.
//
// PRE-REGISTERED: any divergence fails, and is investigated, not tolerated: at ~1e-6 per token a divergence in ~36k
// tokens is a ~3% event, so one is more likely a bug than rounding. The test also fails if the device path never
// engaged (DeviceSampled == 0), so it cannot pass vacuously.
//
// Run: GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestSampledGumbelStreamIdentity -v -timeout 30m ./cuda/
func TestSampledGumbelStreamIdentity(t *testing.T) {
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
	cfgs := []decoder.SamplingParams{
		{Temperature: 1.0, Seed: 42},
		{Temperature: 0.7, Seed: 43},
		{Temperature: 1.3, Seed: 44},
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
				decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_SAMPLE_FASTPATH", "1")
			} else {
				decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_SAMPLE_FASTPATH", "")
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
		for _, sp := range cfgs {
			host, _ := run(sp, true)
			dev, g := run(sp, false)
			first := -1
			for i := 0; i < min(len(host), len(dev)); i++ {
				if host[i] != dev[i] {
					first = i
					break
				}
			}
			t.Logf("%-38s T=%.1f tokens host=%d device=%d  device-sampled=%d  first divergence %d",
				pathBase(path), sp.Temperature, len(host), len(dev), g.DeviceSampled, first)
			if g.DeviceSampled == 0 {
				t.Errorf("%s T=%.1f: the device sampler never engaged — the test would pass vacuously", path, sp.Temperature)
			}
			if first >= 0 {
				t.Errorf("%s T=%.1f: streams diverge at token %d — investigate before relaxing anything", path, sp.Temperature, first)
			}
			if len(host) != len(dev) && first < 0 {
				t.Errorf("%s T=%.1f: stream lengths differ (%d vs %d) with an identical common prefix", path, sp.Temperature, len(host), len(dev))
			}
		}
		m.Close()
		tested++
	}
	if tested == 0 {
		t.Skip("no model available")
	}
}
