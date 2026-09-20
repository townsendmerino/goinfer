//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSampledGumbelStreamIdentity is the WebGPU end-to-end gate for the device Gumbel-max sampler (R7b): with a
// fixed seed, the tokens drawn on-device are IDENTICAL to the host's (GOINFER_NO_SAMPLE_FASTPATH=1) on real
// checkpoints. Same PRE-REGISTERED rule as the CUDA test: any divergence fails and is investigated, and the test
// fails if the device path never engaged (DeviceSampled == 0), so it cannot pass vacuously. WGSL's log/mulhi are
// weaker than CUDA's, so a rounding-level divergence is somewhat likelier here (~1e-5 per token at worst); at 2,000
// tokens that is still a low-percent event, so a failure is more likely a bug than rounding.
//
// Run: GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' -run TestSampledGumbelStreamIdentity -v ./gpu/
func TestSampledGumbelStreamIdentity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint measurement: set GOINFER_HEAVY_TESTS=1")
	}
	models := []string{
		os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		os.ExpandEnv("$HOME/models/llama-3.2-1b-instruct-q4_k_m.gguf"),
	}
	if v := os.Getenv("GOINFER_GUMBEL_MODELS"); v != "" {
		models = strings.Split(v, ",")
	}
	nTok := 500
	cfgs := []decoder.SamplingParams{
		{Temperature: 1.0, Seed: 42},
		{Temperature: 0.7, Seed: 43},
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
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
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
				t.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "1")
			} else {
				t.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "")
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
			t.Logf("%s T=%.1f tokens host=%d device=%d device-sampled=%d first divergence %d", path[strings.LastIndex(path, "/")+1:], sp.Temperature, len(host), len(dev), g.DeviceSampled, first)
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
		t.Skip("no resident WebGPU model available")
	}
}
