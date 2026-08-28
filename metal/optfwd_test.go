//go:build darwin

package metal

import (
	"context"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func loadOptFwdModel(t *testing.T) *decoder.Model {
	t.Helper()
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !m.ResidentActive() {
		t.Skip("resident backend not active on this box -- optFwd never engages without it")
	}
	return m
}

// TestOptFwd_bitIdenticalStream is THE gate for this feature: the optimistic-forward overlap
// changes ONLY scheduling, never what gets sampled. With the SAME seed, the token stream (and the
// reported logprobs) must be identical whether the feature is on (default) or forced off via
// GOINFER_NO_OPTFWD -- regardless of how many steps hit or miss the argmax guess internally.
func TestOptFwd_bitIdenticalStream(t *testing.T) {
	m := loadOptFwdModel(t)
	ctx := context.Background()
	prompt := []int{1, 7, 42, 100, 5}
	const n = 40

	run := func(disable bool) ([]int, []decoder.SampleInfo, *decoder.Generation) {
		if disable {
			os.Setenv("GOINFER_NO_OPTFWD", "1")
			defer os.Unsetenv("GOINFER_NO_OPTFWD")
		}
		// The shipped optFwd cap is 0.2 (decoder/spec_optfwd.go): above it the overlap is a measured
		// loss and does not run. This test EXERCISES the overlap, so it must raise the cap — otherwise
		// it passes with the feature switched off, which is a pass that proves nothing.
		t.Setenv("GOINFER_OPTFWD_MAX_TEMP", "2.0")
		sp := decoder.SamplingParams{Temperature: 0.7, TopP: 0.9, Seed: 1234, Logprobs: true}
		ch, g := m.Generate(ctx, prompt, n, sp)
		var toks []int
		for id := range ch {
			toks = append(toks, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("disable=%v: stream err %v", disable, err)
		}
		return toks, g.Logprobs, g
	}

	onToks, onLP, onGen := run(false)
	offToks, offLP, _ := run(true)

	if !slices.Equal(onToks, offToks) {
		t.Fatalf("optFwd changed the emitted token stream\n  on:  %v\n  off: %v", onToks, offToks)
	}
	if len(onLP) != len(offLP) {
		t.Fatalf("logprob count differs: on=%d off=%d", len(onLP), len(offLP))
	}
	for i := range onLP {
		if onLP[i].ID != offLP[i].ID || onLP[i].Logprob != offLP[i].Logprob {
			t.Fatalf("logprob[%d] differs: on=%+v off=%+v", i, onLP[i], offLP[i])
		}
	}

	if onGen.OptFwd == nil {
		t.Fatal("OptFwd stats missing on the enabled run -- optFwdEligible should have held for this config")
	}
	t.Logf("stream identical (%d tokens); OptFwd: guessed=%d hit=%d rate=%.1f%%",
		len(onToks), onGen.OptFwd.Guessed, onGen.OptFwd.Hit, 100*onGen.OptFwd.HitRate())
	if onGen.OptFwd.Guessed == 0 {
		t.Error("optFwd never attempted a guess over 40 tokens at T=0.7 -- feature did not engage at all")
	}
}

// TestOptFwd_lowTempHighHitRate is a coarse sanity check that the real measured shape (hit rate
// rising as temperature drops) holds through this exact code path, not just the standalone
// sampler-vs-argmax check from the design phase.
func TestOptFwd_lowTempHighHitRate(t *testing.T) {
	// Sweeps temperatures ABOVE the shipped 0.2 cap by design — measuring how the hit rate falls
	// is the point — so it must raise the cap or the high-T arms would report zero guesses.
	t.Setenv("GOINFER_OPTFWD_MAX_TEMP", "2.0")
	m := loadOptFwdModel(t)
	ctx := context.Background()
	prompt := []int{1, 7, 42}
	const n = 100

	rate := func(temp float64) float64 {
		sp := decoder.SamplingParams{Temperature: temp, TopP: 0.9, Seed: 7}
		ch, g := m.Generate(ctx, prompt, n, sp)
		for range ch {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("T=%.1f: stream err %v", temp, err)
		}
		if g.OptFwd == nil || g.OptFwd.Guessed == 0 {
			t.Fatalf("T=%.1f: optFwd did not engage", temp)
		}
		return g.OptFwd.HitRate()
	}

	low := rate(0.2)
	high := rate(1.0)
	t.Logf("hit rate: T=0.2 -> %.1f%%, T=1.0 -> %.1f%%", 100*low, 100*high)
	if low <= high {
		t.Errorf("expected T=0.2's hit rate (%.1f%%) to exceed T=1.0's (%.1f%%), matching the design-phase measurement", 100*low, 100*high)
	}
}
