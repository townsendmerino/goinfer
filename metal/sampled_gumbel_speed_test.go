//go:build darwin

package metal

import (
	"context"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSampledDecodeLadder_speed is R7b Mac's speed measurement (docs/tasks/red-october.md),
// following the same protocol docs/measurements/sampled-gumbel-2026-09-20.md's CUDA/WebGPU numbers
// used (paired against greedy, same session, interleaved with a rotating start, a discarded
// warm-up round, and a do-nothing arm) — CUDA's own TestSampledDecodeLadder was not found
// committed anywhere in this tree to port directly (checked: no match repo-wide), so this is a
// from-scratch harness built to the SAME protocol description, not a line-for-line port.
//
// THREE ARMS, one seed, one prompt, per round:
//   - greedy       (Temperature: 0)
//   - host draw    (Temperature: 1.0, GOINFER_NO_SAMPLE_FASTPATH=1) -- the do-nothing arm: proves
//     the device path is worth having at all, not just that it beats itself
//   - device draw  (Temperature: 1.0, fastpath default)
//
// PAIRED, NOT POOLED (CLAUDE.md measurement discipline, rule 7): each round runs all three arms
// back to back before the next round starts, and the ratio is computed PER ROUND, then those
// per-round ratios are summarized (median + spread) -- never a ratio of pooled means, which would
// carry between-round variance (thermal, scheduler noise) into the comparison.
//
// DECODE-ONLY: timed from the first emitted token to the last (time-to-first-token subtracted),
// not from the call start, so prefill cost (paid once, off this measurement's critical path in a
// real decode-bound workload) doesn't dilute the per-token rate.
//
// Run: GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestSampledDecodeLadder_speed -v -timeout 30m
func TestSampledDecodeLadder_speed(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint and runs a real decode ladder)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (bench from ~/models only, never /Volumes)", path)
	}

	tok, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	const prose = "The lighthouse keeper had lived alone on the rock for eleven winters, and in that " +
		"time he had learned to read the sea the way other men read faces. On the morning the storm " +
		"finally broke, the water lay flat and pewter-colored under a sky the same shade, and he stood " +
		"at the gallery rail counting the seconds between swells the way his father had taught him, " +
		"back when the light still ran on oil and a keeper's whole trade was patience. He wrote in the " +
		"log, as he always did, and then he waited, because there was nothing else a man in his " +
		"position could usefully do."
	prompt, err := tok.Encode(prose, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("prompt: %d tokens", len(prompt))

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("resident backend not active on this box")
	}

	const nTok = 200
	const nRounds = 15 // + 1 discarded warm-up round below, matching CUDA/WebGPU's own n=15

	type arm struct {
		name              string
		disable           bool // GOINFER_NO_SAMPLE_FASTPATH
		sp                decoder.SamplingParams
		results           []float64 // decode tok/s, one per non-warm-up round
		deviceSampledSeen bool
	}
	arms := []*arm{
		{name: "greedy", sp: decoder.SamplingParams{Temperature: 0}},
		{name: "host", disable: true, sp: decoder.SamplingParams{Temperature: 1.0, Seed: 7}},
		{name: "device", sp: decoder.SamplingParams{Temperature: 1.0, Seed: 7}},
	}

	run := func(a *arm) (tokPerSec float64, deviceSampled int) {
		t.Helper()
		if a.disable {
			os.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "1")
			defer os.Unsetenv("GOINFER_NO_SAMPLE_FASTPATH")
		} else {
			os.Setenv("GOINFER_NO_SAMPLE_FASTPATH", "")
		}
		ctx := context.Background()
		ch, g := m.Generate(ctx, prompt, nTok, a.sp)
		var tFirst, tLast time.Time
		n := 0
		for range ch {
			now := time.Now()
			if n == 0 {
				tFirst = now
			}
			tLast = now
			n++
		}
		if err := g.Err(); err != nil {
			t.Fatalf("%s: generate: %v", a.name, err)
		}
		if n < 2 {
			t.Fatalf("%s: only %d tokens emitted, can't measure a decode rate", a.name, n)
		}
		elapsed := tLast.Sub(tFirst).Seconds()
		return float64(n-1) / elapsed, g.DeviceSampled
	}

	start := time.Now()
	t.Logf("started %s (PDT/local); %d rounds x 3 arms + 1 warm-up round, ~%d tokens/round",
		start.Format("15:04:05"), nRounds, nTok)

	// Round 0 is the discarded warm-up: every arm's OWN first execution, one round, rotated order
	// same as every other round so the warm-up sees the same interleaving pattern as real data.
	for round := 0; round <= nRounds; round++ {
		order := make([]*arm, len(arms))
		copy(order, arms)
		rot := round % len(arms)
		order = append(order[rot:], order[:rot]...)
		for _, a := range order {
			rate, devSampled := run(a)
			if round == 0 {
				continue // warm-up: discarded
			}
			a.results = append(a.results, rate)
			if devSampled > 0 {
				a.deviceSampledSeen = true
			}
		}
		if round > 0 && round%5 == 0 {
			t.Logf("round %d/%d done, elapsed %s", round, nRounds, time.Since(start).Round(time.Second))
		}
	}
	t.Logf("all rounds done, elapsed %s (started %s)", time.Since(start).Round(time.Second), start.Format("15:04:05"))

	byArm := map[string]*arm{}
	for _, a := range arms {
		byArm[a.name] = a
		med := median(a.results)
		t.Logf("%-8s median %.1f tok/s (n=%d, sd %.2f) device-sampled-ever=%v",
			a.name, med, len(a.results), stddev(a.results), a.deviceSampledSeen)
	}

	if !byArm["device"].deviceSampledSeen {
		t.Error("device arm: the device sampler never engaged in any round — this measurement would be greedy/host-fastpath timing mislabeled as device timing")
	}

	// PAIRED ratios: per round, device/greedy and host/greedy, THEN summarized -- never a ratio of
	// the pooled medians above, which is reported only for human-readability of the raw rates.
	pairedRatio := func(num, den *arm) []float64 {
		n := min(len(num.results), len(den.results))
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			out[i] = num.results[i] / den.results[i]
		}
		return out
	}
	hostRatios := pairedRatio(byArm["host"], byArm["greedy"])
	devRatios := pairedRatio(byArm["device"], byArm["greedy"])
	t.Logf("paired ratio vs greedy: host  median %.3f (sd %.3f, n=%d)", median(hostRatios), stddev(hostRatios), len(hostRatios))
	t.Logf("paired ratio vs greedy: device median %.3f (sd %.3f, n=%d)", median(devRatios), stddev(devRatios), len(devRatios))
	t.Logf("device vs host (device draw's own win over the do-nothing arm): median %.3f",
		median(pairedRatio(byArm["device"], byArm["host"])))

	t.Logf("provenance: model=qwen2.5-coder-0.5b-instruct-q4_k_m.gguf quant=int4 backend=metal prompt_tokens=%d gen_tokens=%d date=%s",
		len(prompt), nTok, time.Now().Format("2006-01-02")) // Go's reference layout, not this year
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func stddev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	var mean float64
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	var ss float64
	for _, x := range xs {
		d := x - mean
		ss += d * d
	}
	return math.Sqrt(ss / float64(len(xs)-1))
}
