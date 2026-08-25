//go:build cuda && goinfer_testhooks

// Does defaultVerifyWidth = 8 hold, or is 7 better?
//
// The ship-gate run (docs/measurements/adaptive-width-shipgates-2026-08-25.md) found static7
// beating static8 by +7.1% on code and +5.1% on math -- a free win for every `--drafter` user
// from a one-character change. It was NOT acted on, because that evidence was two prompts per
// suite, one session, one target quant. This sweep is what clearing that bar looks like on the
// box that has a viable pairing:
//
//   - more prompts per suite (6 code / 6 math, not 2)
//   - REPEATS, so within-condition spread is visible and the 5-7% claim can be read against it
//   - BOTH target quants. This is the substantive addition, not padding: optimal width is set
//     by the ratio between a plain decode step and a batched verify, and changing the target's
//     quantization moves exactly that ratio. If 7 wins at int4 and 8 wins at int8, the default
//     is quant-dependent and neither constant is right.
//
// What it still is NOT: a second PAIRING. This box has one viable one (qwen3-4b dense + DFlash);
// the other drafters are absent and their targets are MoE, where batched verify touches ~8x the
// expert weight. The cross-pairing cell belongs on the Mac -- docs/prompts/mac-default-verify-width.md.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ \
//	  -run TestDefaultVerifyWidth -v -timeout 4h
package cuda

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

var widthSweepSuites = map[string][]string{
	"code": {
		"Write a Python function that returns the nth Fibonacci number.",
		"Write a Go function that reverses a slice of ints in place.",
		"Write a SQL query that selects the top 5 customers by total order value.",
		"Write a Python function that checks whether a string is a palindrome.",
		"Write a Go function that merges two sorted int slices.",
		"Write a Python class with a method that memoizes its results.",
	},
	"math": {
		"What is 17 * 23? Show your working.",
		"A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
		"What is 15% of 240? Show your working.",
		"Solve for x: 3x + 7 = 25. Show each step.",
		"What is the sum of the first 20 positive integers? Show your working.",
		"A rectangle is 12 cm by 7 cm. What are its area and perimeter?",
	},
}

func stddev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := 0.0
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	v := 0.0
	for _, x := range xs {
		v += (x - m) * (x - m)
	}
	return math.Sqrt(v / float64(len(xs)-1))
}

func TestDefaultVerifyWidth_sweep(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()

	widths := []int{5, 6, 7, 8, 9, 10}
	const repeats = 2
	maxNew := 96

	type cell struct{ quant, suite string }
	best := map[cell]string{}

	for _, quant := range []string{"int4", "int8"} {
		mc, e := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: quant})
		if e != nil {
			t.Logf("quant %s: load failed (%v) — skipping this arm", quant, e)
			continue
		}
		if !mc.BlockSpecCapable() {
			t.Logf("quant %s: not BlockSpecCapable — skipping", quant)
			mc.Close()
			continue
		}
		tk, e := decoder.LoadTokenizerForTest(tgt)
		if e != nil {
			mc.Close()
			t.Skipf("tokenizer: %v", e)
		}
		spec, e := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
		if e != nil {
			mc.Close()
			t.Fatalf("NewBlockSpec(%s): %v", quant, e)
		}

		suites := make([]string, 0, len(widthSweepSuites))
		for k := range widthSweepSuites {
			suites = append(suites, k)
		}
		sort.Strings(suites)

		for _, suite := range suites {
			// tok/s per width, pooled over prompts x repeats
			byWidth := map[int][]float64{}
			for _, ptext := range widthSweepSuites[suite] {
				prompt, e2 := decoder.EncodeChatForTest(tk, ptext)
				if e2 != nil {
					t.Fatalf("encode: %v", e2)
				}
				for r := 0; r < repeats; r++ {
					// INTERLEAVED over width within a prompt+repeat, so drift and thermal
					// state hit every width alike instead of accumulating down one column.
					for _, w := range widths {
						t0 := time.Now()
						got, _, e3 := spec.Generate(prompt, decoder.BlockSpecOptions{
							MaxTokens: maxNew, VerifyWidth: w,
						})
						d := time.Since(t0)
						if e3 != nil {
							t.Fatalf("generate w=%d: %v", w, e3)
						}
						byWidth[w] = append(byWidth[w], float64(len(got))/d.Seconds())
					}
				}
			}
			t.Logf("=== %s / %s (n=%d per width) ===", quant, suite, len(byWidth[widths[0]]))
			bw, brate := 0, 0.0
			for _, w := range widths {
				m := mean(byWidth[w])
				sd := stddev(byWidth[w])
				t.Logf("   width %-3d %7.1f tok/s  (sd %.1f)", w, m, sd)
				if m > brate {
					bw, brate = w, m
				}
			}
			// PAIRED, not pooled. The sd above is dominated by prompt-to-prompt variance —
			// prompts differ in length and content, so pooling makes a real per-prompt effect
			// look like noise. Widths are measured INTERLEAVED within each (prompt, repeat), so
			// observation i of width 7 and observation i of width 8 are the same conditions and
			// can be differenced directly. That removes the between-prompt variance entirely.
			a, b := byWidth[7], byWidth[8]
			n := min(len(a), len(b))
			var diffs []float64
			wins := 0
			for i := 0; i < n; i++ {
				d := (a[i]/b[i] - 1) * 100
				diffs = append(diffs, d)
				if a[i] > b[i] {
					wins++
				}
			}
			t.Logf("   -> best width %d;  7-vs-8 POOLED %+.1f%%  |  PAIRED %+.1f%% (sd %.1f), "+
				"7 wins %d/%d pairs", bw, (mean(a)/mean(b)-1)*100, mean(diffs), stddev(diffs), wins, n)
			best[cell{quant, suite}] = fmt.Sprintf("%d", bw)
		}
		spec = nil
		mc.Close()
	}

	t.Logf("=== best width per (quant, suite) ===")
	for k, v := range best {
		t.Logf("   %-5s %-5s -> %s", k.quant, k.suite, v)
	}
}
