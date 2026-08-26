//go:build cuda && goinfer_testhooks

// Per-suite drafter-vs-OFF comparison, including the mixed-content suite.
//
// Born as the adaptive-width ship-gates; the adaptive arm and its gates came out when Phase 2's
// premise died and took the controller with it. What remains is the part that earned its keep:
// every static width scored against running NO drafter at all, per traffic class. Runs the REAL BlockSpec path on a CUDA-resident target, not a reimplemented loop:
// cuda/drafter_loop_test.go's dflashLoop is a standalone copy of the round loop and would
// measure a controller that is not in it.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ \
//	  -run TestDrafterVsOff -v -timeout 4h
package cuda

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// adaptiveSuites: the three existing traffic classes plus MIXED, which is new and is the case
// the whole idea exists for.
//
// WHY MIXED HAD TO BE BUILT. code/math/chat each sit in ONE regime for a whole generation, so
// a static width can be optimal for the entire run and adaptivity has nothing to win. The
// claimed advantage lives at a prose->structured BOUNDARY, where no single static value is
// right for both halves. Every prompt here forces the transition mid-generation: explain in
// prose first, THEN emit something structured.
var adaptiveSuites = map[string][]string{
	"code": {
		"Write a Python function that returns the nth Fibonacci number.",
		"Write a Go function that reverses a slice of ints in place.",
	},
	"math": {
		"What is 17 * 23? Show your working.",
		"A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
	},
	"chat": {
		"Explain what a hash table is, in two sentences.",
		"Give me three tips for keeping houseplants alive.",
	},
	"mixed": {
		"Explain in two sentences why hash tables are fast, then output a JSON object with keys \"name\" and \"average_complexity\".",
		"Briefly describe what binary search does, in prose, then write the Python function that implements it.",
		"In one sentence, say what a CSV file is. Then output exactly five rows of CSV with columns id,name,score.",
	},
}

// structuralMarkers locate where a mixed prompt's output stops being prose. Used only to find
// the TRANSITION ROUND for the windowed analysis -- never to score anything.
var structuralMarkers = []string{"{", "```", "def ", "id,name", "[\n", "- "}

type roundRec struct{ width, committed int }

type armResult struct {
	label     string
	tokens    int
	dur       time.Duration
	rounds    []roundRec
	ids       []int
	tokPerSec float64
}

// transitionRound maps the first structural marker in the decoded text back to the round that
// emitted it, by walking the per-round committed counts.
func transitionRound(t *testing.T, tk *tokenizer.Tokenizer, ids []int, rounds []roundRec) (int, bool) {
	t.Helper()
	text, err := tk.Decode(ids)
	if err != nil {
		return 0, false
	}
	best := -1
	for _, mk := range structuralMarkers {
		if i := strings.Index(text, mk); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	if best <= 0 {
		return 0, false
	}
	// Character offset -> token index: decode prefixes until the marker is covered.
	tokIdx := len(ids)
	for n := 1; n <= len(ids); n++ {
		if p, e := tk.Decode(ids[:n]); e == nil && len(p) > best {
			tokIdx = n
			break
		}
	}
	acc := 0
	for r, rr := range rounds {
		acc += rr.committed
		if acc >= tokIdx {
			return r, true
		}
	}
	return 0, false
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func TestDrafterVsOff_perSuite(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	if !mc.BlockSpecCapable() {
		t.Fatal("BlockSpecCapable() = false on a cuda resident")
	}
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()
	tk, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	spec, err := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
	if err != nil {
		t.Fatalf("NewBlockSpec: %v", err)
	}
	maxNew := 96
	blockSize := dr.BlockSize()

	// THE COMPARISON SET INCLUDES OFF. Adaptive beating every static width proves nothing on a
	// suite where running no drafter at all is faster -- which is precisely the guard's answer
	// for chat (it disables, measuring 0.92x). Without this column the chat cell cannot answer
	// the only question that matters there.
	staticWidths := []int{2, 3, 4, 5, 6, 7, 8, 10, 12, 16}

	suiteNames := make([]string, 0, len(adaptiveSuites))
	for k := range adaptiveSuites {
		suiteNames = append(suiteNames, k)
	}
	sort.Strings(suiteNames)

	for _, suite := range suiteNames {
		t.Run(suite, func(t *testing.T) {
			var offTokPerSec []float64
			results := map[string][]float64{}
			var mixedTraces []armResult

			for _, promptText := range adaptiveSuites[suite] {
				prompt, e := decoder.EncodeChatForTest(tk, promptText)
				if e != nil {
					t.Fatalf("encode: %v", e)
				}

				// --- arm: OFF (plain resident greedy, no drafter at all) ---
				t0 := time.Now()
				n := 0
				ch, _ := mc.Generate(context.Background(), prompt, maxNew, decoder.SamplingParams{})
				for range ch {
					n++
				}
				offDur := time.Since(t0)
				offRate := float64(n) / offDur.Seconds()
				offTokPerSec = append(offTokPerSec, offRate)

				run := func(label string, opt decoder.BlockSpecOptions) armResult {
					var rec []roundRec
					opt.MaxTokens = maxNew
					opt.OnRound = func(w, c int) { rec = append(rec, roundRec{w, c}) }
					tt := time.Now()
					got, _, e := spec.Generate(prompt, opt)
					d := time.Since(tt)
					if e != nil {
						t.Fatalf("%s: %v", label, e)
					}
					return armResult{label: label, tokens: len(got), dur: d, rounds: rec, ids: got,
						tokPerSec: float64(len(got)) / d.Seconds()}
				}

				for _, w := range staticWidths {
					if w > blockSize {
						continue
					}
					r := run(fmt.Sprintf("static%d", w), decoder.BlockSpecOptions{VerifyWidth: w})
					results[r.label] = append(results[r.label], r.tokPerSec/offRate)
				}
				// The mixed suite's per-round trace is kept: it is how Finding 2 was found
				// and how the next drafter change gets checked against it. Recorded at the
				// widest arm, since that is where a transition shows most.
				if suite == "mixed" {
					tr := run("trace", decoder.BlockSpecOptions{VerifyWidth: blockSize})
					tr.label = promptText
					mixedTraces = append(mixedTraces, tr)
				}
			}

			t.Logf("=== suite %s (ratios vs OFF = plain greedy, no drafter; off mean %.1f tok/s) ===",
				suite, mean(offTokPerSec))
			type row struct {
				label string
				ratio float64
			}
			var rows []row
			for k, v := range results {
				rows = append(rows, row{k, mean(v)})
			}
			sort.Slice(rows, func(i, j int) bool { return rows[i].ratio > rows[j].ratio })
			bestStatic, bestStaticLabel := 0.0, ""
			for _, r := range rows {
				t.Logf("   %-10s %.3fx", r.label, r.ratio)
				if r.label != "adaptive" && r.ratio > bestStatic {
					bestStatic, bestStaticLabel = r.ratio, r.label
				}
			}
			// THE SURVIVING GATE: does the drafter beat NO drafter at all on this suite?
			// The adaptive arm and its two gates were removed with the controller (Phase 2's
			// premise died, so nothing was left to reuse it) — but `off` as a competitor is
			// the part that earned its keep, and it stays.
			if bestStatic < 1.0 {
				t.Logf("NOTE: no width beats OFF on %s (best %s %.3fx) — block drafting does not "+
					"pay on this traffic class with this pairing", suite, bestStaticLabel, bestStatic)
			}

			// THE WINDOW. A cumulative average weights round N at 1/N, so at a mid-generation
			// regime change the signal STRUCTURALLY lags — and end-to-end numbers hide it.
			for _, tr := range mixedTraces {
				tRound, ok := transitionRound(t, tk, tr.ids, tr.rounds)
				if !ok {
					t.Logf("   [window] no structural marker found for %.40q — window not measured", tr.label)
					continue
				}
				const K = 6
				var preW, postW, postC []float64
				for i, rr := range tr.rounds {
					switch {
					case i < tRound:
						preW = append(preW, float64(rr.width))
					case i < tRound+K:
						postW = append(postW, float64(rr.width))
						postC = append(postC, float64(rr.committed))
					}
				}
				t.Logf("   [window] %.40q transition@round %d/%d: width pre %.1f -> post %.1f, committed post %.2f",
					tr.label, tRound, len(tr.rounds), mean(preW), mean(postW), mean(postC))
				if len(postC) > 0 && mean(postW) < mean(postC) {
					t.Logf("      LAG: for %d rounds after the transition the controller drafted %.1f "+
						"while the round was committing %.2f — the cumulative signal is behind the regime",
						len(postW), mean(postW), mean(postC))
				}
			}
		})
	}
}
