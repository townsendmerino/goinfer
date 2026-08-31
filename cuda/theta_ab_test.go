//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestThetaAB is the pre-registered A/B for the measured Theta on the CUDA resident
// path (docs/spec/02, "Theta A/B on the resident path"). Arms: off / fixed-8 /
// adaptive at the shipped 0.5 / adaptive at this backend's measured Theta /
// adaptive at the conservative sensitivity value.
//
// `off` is in the arm set because a speculation suite in this repo was once found
// where no configuration beat running no drafter at all, and that was only visible
// because off was a competitor.
//
// Inputs are REAL repo files read at run time, not the constructed specWorkloads
// corpus (measured at 4-7x the copy density of real code — docs/spec/02). The
// pre-registered loop exclusion is applied before any timing is read.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_THETA_AB=1 \
//	  go test -tags "cuda goinfer_testhooks" ./ -run TestThetaAB -v
func TestThetaAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_THETA_AB") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_THETA_AB=1")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	read := func(p string) string {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Skipf("corpus file missing (%s): %v", p, err)
		}
		return string(b)
	}
	trunc := func(s string, n int) string {
		if len(s) > n {
			return s[:n]
		}
		return s
	}
	adaptive := read("../decoder/spec_adaptive.go")
	sampleSrc := read("../decoder/spec_sample.go")
	const answerSystem = "You are a concise Go expert answering questions about the Go standard library. " +
		"Ground every claim in the search results you are given and cite them as file:line. " +
		"If the results do not contain the answer, say so instead of guessing."
	inputs := []struct{ name, prompt string }{
		{"code-continue", trunc(adaptive, 2600)},
		{"code-continue-2", trunc(sampleSrc, 2600)},
		{"agent-loop-turn2", answerSystem + "\n\nUser: How does the adaptive draft depth controller decide how deep to draft?\n\nSearch results:\n" +
			trunc(adaptive, 1500) + "\n\nAssistant: The controller extends depth while the expected marginal committed token still beats the marginal verify cost.\n\n" +
			"User: And how does the sampled path stay lossless when it rejects a draft token?\n\nSearch results:\n" + trunc(sampleSrc, 1500) + "\n\nAssistant:"},
		{"prose-doc", trunc(read("../docs/spec/00-core.md"), 2600)},
	}

	const maxTok = 160
	const reps = 5
	greedy := decoder.SamplingParams{Temperature: 0}
	ctx := context.Background()

	for _, mc := range []struct {
		file           string
		measured, sens float64
	}{
		{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 0.155, 0.30},
		{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 0.235, 0.30},
	} {
		path := modelPath(mc.file)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skip %s: %v", mc.file, err)
			continue
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load %s: %v", mc.file, err)
		}
		if !m.ResidentActive() {
			m.Close()
			t.Logf("skip %s: not resident (%s)", mc.file, m.ResidentDecline())
			continue
		}
		tk, err := tokenizer.LoadGGUF(path)
		if err != nil {
			m.Close()
			t.Fatalf("tokenizer: %v", err)
		}

		// WARM-UP, discarded. The first cell after a model load carries JIT, allocator
		// and cache effects: between two otherwise-identical runs the `off` control —
		// which no code change here can affect — moved +32.2% on exactly that cell,
		// putting the noise floor above the effect being measured. One throwaway
		// generation before any timed cell removes it.
		if wp, werr := tk.Encode(inputs[0].prompt, true); werr == nil {
			wch, _ := m.Generate(ctx, wp, 32, greedy)
			_ = collectToks(wch)
		}

		for _, in := range inputs {
			prompt, err := tk.Encode(in.prompt, true)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			// Reference + loop guard FIRST, before any timing is read.
			refCh, _ := m.Generate(ctx, prompt, maxTok, greedy)
			ref := collectToks(refCh)
			if tri := distinctTri(ref); tri < 0.70 {
				t.Logf("%-10s %-17s EXCLUDED by the pre-registered loop rule (distinct-trigram %.3f)", short(mc.file), in.name, tri)
				continue
			}

			type arm struct {
				name string
				run  func() ([]int, error)
			}
			arms := []arm{
				{"off", func() ([]int, error) {
					ch, g := m.Generate(ctx, prompt, maxTok, greedy)
					return collectToks(ch), g.Err()
				}},
				{"fixed-8", func() ([]int, error) {
					ch, g, err := m.GenerateNgramSpeculative(ctx, prompt, maxTok, &decoder.NgramDrafter{}, 8, greedy)
					if err != nil {
						return nil, err
					}
					out := collectToks(ch)
					return out, g.Err()
				}},
			}
			for _, th := range []struct {
				label string
				theta float64
			}{{"ada@0.5", 0.5}, {"ada@measured", mc.measured}, {"ada@0.30", mc.sens}} {
				theta := th.theta
				arms = append(arms, arm{th.label, func() ([]int, error) {
					ad := &decoder.AdaptiveDepth{MaxDraft: 8, Theta: theta}
					ch, g, err := m.GenerateNgramSpeculativeAdaptive(ctx, prompt, maxTok, &decoder.NgramDrafter{}, ad, greedy)
					if err != nil {
						return nil, err
					}
					out := collectToks(ch)
					return out, g.Err()
				}})
			}

			// Interleave arm-by-arm across repetitions so drift cannot bias one arm.
			times := make(map[string][]float64, len(arms))
			for r := 0; r < reps; r++ {
				for _, a := range arms {
					t0 := time.Now()
					got, err := a.run()
					el := float64(time.Since(t0).Microseconds()) / 1000
					if err != nil {
						t.Fatalf("%s/%s/%s: %v", mc.file, in.name, a.name, err)
					}
					// Lossless is absolute: a divergent arm fails the run.
					if !slices.Equal(got, ref) {
						t.Fatalf("LOSSLESS VIOLATION %s/%s/%s: token stream differs from off", mc.file, in.name, a.name)
					}
					times[a.name] = append(times[a.name], el)
				}
			}
			base := medianF(times["ada@0.5"])
			offT := medianF(times["off"])
			var b strings.Builder
			fmt.Fprintf(&b, "%-10s %-17s ", short(mc.file), in.name)
			for _, a := range arms {
				md := medianF(times[a.name])
				fmt.Fprintf(&b, "| %s %7.0fms (vs off %4.2fx, vs ada@0.5 %+5.1f%%) ", a.name, md, offT/md, 100*(base/md-1))
			}
			t.Log(b.String())
		}
		m.Close()
	}
}

func short(s string) string {
	return strings.SplitN(strings.TrimPrefix(s, "qwen2.5-coder-"), "-", 2)[0]
}

func collectToks(ch <-chan int) []int {
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	return out
}

func distinctTri(toks []int) float64 {
	if len(toks) < 3 {
		return 1
	}
	seen := map[[3]int]bool{}
	for i := 0; i+3 <= len(toks); i++ {
		seen[[3]int{toks[i], toks[i+1], toks[i+2]}] = true
	}
	return float64(len(seen)) / float64(len(toks)-2)
}
