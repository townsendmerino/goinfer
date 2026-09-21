//go:build cuda && goinfer_testhooks

package cuda

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

// TestSpecNonCopyLane measures what n-gram speculation and the flash-decode lane are worth as the traffic stops being copy-heavy, and records the
// acceptance that explains it. One process, one resident; the arms are switched at runtime (faSplit 0 = exact tree, 16 = lane, with the multi-row verify lane on):
//
//	P0 plain exact   PL plain lane   S0 exact + adaptive n-gram spec   SL lane + adaptive n-gram spec (multi-row verify)
//
// over a ~4000-token document followed by three instructions: COPY (rewrite it exactly — the best case for prompt-lookup drafting), SUMMARIZE (partial reuse of
// its wording) and FRESH (ignore it; write something new — no reuse). Decode time = time(N tokens) - time(1 token) so the shared prefill is subtracted; arms are
// interleaved across 3 rounds and the median is reported. The speculative arms must equal their plain counterparts token for token (lossless on their tree), which
// this asserts. It is a measurement, not a gate: the numbers are printed.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestSpecNonCopyLane -v -timeout 1h
func TestSpecNonCopyLane(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 6144})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || rf.faSplit != 16 || !rf.faVerify {
		t.Skip("resident/lane/multi-row verify not available")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	doc, err := os.ReadFile("../docs/benchmarks.md")
	if err != nil {
		t.Fatalf("doc: %v", err)
	}
	text := string(doc)[:14800]
	// Each kind is (name, question placed BEFORE the document, closing line). The FRESH kinds ask a question the document cannot answer and tell the model to
	// answer it; a first attempt that put "ignore the text above" AFTER the document was ignored by the 1.5B, which just kept copying the document (identical
	// acceptance to COPY), so the printed output head is checked for novelty every run.
	kinds := []struct{ name, pre, ask string }{
		{"COPY", "", "Rewrite the text above EXACTLY, character for character, with no changes and no commentary."},
		{"SUMMARIZE", "", "Summarize the key points of the text above in one short paragraph."},
		{"FRESH-essay", "Question: Explain in detail how photosynthesis works in plants, step by step.\n\nThe document below is unrelated background noise; do not use or quote it.\n\n", "Now answer the question at the top of this message. Do not repeat the document."},
		{"FRESH-story", "Task: Write a short story about a lighthouse keeper who finds a message in a bottle.\n\nThe document below is unrelated background noise; do not use or quote it.\n\n", "Now write the story. Do not repeat the document."},
	}
	const nNew = 160
	greedy := decoder.SamplingParams{}
	type arm struct {
		name  string
		split int
		spec  bool
	}
	arms := []arm{{"P0", 0, false}, {"PL", 16, false}, {"S0", 0, true}, {"SL", 16, true}}
	for _, kd := range kinds {
		prompt, err := decoder.EncodeChatForTest(tk, kd.pre+text+"\n\n"+kd.ask)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		type res struct {
			ms           float64
			toks         []int
			rounds, emit int
			accepted, dr int
		}
		run := func(a arm, n int) res {
			rf.faSplit = a.split
			t0 := time.Now()
			var out []int
			var r res
			if a.spec {
				ch, g, e := m.GenerateNgramSpeculativeAdaptive(context.Background(), prompt, n, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, greedy)
				if e != nil {
					t.Fatalf("%s: %v", a.name, e)
				}
				for id := range ch {
					out = append(out, id)
				}
				if g.Err() != nil {
					t.Fatalf("%s: %v", a.name, g.Err())
				}
				if g.Spec != nil {
					r.rounds, r.emit, r.accepted, r.dr = g.Spec.Rounds, g.Spec.Emitted, g.Spec.Accepted, g.Spec.Drafted
				}
			} else {
				ch, g := m.Generate(context.Background(), prompt, n, greedy)
				for id := range ch {
					out = append(out, id)
				}
				if g.Err() != nil {
					t.Fatalf("%s: %v", a.name, g.Err())
				}
			}
			r.ms = float64(time.Since(t0).Microseconds()) / 1000
			r.toks = out
			return r
		}
		// prefill-only cost per arm (1 token), so decode time excludes it
		pre := map[string]float64{}
		for _, a := range arms {
			run(a, 1)
			var xs []float64
			for i := 0; i < 3; i++ {
				xs = append(xs, run(a, 1).ms)
			}
			sort.Float64s(xs)
			pre[a.name] = xs[1]
		}
		times := map[string][]float64{}
		last := map[string]res{}
		for round := 0; round < 3; round++ {
			for k := range arms {
				a := arms[(k+round)%len(arms)]
				r := run(a, nNew)
				times[a.name] = append(times[a.name], r.ms-pre[a.name])
				last[a.name] = r
			}
		}
		med := func(n string) float64 { x := append([]float64(nil), times[n]...); sort.Float64s(x); return x[len(x)/2] }
		tps := func(n string) float64 { return float64(len(last[n].toks)-1) / (med(n) / 1000) }
		// lossless on their own tree
		eq := func(a, b []int) bool {
			if len(a) != len(b) {
				return false
			}
			for i := range a {
				if a[i] != b[i] {
					return false
				}
			}
			return true
		}
		if !eq(last["S0"].toks, last["P0"].toks) {
			t.Errorf("%s: exact+spec != plain exact", kd.name)
		}
		if !eq(last["SL"].toks, last["PL"].toks) {
			t.Errorf("%s: lane+spec != plain lane", kd.name)
		}
		acc := func(n string) string {
			r := last[n]
			if r.rounds == 0 {
				return "-"
			}
			return fmt.Sprintf("%.2f tok/round, %.0f%% of drafts accepted", float64(r.emit)/float64(r.rounds), 100*float64(r.accepted)/float64(max(r.dr, 1)))
		}
		if txt, e := tk.Decode(last["P0"].toks); e == nil {
			fmt.Printf("[noncopy] %-9s output head: %q\n", kd.name, txt[:min(len(txt), 110)])
		}
		fmt.Printf("[noncopy] %-9s prompt %d tok | decode tok/s: P0 %.1f  PL %.1f  S0 %.1f  SL %.1f | SL/S0 %.3f  S0/P0 %.3f  SL/PL %.3f | S0: %s | SL: %s\n",
			kd.name, len(prompt), tps("P0"), tps("PL"), tps("S0"), tps("SL"), tps("SL")/tps("S0"), tps("S0")/tps("P0"), tps("SL")/tps("PL"), acc("S0"), acc("SL"))
	}
}
