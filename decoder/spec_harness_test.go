package decoder

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// specWorkloads are completion-style prompts with heavy internal repetition — the
// regime n-gram drafting is built for (code edits / RAG / agent loops echo their
// context). The model's greedy continuation tends to reuse earlier structure, so
// the prompt-lookup drafter fires and commits multiple tokens per verify.
var specWorkloads = []struct {
	name   string
	prompt string
}{
	{"codeedit", `package main

type Vec struct {
	X float64
	Y float64
	Z float64
}

func (v Vec) GetX() float64 { return v.X }
func (v Vec) GetY() float64 { return v.Y }
func (v Vec) GetZ() float64 { return v.Z }

func (v *Vec) SetX(x float64) { v.X = x }
func (v *Vec) SetY(y float64) { v.Y = y }
func (v *Vec) Set`},
	{"rag-copy", `Source list:
- alpha = 1
- bravo = 2
- charlie = 3
- delta = 4
- echo = 5

Reproduce the source list exactly:
- alpha = 1
- bravo = 2
- charlie = `},
	{"agent-json", `{"tool":"search","args":{"query":"foo","limit":10}}
{"tool":"search","args":{"query":"bar","limit":10}}
{"tool":"search","args":{"query":"baz","limit":`},
}

// benchGGUFPath returns the registry's path, or the first registered candidate when nothing is
// usable — callers here already handle a load failure, and returning a path that does not resolve
// keeps their error message pointing at a real filename rather than at "".
func benchGGUFPath() string {
	if p, err := lookupAsset("GOINFER_PREQUANT_GGUF"); err == nil {
		return p
	}
	return "../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
}

// TestNgramSpecHarness is the §00-core §7 measurement harness: per workload it
// runs plain greedy and n-gram-speculative greedy, asserts they are token-identical
// (lossless), and reports the machine-independent acceptance metrics (α̅, committed
// tokens/verify) plus this-machine wall-clock speedup. With GOINFER_SPECTRACE_OUT
// set it also dumps the per-position SpecTrace JSONL (the §06 dataset).
//
// Run: go test ./decoder -run TestNgramSpecHarness -v
func TestNgramSpecHarness(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}

	const maxTok = 128
	const K = 8
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	var sink *os.File
	if out := os.Getenv("GOINFER_SPECTRACE_OUT"); out != "" {
		if sink, err = os.Create(out); err != nil {
			t.Fatalf("create trace sink: %v", err)
		}
		defer sink.Close()
		t.Logf("SpecTrace JSONL → %s", out)
	}

	t.Logf("%-11s %-8s %6s  %6s  %6s  %6s  %7s", "workload", "mode", "tok", "α̅/α", "acc", "tok/v", "speedup")
	for _, w := range specWorkloads {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("%s: encode: %v", w.name, err)
		}

		// Plain greedy baseline (timed) — the lossless reference and the speedup denom.
		t0 := time.Now()
		refCh, _ := m.Generate(ctx, prompt, maxTok, greedy)
		ref := collectTokens(refCh)
		plainDur := time.Since(t0)

		// Fixed-K n-gram speculative (timed, traced → the §06 JSONL dataset).
		col := NewTraceCollector(nil)
		if sink != nil {
			col = NewTraceCollector(sink)
			hdr, _ := json.Marshal(map[string]any{"_header": true, "workload": w.name, "model": benchGGUFPath(), "k": K, "max_tok": maxTok})
			sink.Write(append(hdr, '\n'))
		}
		t0 = time.Now()
		fixedCh, gFixed, err := m.genNgram(ctx, prompt, maxTok, &NgramDrafter{}, K, greedy, col.Record, nil)
		if err != nil {
			t.Fatalf("%s fixed: %v", w.name, err)
		}
		fixed := collectTokens(fixedCh)
		fixedDur := time.Since(t0)
		if err := col.Flush(); err != nil {
			t.Fatalf("%s: flush trace: %v", w.name, err)
		}

		// Adaptive-depth n-gram speculative (timed). Same drafter, MaxDraft == K.
		ad := &AdaptiveDepth{MaxDraft: K}
		t0 = time.Now()
		adaCh, gAda, err := m.GenerateNgramSpeculativeAdaptive(ctx, prompt, maxTok, &NgramDrafter{}, ad, greedy)
		if err != nil {
			t.Fatalf("%s adaptive: %v", w.name, err)
		}
		ada := collectTokens(adaCh)
		adaDur := time.Since(t0)

		// Lossless gate: both modes must equal plain greedy exactly.
		if gFixed.Err() != nil || !slices.Equal(fixed, ref) {
			t.Fatalf("%s fixed: speculative != greedy (err %v)\n got %v\n ref %v", w.name, gFixed.Err(), fixed, ref)
		}
		if gAda.Err() != nil || !slices.Equal(ada, ref) {
			t.Fatalf("%s adaptive: speculative != greedy (err %v)\n got %v\n ref %v", w.name, gAda.Err(), ada, ref)
		}

		t.Logf("%-11s %-8s %6d  %6.3f  %6.3f  %6.2f  %6.2fx",
			w.name, "fixed", len(fixed), col.MeanAccept(), gFixed.Spec.AcceptanceRate(), gFixed.Spec.TokensPerRound(), float64(plainDur)/float64(fixedDur))
		t.Logf("%-11s %-8s %6d  %6.3f  %6.3f  %6.2f  %6.2fx",
			w.name, "adaptive", len(ada), ad.Alpha(), gAda.Spec.AcceptanceRate(), gAda.Spec.TokensPerRound(), float64(plainDur)/float64(adaDur))
	}
}
