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

func benchGGUFPath() string {
	if p := os.Getenv("GINFER_PREQUANT_GGUF"); p != "" {
		return p
	}
	return "../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
}

// TestNgramSpecHarness is the §00-core §7 measurement harness: per workload it
// runs plain greedy and n-gram-speculative greedy, asserts they are token-identical
// (lossless), and reports the machine-independent acceptance metrics (α̅, committed
// tokens/verify) plus this-machine wall-clock speedup. With GINFER_SPECTRACE_OUT
// set it also dumps the per-position SpecTrace JSONL (the §06 dataset).
//
// Run: go test ./decoder -run TestNgramSpecHarness -v
func TestNgramSpecHarness(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
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
	if out := os.Getenv("GINFER_SPECTRACE_OUT"); out != "" {
		if sink, err = os.Create(out); err != nil {
			t.Fatalf("create trace sink: %v", err)
		}
		defer sink.Close()
		t.Logf("SpecTrace JSONL → %s", out)
	}

	t.Logf("%-11s %6s  %6s  %6s  %6s  %8s %8s %7s", "workload", "tok", "α̅", "acc", "tok/v", "plain/s", "spec/s", "speedup")
	for _, w := range specWorkloads {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("%s: encode: %v", w.name, err)
		}

		// Plain greedy baseline (timed).
		t0 := time.Now()
		refCh, _ := m.Generate(ctx, prompt, maxTok, greedy)
		ref := collectTokens(refCh)
		plainDur := time.Since(t0)

		// n-gram speculative greedy (timed, traced).
		col := NewTraceCollector(nil)
		if sink != nil {
			col = NewTraceCollector(sink)
			hdr, _ := json.Marshal(map[string]any{"_header": true, "workload": w.name, "model": benchGGUFPath(), "k": K, "max_tok": maxTok})
			sink.Write(append(hdr, '\n'))
		}
		drafter := &NgramDrafter{}
		t0 = time.Now()
		ch, g, err := m.genNgram(ctx, prompt, maxTok, drafter, K, greedy, col.Record)
		if err != nil {
			t.Fatalf("%s: %v", w.name, err)
		}
		got := collectTokens(ch)
		specDur := time.Since(t0)
		if err := col.Flush(); err != nil {
			t.Fatalf("%s: flush trace: %v", w.name, err)
		}
		if g.Err() != nil {
			t.Fatalf("%s: stream err %v", w.name, g.Err())
		}

		// Lossless gate: must match plain greedy exactly.
		if !slices.Equal(got, ref) {
			t.Fatalf("%s: speculative != greedy\n got %v\n ref %v", w.name, got, ref)
		}

		speedup := float64(plainDur) / float64(specDur)
		plainTPS := float64(len(ref)) / plainDur.Seconds()
		specTPS := float64(len(got)) / specDur.Seconds()
		t.Logf("%-11s %6d  %6.3f  %6.3f  %6.2f  %8.1f %8.1f %6.2fx",
			w.name, len(got), col.MeanAccept(), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound(), plainTPS, specTPS, speedup)
	}
}
