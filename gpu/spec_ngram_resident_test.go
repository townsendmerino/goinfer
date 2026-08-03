//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// ngramWorkloads mirror the CPU harness (decoder/spec_harness_test.go): completion
// prompts with heavy internal repetition, the regime n-gram drafting wins on.
var ngramWorkloads = []struct {
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

// TestNgramSpecResident_throughput is the GPU-resident re-measure of the n-gram
// spoke (02) and adaptive depth (04). Acceptance is machine-independent and already
// measured on CPU; this measures the wall-clock that the CPU run could not — the
// batched-ForwardN verify on the device, with NO draft model occupying VRAM (the
// n-gram drafter is pure-Go, so unlike GenerateSpeculative the whole 8 GB is the
// target's, and every accepted token is pure win). Parity is hard-gated; speedup is
// logged per workload for fixed K=8 vs adaptive vs plain resident greedy.
//
//	GOINFER_SPEC_TARGET=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -tags gpu ./gpu/ -run TestNgramSpecResident_throughput -v -timeout 30m
func TestNgramSpecResident_throughput(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if testing.Short() {
		t.Skip("throughput measurement: skipped in -short")
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		tpath = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(tpath); err != nil {
		t.Skipf("missing model %s: %v", tpath, err)
	}

	target, err := decoder.Load(tpath, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not GPU-resident (ineligible / no residency)")
	}
	tk, err := tokenizer.LoadGGUF(tpath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	collect := func(ch <-chan int) []int {
		var v []int
		for id := range ch {
			v = append(v, id)
		}
		return v
	}
	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const n = 160
	const K = 8

	// timeRun runs once to warm pipelines/caches (discarded), then times a second run.
	timeRun := func(run func() ([]int, *decoder.Generation)) ([]int, float64, *decoder.Generation) {
		run()
		t0 := time.Now()
		toks, g := run()
		dt := time.Since(t0)
		return toks, float64(len(toks)) / dt.Seconds(), g
	}

	t.Logf("%-11s %-8s %6s  %6s  %7s  %7s", "workload", "mode", "tok", "tok/v", "tok/s", "speedup")
	for _, w := range ngramWorkloads {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("%s: encode: %v", w.name, err)
		}

		ref, refTPS, _ := timeRun(func() ([]int, *decoder.Generation) {
			ch, g := target.Generate(ctx, prompt, n, greedy)
			return collect(ch), g
		})
		t.Logf("%-11s %-8s %6d  %6s  %7.1f  %7s", w.name, "plain", len(ref), "-", refTPS, "1.00x")

		fixed, fixedTPS, gFixed := timeRun(func() ([]int, *decoder.Generation) {
			ch, g, err := target.GenerateNgramSpeculative(ctx, prompt, n, &decoder.NgramDrafter{}, K, greedy)
			if err != nil {
				t.Fatalf("%s fixed: %v", w.name, err)
			}
			return collect(ch), g
		})
		if gFixed.Err() != nil || !slices.Equal(fixed, ref) {
			t.Fatalf("%s fixed: resident spec != greedy (err %v)", w.name, gFixed.Err())
		}
		t.Logf("%-11s %-8s %6d  %6.2f  %7.1f  %6.2fx", w.name, "fixed", len(fixed), gFixed.Spec.TokensPerRound(), fixedTPS, fixedTPS/refTPS)

		ada, adaTPS, gAda := timeRun(func() ([]int, *decoder.Generation) {
			ch, g, err := target.GenerateNgramSpeculativeAdaptive(ctx, prompt, n, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: K}, greedy)
			if err != nil {
				t.Fatalf("%s adaptive: %v", w.name, err)
			}
			return collect(ch), g
		})
		if gAda.Err() != nil || !slices.Equal(ada, ref) {
			t.Fatalf("%s adaptive: resident spec != greedy (err %v)", w.name, gAda.Err())
		}
		t.Logf("%-11s %-8s %6d  %6.2f  %7.1f  %6.2fx", w.name, "adaptive", len(ada), gAda.Spec.TokensPerRound(), adaTPS, adaTPS/refTPS)
	}
}
