package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMC2_decodeMultiStepBitIdentical is MC2's identity gate: B sequences stepped together through decodeMultiStep
// emit logits bit-identical to the production single-token forward over copies of the same caches, teacher-forced,
// at different depths per sequence. It runs on the committed llama-tiny fixture (CI), and on
// GOINFER_MC2_MODEL when set (int4 and int8int8).
func TestMC2_decodeMultiStepBitIdentical(t *testing.T) {
	type cfg struct{ path, quant string }
	cfgs := []cfg{{"../testdata/llama-tiny", ""}}
	if p := os.Getenv("GOINFER_MC2_MODEL"); p != "" {
		cfgs = append(cfgs, cfg{p, "int4"}, cfg{p, "int8int8"})
	}
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path)+"/"+c.quant, func(t *testing.T) {
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no checkpoint at %s: %v", c.path, err)
			}
			m, err := Load(c.path, Options{Quant: c.quant, Backend: os.Getenv("GOINFER_MC2_BACKEND")})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if err := m.cpuBatchModelEligible(); err != nil {
				t.Skipf("%v", err)
			}
			depths := []int{5, 23, 40, 11}
			assertBatchedStepBitIdentical(t, m, depths, 12)
			t.Logf("%s %q: %d sequences x 12 steps bit-identical to the single-token forward", filepath.Base(c.path), c.quant, len(depths))
		})
	}
}

func argmaxF32(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestMC2_batchedDecodeThroughput is MC2's measurement (docs/tasks/task-concurrency-2026-09.md): aggregate decode tok/s
// on CPU for B sequences through decodeMultiStep (B = 1, 2, 4, 8), against the production single-token forward taking
// the B sequences one at a time ("serial", today's behaviour) and against J8's cell, N independent decode workers on
// the same model (N = 2, 4 concurrent forwards, each on its own cache). Every arm decodes S steps per sequence from a
// cache prefilled to the same depth; arms are interleaved rep by rep, and the arm order rotates.
//
//	GOINFER_MC2=1 GOINFER_MC2_MODEL=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -count=1 -timeout 60m -run '^TestMC2_batchedDecodeThroughput$' -v ./decoder/
//
// Env: GOINFER_MC2_BACKEND (default "": a row4-only .giw, the cpu-arm64 / cpu-amd64 prequant targets, needs "cpu"),
// GOINFER_MC2_QUANT (default int4), GOINFER_MC2_DEPTH (default 128), GOINFER_MC2_STEPS (default 16),
// GOINFER_MC2_REPS (default 5).
func TestMC2_batchedDecodeThroughput(t *testing.T) {
	if os.Getenv("GOINFER_MC2") != "1" {
		t.Skip("set GOINFER_MC2=1 and GOINFER_MC2_MODEL (loads a real checkpoint; minutes of CPU time)")
	}
	path := os.Getenv("GOINFER_MC2_MODEL")
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	envInt := func(k string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
			return v
		}
		return def
	}
	quant := os.Getenv("GOINFER_MC2_QUANT")
	if quant == "" {
		quant = "int4"
	}
	depth, steps, reps := envInt("GOINFER_MC2_DEPTH", 128), envInt("GOINFER_MC2_STEPS", 16), envInt("GOINFER_MC2_REPS", 5)
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc2 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	m, err := Load(path, Options{Quant: quant, Backend: os.Getenv("GOINFER_MC2_BACKEND")})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := m.cpuBatchModelEligible(); err != nil {
		t.Skipf("%v", err)
	}
	vocab := m.w.arch.VocabSize
	ctx := context.Background()
	const maxB = 8
	// maxB caches prefilled to depth once; each arm rewinds them to depth before running (positional KV: truncation
	// is exact), so every arm decodes over identical history.
	caches := make([]*KVCache, maxB)
	for b := range caches {
		prompt := make([]int, depth)
		for i := range prompt {
			prompt[i] = (i*37 + b*11 + 3) % vocab
		}
		caches[b] = m.NewCache(depth + steps + 8)
		if _, err := m.prefillLogits(ctx, prompt, caches[b]); err != nil {
			t.Fatalf("prefill: %v", err)
		}
	}
	rewind := func(n int) {
		for b := range n {
			caches[b].TruncateTo(depth)
		}
	}
	startIDs := func(n int) []int {
		ids := make([]int, n)
		for b := range ids {
			ids[b] = (b*53 + 7) % vocab
		}
		return ids
	}
	type arm struct {
		name string
		n    int // sequences
		run  func() error
	}
	batched := func(B int) func() error {
		return func() error {
			ids := startIDs(B)
			for range steps {
				lg, err := m.decodeMultiStep(ids, caches[:B])
				if err != nil {
					return err
				}
				for b := range ids {
					ids[b] = argmaxF32(lg[b])
				}
			}
			return nil
		}
	}
	serial := func(B int) func() error { // today: one generation at a time, each sequence's steps in turn
		return func() error {
			ids := startIDs(B)
			for range steps {
				for b := range ids {
					lg, err := m.forward(ids[b], caches[b])
					if err != nil {
						return err
					}
					ids[b] = argmaxF32(lg)
				}
			}
			return nil
		}
	}
	workers := func(N int) func() error { // J8: N concurrent decode workers, each on its own cache
		return func() error {
			ids := startIDs(N)
			var wg sync.WaitGroup
			errs := make([]error, N)
			for b := range N {
				wg.Add(1)
				go func(b int) {
					defer wg.Done()
					id := ids[b]
					for range steps {
						lg, err := m.forward(id, caches[b])
						if err != nil {
							errs[b] = err
							return
						}
						id = argmaxF32(lg)
					}
				}(b)
			}
			wg.Wait()
			for _, e := range errs {
				if e != nil {
					return e
				}
			}
			return nil
		}
	}
	arms := []arm{
		{"serial x1", 1, serial(1)}, {"serial x4", 4, serial(4)},
		{"batched B=1", 1, batched(1)}, {"batched B=2", 2, batched(2)}, {"batched B=4", 4, batched(4)}, {"batched B=8", 8, batched(8)},
		{"J8 workers N=2", 2, workers(2)}, {"J8 workers N=4", 4, workers(4)},
	}
	hb("loaded %s (%s): depth %d, %d steps per sequence, %d reps, %d arms", filepath.Base(path), quant, depth, steps, reps, len(arms))
	for _, a := range arms { // warm every arm once
		rewind(a.n)
		if err := a.run(); err != nil {
			t.Fatalf("%s: %v", a.name, err)
		}
	}
	rates := make([][]float64, len(arms))
	for rep := range reps {
		line := fmt.Sprintf("rep %d/%d:", rep+1, reps)
		for k := range arms {
			ai := (k + rep) % len(arms)
			a := arms[ai]
			rewind(a.n)
			st := time.Now()
			if err := a.run(); err != nil {
				t.Fatalf("%s: %v", a.name, err)
			}
			r := float64(a.n*steps) / time.Since(st).Seconds()
			rates[ai] = append(rates[ai], r)
		}
		for ai, a := range arms {
			line += fmt.Sprintf("  %s %.1f", a.name, rates[ai][rep])
		}
		hb("%s tok/s", line)
	}
	med := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	base := med(rates[0])
	hb("aggregate decode tok/s, median of %d reps (%s %s, depth %d):", reps, filepath.Base(path), quant, depth)
	for ai, a := range arms {
		hb("  %-16s %7.1f tok/s  %.3fx serial x1", a.name, med(rates[ai]), med(rates[ai])/base)
	}
	// The registered MC2 metric: batched B=4 against B=1 (the production single-token forward, one sequence).
	ratios := make([]float64, reps)
	for i := range ratios {
		ratios[i] = rates[4][i] / rates[0][i]
	}
	sort.Float64s(ratios)
	hb("MC2 METRIC batched B=4 / serial x1 = %.3fx (per-rep sorted %v); J8 N=4 / serial x1 = %.3fx",
		ratios[len(ratios)/2], ratios, med(rates[7])/base)
}
