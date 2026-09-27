package decoder

import (
	"context"
	"os"
	"slices"
	"sync"
	"testing"
)

// TestConcurrentSessions_matchAlone is MC3c's decoder-side precondition (docs/tasks/task-concurrency-2026-09.md): N
// conversations on distinct Sessions of ONE Model, generating at the same time on the CPU, each emit exactly the ids
// they emit when run alone — the property serve's -max-concurrent relies on. Run it under -race: a data race between
// generations (shared scratch, a shared pool, a lazily built table) is the failure it exists to catch.
//
// It uses the committed llama-tiny fixture; GOINFER_MC3C_MODEL points it at a real checkpoint instead (int4).
func TestConcurrentSessions_matchAlone(t *testing.T) {
	path, quant := "../testdata/llama-tiny", ""
	if p := os.Getenv("GOINFER_MC3C_MODEL"); p != "" {
		path, quant = p, "int4"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if m.resident != nil {
		t.Skip("MC3c covers the CPU path; this model went GPU-resident")
	}
	vocab := m.w.arch.VocabSize
	const nConv, turns, maxTok = 4, 3, 10
	// run plays conversation c for `turns` turns on its own session and returns every turn's ids.
	run := func(c int) [][]int {
		s := m.NewSession(256)
		prompt := []int{1, 2, 3, (c*17 + 5) % vocab, (c*29 + 9) % vocab}
		var out [][]int
		for tn := range turns {
			ch, gen := s.Generate(context.Background(), prompt, maxTok, SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Errorf("conversation %d turn %d: %v", c, tn, err)
				return out
			}
			out = append(out, ids)
			prompt = append(append(slices.Clone(prompt), ids...), (c*7+tn)%vocab, (c*11+tn)%vocab)
		}
		return out
	}
	alone := make([][][]int, nConv)
	for c := range nConv {
		alone[c] = run(c)
	}
	concurrent := make([][][]int, nConv)
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			concurrent[c] = run(c)
		}(c)
	}
	wg.Wait()
	for c := range nConv {
		for tn := range alone[c] {
			if !slices.Equal(alone[c][tn], concurrent[c][tn]) {
				t.Errorf("conversation %d turn %d: concurrent %v, alone %v", c, tn, concurrent[c][tn], alone[c][tn])
			}
		}
	}
}
