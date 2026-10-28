//go:build goinfer_testhooks

package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// ResidentStreamCase is one generateInto run on a resident model: its tokens and the counters that say which decode path
// served them (SB5, docs/tasks/task-subsystems-2026-10.md §8 SB5).
type ResidentStreamCase struct {
	Tokens        []int `json:"tokens"`
	DeviceSampled int   `json:"device_sampled,omitempty"`
	TopKServed    int   `json:"topk_served,omitempty"`
	TopKFallbacks int   `json:"topk_fallbacks,omitempty"`
	OptFwdGuessed int   `json:"optfwd_guessed,omitempty"`
	OptFwdHit     int   `json:"optfwd_hit,omitempty"`
	Budget        int   `json:"budget"`
	BudgetClamped bool  `json:"budget_clamped,omitempty"`
	Logprobs      int   `json:"logprobs,omitempty"`
}

// ResidentStreamsForTest runs each of generateInto's decode paths once on m, a model loaded resident on llama-tiny (a
// byte-level vocabulary), and returns the cases by name, prefixed with prefix. Each path is chosen by its knob and its
// sampling parameters; a knob is restored after its case. Under MC3 (m.EnableResidentConcurrency) the same cases run on
// the batcher. The streams are the before/after identity record for moving generateInto onto Forward.
func ResidentStreamsForTest(m *Model, prefix string) (map[string]ResidentStreamCase, error) {
	if !m.ResidentActive() {
		return nil, fmt.Errorf("model is not resident")
	}
	prompt := []int{}
	for _, b := range []byte("the cat sat on the mat. the cat sat on the mat. the cat") {
		prompt = append(prompt, int(b))
	}
	const n = 32
	maskOdd := func(_ []int, logits []float32) {
		for i := 1; i < len(logits); i += 2 {
			logits[i] = -1e30
		}
	}
	every3 := func(gen []int) bool { return len(gen)%3 == 0 }
	type cse struct {
		name  string
		sp    SamplingParams
		knobs map[string]string
	}
	noOpt := map[string]string{knobNoOptFwd: "1"}
	with := func(a map[string]string, k, v string) map[string]string {
		out := map[string]string{k: v}
		for x, y := range a {
			out[x] = y
		}
		return out
	}
	cases := []cse{
		{"greedy", SamplingParams{}, nil},
		{"greedy-logits", SamplingParams{}, map[string]string{knobNoGreedyFastpath: "1"}},
		{"topk1-temp", SamplingParams{TopK: 1, Temperature: 0.7}, nil},
		{"greedy-stop", SamplingParams{StopIDs: []int{38}}, nil},
		{"greedy-logprobs", SamplingParams{Logprobs: true}, nil},
		{"greedy-processor", SamplingParams{LogitProcessor: maskOdd}, nil},
		{"greedy-gated", SamplingParams{LogitProcessor: maskOdd, LogitProcessorGate: every3}, nil},
		{"sampled", SamplingParams{Temperature: 0.8, Seed: 7}, nil},
		{"sampled-no-optfwd", SamplingParams{Temperature: 0.8, Seed: 7}, noOpt},
		{"sampled-host", SamplingParams{Temperature: 0.8, Seed: 7}, with(noOpt, knobNoSampleFastpath, "1")},
		{"topk", SamplingParams{Temperature: 0.8, TopK: 40, Seed: 7}, noOpt},
		{"topk-host", SamplingParams{Temperature: 0.8, TopK: 40, Seed: 7}, with(noOpt, knobNoTopKFastpath, "1")},
		{"topp", SamplingParams{Temperature: 0.8, TopP: 0.9, Seed: 7}, noOpt},
		{"penalty", SamplingParams{Temperature: 0.8, RepeatPenalty: 1.3, Seed: 7}, noOpt},
		{"optfwd", SamplingParams{Temperature: 0.15, Seed: 7}, nil},
		{"optfwd-off", SamplingParams{Temperature: 0.15, Seed: 7}, noOpt},
	}
	out := map[string]ResidentStreamCase{}
	record := func(name string, toks []int, g *Generation) {
		r := ResidentStreamCase{Tokens: toks, DeviceSampled: g.DeviceSampled, TopKServed: g.TopKServed,
			TopKFallbacks: g.TopKFallbacks, Budget: g.Budget, BudgetClamped: g.BudgetClamped, Logprobs: len(g.Logprobs)}
		if r.Tokens == nil {
			r.Tokens = []int{}
		}
		if g.OptFwd != nil {
			r.OptFwdGuessed, r.OptFwdHit = g.OptFwd.Guessed, g.OptFwd.Hit
		}
		out[prefix+name] = r
	}
	for _, c := range cases {
		var restores []func()
		for k, v := range c.knobs {
			restores = append(restores, SetKnobForTest(m, k, v))
		}
		ch, g := m.Generate(context.Background(), prompt, n, c.sp)
		var toks []int
		for id := range ch {
			toks = append(toks, id)
		}
		for i := len(restores) - 1; i >= 0; i-- {
			restores[i]()
		}
		if err := g.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", c.name, err)
		}
		record(c.name, toks, g)
	}
	if m.batcher == nil {
		return out, nil
	}
	// Under MC3, three generations at once, so the batcher's shared steps run: each stream must still be the one its
	// generation decodes alone (the batched row is bit-identical to the solo call).
	conc := []cse{cases[0], cases[8], cases[10]}
	type res struct {
		toks []int
		g    *Generation
	}
	results := make([]res, len(conc))
	var wg sync.WaitGroup
	for i, c := range conc {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch, g := m.Generate(context.Background(), prompt, n, c.sp)
			var toks []int
			for id := range ch {
				toks = append(toks, id)
			}
			results[i] = res{toks, g}
		}()
	}
	wg.Wait()
	for i, c := range conc {
		if err := results[i].g.Err(); err != nil {
			return nil, fmt.Errorf("concurrent/%s: %w", c.name, err)
		}
		record("concurrent/"+c.name, results[i].toks, results[i].g)
	}
	return out, nil
}

// CompareResidentStreamsForTest writes got to outPath when it is set, and holds it to the record at wantPath when that is
// set. It returns one line per case that differs (or is missing on either side).
func CompareResidentStreamsForTest(got map[string]ResidentStreamCase, outPath, wantPath string) ([]string, error) {
	b, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		return nil, err
	}
	if outPath != "" {
		if err := os.WriteFile(outPath, append(b, '\n'), 0o644); err != nil {
			return nil, err
		}
	}
	if wantPath == "" {
		return nil, nil
	}
	wb, err := os.ReadFile(wantPath)
	if err != nil {
		return nil, err
	}
	var want map[string]ResidentStreamCase
	if err := json.Unmarshal(wb, &want); err != nil {
		return nil, err
	}
	var names []string
	for k := range want {
		names = append(names, k)
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	var diffs []string
	for _, k := range names {
		w, wok := want[k]
		g, gok := got[k]
		a, _ := json.Marshal(w)
		c, _ := json.Marshal(g)
		if !wok || !gok || string(a) != string(c) {
			diffs = append(diffs, fmt.Sprintf("%s:\n  want %s\n  got  %s", k, a, c))
		}
	}
	return diffs, nil
}
