package serveapp

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

func TestParseBudgetFlag(t *testing.T) {
	cases := []struct {
		in      string
		want    budgetFlag
		wantErr bool
	}{
		{"", budgetFlag{}, false}, {"auto", budgetFlag{}, false}, {"AUTO", budgetFlag{}, false},
		{"unlimited", budgetFlag{unlimited: true}, false}, {"512", budgetFlag{n: 512}, false},
		{"0", budgetFlag{}, true}, {"-3", budgetFlag{}, true}, {"lots", budgetFlag{}, true},
	}
	for _, tc := range cases {
		got, err := parseBudgetFlag(tc.in)
		if (err != nil) != tc.wantErr || (err == nil && got != tc.want) {
			t.Errorf("parseBudgetFlag(%q) = %+v, %v; want %+v, err=%v", tc.in, got, err, tc.want, tc.wantErr)
		}
	}
}

// resolveBudget: the room rule (thinking leaves a quarter of the turn, min 2 tokens), and how a request's own budget and the
// flag combine with it.
func TestResolveBudget(t *testing.T) {
	auto, unlimited, cap100 := budgetFlag{}, budgetFlag{unlimited: true}, budgetFlag{n: 100}
	cases := []struct {
		name      string
		f         budgetFlag
		explicit  int
		maxTokens int
		want      int
		ok        bool
	}{
		{"auto: three quarters of a 160-token turn", auto, 0, 160, 120, true},
		{"auto: a 4096-token turn", auto, 0, 4096, 3072, true},
		{"auto: 8 tokens leaves 2 to answer", auto, 0, 8, 6, true},
		{"a turn under 4 tokens is left alone", auto, 0, 3, 0, false},
		{"request budget below the room holds", auto, 50, 160, 50, true},
		{"request budget above the room is clamped to it", auto, 1024, 1200, 900, true},
		{"Anthropic-style budget equal to max_tokens is clamped", auto, 1200, 1200, 900, true},
		{"unlimited applies no ceiling of its own", unlimited, 0, 4096, 0, false},
		{"unlimited still honours a request budget", unlimited, 200, 4096, 200, true},
		{"-reasoning-budget N caps an unasked request", cap100, 0, 4096, 100, true},
		{"N is itself clamped by the room", cap100, 0, 40, 30, true},
		{"a request budget overrides N", cap100, 300, 4096, 300, true},
	}
	for _, tc := range cases {
		got, ok := resolveBudget(tc.f, tc.explicit, tc.maxTokens)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("%s: got (%d, %v), want (%d, %v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
}

// due/Process/Gate: the processor as a pure function of the generated ids.
func TestReasoningBudget_due(t *testing.T) {
	const open, closeID = 10, 11
	prompt := &reasoningBudget{open: open, closeID: closeID, opens: true, limit: 3} // the prompt ended inside an open block
	self := &reasoningBudget{open: open, closeID: closeID, opens: false, limit: 3}  // the model opens the block itself
	cases := []struct {
		name string
		b    *reasoningBudget
		gen  []int
		want bool
	}{
		{"prompt-opened: nothing generated yet", prompt, nil, false},
		{"prompt-opened: one token under the limit", prompt, []int{1, 2}, false},
		{"prompt-opened: exactly at the limit forces", prompt, []int{1, 2, 3}, true},
		{"prompt-opened: over the limit still forces (a missed step)", prompt, []int{1, 2, 3, 4}, true},
		{"prompt-opened: already closed is never forced", prompt, []int{1, closeID, 3, 4, 5}, false},
		{"prompt-opened: closed exactly at the limit", prompt, []int{1, 2, closeID}, false},
		{"self-opened: no open token, no budget", self, []int{1, 2, 3, 4, 5}, false},
		{"self-opened: counts from after the open token", self, []int{7, open, 1, 2}, false},
		{"self-opened: at the limit forces", self, []int{7, open, 1, 2, 3}, true},
		{"self-opened: closed before the limit", self, []int{open, 1, closeID, 2, 3, 4}, false},
		{"self-opened: a block that opened, closed and answered is done", self, []int{open, 1, 2, 3, closeID, 9, 9, 9, 9}, false},
	}
	for _, tc := range cases {
		if got := tc.b.due(tc.gen); got != tc.want {
			t.Errorf("%s: due(%v) = %v, want %v", tc.name, tc.gen, got, tc.want)
		}
		if got := tc.b.Gate(tc.gen); got != tc.want {
			t.Errorf("%s: Gate(%v) = %v, want %v", tc.name, tc.gen, got, tc.want)
		}
		logits := make([]float32, 16)
		for i := range logits {
			logits[i] = float32(i) * 0.1
		}
		tc.b.Process(tc.gen, logits)
		if tc.want {
			for i, v := range logits {
				if i == closeID && v != 0 {
					t.Errorf("%s: the close token's logit was changed to %v, want 0", tc.name, v)
				}
				if i != closeID && !math.IsInf(float64(v), -1) {
					t.Errorf("%s: logit %d = %v, want -Inf (only the close token may survive)", tc.name, i, v)
				}
			}
		} else if logits[3] != 0.3 {
			t.Errorf("%s: logits were touched while the budget was not due", tc.name)
		}
	}
}

// The budget in the REAL decode loop. A gated processor is only as right as its first step: the gate is asked before any
// token exists, and again after each emitted one, and the fast paths resume when it says no. This runs the tiny model with
// the processor both gated and ungated and checks the same thing each way: the stream is identical to an unprocessed run up to
// the budget, the budget-th token is the close token, and nothing after it is forced.
func TestReasoningBudget_inTheDecodeLoop(t *testing.T) {
	_, lm := tinyServed(t)
	prompt := []int{3, 9, 27, 81, 5, 11}
	run := func(sp decoder.SamplingParams, n int) []int {
		t.Helper()
		stream, gen := lm.model.Generate(context.Background(), prompt, n, sp)
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if err := gen.Err(); err != nil {
			t.Fatal(err)
		}
		return out
	}
	base := run(decoder.SamplingParams{}, 14)
	if len(base) < 10 {
		t.Fatalf("baseline produced only %d tokens", len(base))
	}
	const limit = 5
	vocab := lm.model.Config().VocabSize
	closeID := (base[limit] + 1) % vocab // any id the model would NOT have picked at that step
	for _, gated := range []bool{true, false} {
		b := &reasoningBudget{open: 1, closeID: closeID, opens: true, limit: limit}
		sp := decoder.SamplingParams{LogitProcessor: b.Process}
		if gated {
			sp.LogitProcessorGate = b.Gate
		}
		got := run(sp, 14)
		if !slices.Equal(got[:limit], base[:limit]) {
			t.Errorf("gated=%v: the first %d tokens changed before the budget was due: %v vs %v", gated, limit, got[:limit], base[:limit])
		}
		if len(got) <= limit || got[limit] != closeID {
			t.Fatalf("gated=%v: token %d = %v, want the forced close id %d (stream %v)", gated, limit, got, closeID, got)
		}
		// One forced token, then the model is free: the processor must not keep forcing it.
		if len(got) > limit+2 && got[limit+1] == closeID && got[limit+2] == closeID {
			t.Errorf("gated=%v: the close token was forced more than once: %v", gated, got)
		}
	}
	// A budget that is never reached changes nothing.
	b := &reasoningBudget{open: 1, closeID: closeID, opens: true, limit: 1000}
	if got := run(decoder.SamplingParams{LogitProcessor: b.Process, LogitProcessorGate: b.Gate}, 14); !slices.Equal(got, base) {
		t.Errorf("an unreached budget changed the stream: %v vs %v", got, base)
	}
}

// applyBudget: where it installs, where it must not, and that it composes with the lazy tool union instead of replacing it.
func TestApplyBudget(t *testing.T) {
	srv := &server{}
	nine := templateFromGolden(t, "qwen3.5-9b").WithThinking(chat.ThinkTemplate) // default on, prompt opens the block
	small := templateFromGolden(t, "qwen3.5-0.8b").WithThinking(chat.ThinkTemplate)
	tk, vocab := thinkTokenizer(t, thinkAlphabet)
	_ = vocab
	lm := &loadedModel{tk: tk}
	ts := thinkSettings{}

	apply := func(lm *loadedModel, tm *chat.Template, gr *genRequest) { srv.applyBudget(lm, gr, tm, nil, ts) }

	gr := genRequest{maxTokens: 160}
	apply(lm, nine, &gr)
	if gr.sp.LogitProcessor == nil || gr.sp.LogitProcessorGate == nil {
		t.Fatal("the 9B (thinking on by default) must get a gated budget processor")
	}

	gr = genRequest{maxTokens: 160}
	apply(lm, small, &gr)
	if gr.sp.LogitProcessor != nil {
		t.Error("the 0.8B under its default (thinking off) can never be inside a block: no processor")
	}

	gr = genRequest{maxTokens: 160}
	apply(lm, nine.WithThinking(chat.ThinkOff), &gr)
	if gr.sp.LogitProcessor != nil {
		t.Error("thinking off: no processor")
	}

	gr = genRequest{maxTokens: 160}
	apply(&loadedModel{tk: tk, spec: true}, nine, &gr)
	if gr.sp.LogitProcessor != nil {
		t.Error("speculative decoding on: a processor would silently disable the drafter, so the budget must not install")
	}

	gr = genRequest{maxTokens: 160}
	apply(lm, chat.ChatML(), &gr)
	if gr.sp.LogitProcessor != nil {
		t.Error("a template with no recognised thinking control must never get a budget")
	}

	// Composition with a gated constraint (the lazy tool union): both must run, and the gate is the OR of the two.
	var prevCalls, prevGateCalls int
	gr = genRequest{maxTokens: 160}
	gr.sp.LogitProcessor = func([]int, []float32) { prevCalls++ }
	gr.sp.LogitProcessorGate = func([]int) bool { prevGateCalls++; return false }
	apply(lm, nine, &gr)
	if gr.sp.LogitProcessorGate([]int{1}) {
		t.Error("neither is due: the combined gate must stay false so the fast paths survive")
	}
	if prevGateCalls == 0 {
		t.Error("the combined gate never consulted the existing one — its state would stop advancing")
	}
	gr.sp.LogitProcessor([]int{1}, make([]float32, 8))
	if prevCalls != 1 {
		t.Errorf("the existing processor ran %d times, want 1", prevCalls)
	}

	// An ungated constraint owns every step; the budget stays out of its way.
	gr = genRequest{maxTokens: 160}
	calls := 0
	gr.sp.LogitProcessor = func([]int, []float32) { calls++ }
	apply(lm, nine, &gr)
	gr.sp.LogitProcessor(nil, make([]float32, 8))
	if calls != 1 || gr.sp.LogitProcessorGate != nil {
		t.Errorf("an ungated constraint must be left exactly as it was (calls=%d, gate set=%v)", calls, gr.sp.LogitProcessorGate != nil)
	}
}
