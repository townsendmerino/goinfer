package serveapp

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

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
		b := chat.NewReasoningBudget(1, closeID, true, limit)
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
	b := chat.NewReasoningBudget(1, closeID, true, 1000)
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
