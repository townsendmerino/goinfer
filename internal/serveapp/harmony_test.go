package serveapp

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// A real gpt-oss-20b reply (goinfer-chat, temperature 0, 2026-09-30; the same capture chat/harmony_parse_test.go pins) through
// serve's own reasoning router — newThinkOut is the object streamTokens holds for every route — cut into token-sized pieces the
// way a decode loop delivers it. Before the Harmony parser every marker reached `content`; here the client-facing streams must
// carry the analysis as reasoning and the final channel as the answer, and nothing else.
const realGptOssReply = "<|channel|>analysis<|message|>The user asks: \"What is 17 times 3? Then write one short sentence about the sea.\"\n\nWe need to answer the multiplication: 17 times 3 equals 51. Then write one short sentence about the sea. Provide short sentence about the sea. Probably something like \"The sea is vast and full of mysteries.\" Provide short sentence. Ensure short.<|end|><|start|>assistant<|channel|>final<|message|>17 × 3 = 51.  \nThe sea stretches endlessly, its waves whispering ancient secrets."

func TestThinkOut_harmonyRealReply(t *testing.T) {
	wantReasoning := realGptOssReply[len("<|channel|>analysis<|message|>"):strings.Index(realGptOssReply, "<|end|>")]
	wantAnswer := realGptOssReply[strings.Index(realGptOssReply, "<|channel|>final<|message|>")+len("<|channel|>final<|message|>"):]

	run := func(ts thinkSettings, pieceLen int) (reasoning, answer string, truncated bool, th *thinkOut) {
		var rb strings.Builder
		th = newThinkOut(chat.Harmony(), nil, ts, func(s string) { rb.WriteString(s) })
		if th == nil {
			return "", realGptOssReply, false, nil
		}
		var ab strings.Builder
		for i := 0; i < len(realGptOssReply); i += pieceLen {
			ab.WriteString(th.feed(realGptOssReply[i:min(i+pieceLen, len(realGptOssReply))]))
		}
		ab.WriteString(th.finish())
		return rb.String(), ab.String(), th.endedInReasoning, th
	}

	for _, n := range []int{1, 2, 3, 5, 8, 13, 40} {
		r, a, trunc, _ := run(thinkSettings{mode: chat.ThinkTemplate, format: rfSplit}, n)
		if r != wantReasoning || a != wantAnswer || trunc {
			t.Errorf("pieces of %d: reasoning=%q answer=%q truncated=%v", n, r, a, trunc)
		}
	}

	// -reasoning-format none restores the old raw output: no router, every marker in content (an operator's escape hatch).
	if _, a, _, th := run(thinkSettings{mode: chat.ThinkTemplate, format: rfNone}, 7); th != nil || a != realGptOssReply {
		t.Errorf("-reasoning-format none must leave the reply untouched, got router=%v answer=%q", th != nil, a)
	}

	// Cut off while thinking (max_tokens): no answer, and the router says so, so the route reports finish_reason "length".
	var rb strings.Builder
	th := newThinkOut(chat.Harmony(), nil, thinkSettings{mode: chat.ThinkTemplate, format: rfSplit}, func(s string) { rb.WriteString(s) })
	cut := realGptOssReply[:len("<|channel|>analysis<|message|>The user asks: what is")]
	if ans := th.feed(cut) + th.finish(); ans != "" || !th.endedInReasoning || rb.String() != strings.TrimPrefix(cut, "<|channel|>analysis<|message|>") {
		t.Errorf("cut off in the analysis: answer=%q endedInReasoning=%v reasoning=%q", ans, th.endedInReasoning, rb.String())
	}

	// Stop strings are matched on the answer only: a stop string in the analysis does not end the reply.
	th = newThinkOut(chat.Harmony(), nil, thinkSettings{mode: chat.ThinkTemplate, format: rfSplit}, func(string) {})
	seen := th.feed(realGptOssReply)
	if _, _, hit := firstStop(seen, []string{"multiplication"}); hit {
		t.Error("the stop logic was shown the analysis text")
	}
	if _, _, hit := firstStop(seen, []string{"sea stretches"}); !hit {
		t.Error("the stop logic was not shown the answer text")
	}
}

// The load log says what serve does about reasoning for this model: for gpt-oss, that its channel messages are split and that
// -thinking has no off form to apply — otherwise an operator who passes -thinking off sees reasoning_content anyway and has no hint why.
func TestThinkingNote_harmony(t *testing.T) {
	got := thinkingNote(chat.Harmony())
	for _, want := range []string{"analysis channel", "cannot turn it off"} {
		if !strings.Contains(got, want) {
			t.Errorf("thinkingNote(harmony) = %q, want it to say %q", got, want)
		}
	}
	if note := thinkingNote(chat.Llama3()); note != "" {
		t.Errorf("a family with no reasoning at all gets no note, got %q", note)
	}
}

// harmonyTokenizer is a vocab in which the seven words of the gpt-oss channel protocol are single tokens, as they are in the
// real vocabulary (verified in chat's TestHarmonyBudget_realVocabulary).
func harmonyTokenizer(t *testing.T) (*tokenizer.Tokenizer, map[string]int) {
	t.Helper()
	vocab := map[string]int{}
	for _, tok := range []string{"a", "b", "c", "<|channel|>", "analysis", "<|message|>", "<|end|>", "<|start|>", "assistant", "final"} {
		vocab[tok] = len(vocab)
	}
	raw, _ := json.Marshal(map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder": map[string]any{"type": "ByteLevel"},
	})
	tk, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tk, vocab
}

// applyBudget for gpt-oss: installed under every -thinking mode (its prompt has no off form, so it reasons whatever the mode),
// not for a turn with no room, and not where speculative decoding would be silently switched off by a processor.
func TestApplyBudget_harmony(t *testing.T) {
	srv := &server{}
	tk, _ := harmonyTokenizer(t)
	lm := &loadedModel{tk: tk}
	for _, mode := range []chat.ThinkMode{chat.ThinkAsIs, chat.ThinkTemplate, chat.ThinkOn, chat.ThinkOff} {
		gr := genRequest{maxTokens: 160}
		srv.applyBudget(lm, &gr, chat.Harmony().WithThinking(mode), nil, thinkSettings{mode: mode})
		if gr.sp.LogitProcessor == nil || gr.sp.LogitProcessorGate == nil {
			t.Errorf("mode %v: no gated budget processor for gpt-oss", mode)
		}
	}
	gr := genRequest{maxTokens: 160}
	srv.applyBudget(lm, &gr, chat.Harmony(), nil, thinkSettings{budget: 40})
	if gr.sp.LogitProcessor == nil {
		t.Error("a request's own thinking_token_budget must still install the budget")
	}
	gr = genRequest{maxTokens: 12} // room 9: header (3) + forced sequence (6) leave no reasoning token
	srv.applyBudget(lm, &gr, chat.Harmony(), nil, thinkSettings{})
	if gr.sp.LogitProcessor != nil {
		t.Error("a turn too short to hold the header and the forced sequence must get no budget")
	}
	gr = genRequest{maxTokens: 160}
	srv.applyBudget(&loadedModel{tk: tk, spec: true}, &gr, chat.Harmony(), nil, thinkSettings{})
	if gr.sp.LogitProcessor != nil {
		t.Error("speculative decoding on: a processor would silently disable the drafter")
	}
}

// The forced SEQUENCE in the real decode loop, gated and ungated: the stream is untouched up to the trigger, then exactly the six
// forced tokens, whatever the model would have said, and the model is free afterwards. (A gated processor is only as right as its
// first step; the single-token budget has the same test above.)
func TestHarmonyBudget_inTheDecodeLoop(t *testing.T) {
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
	base := run(decoder.SamplingParams{}, 24)
	if len(base) < 20 {
		t.Fatalf("baseline produced only %d tokens", len(base))
	}
	vocab := lm.model.Config().VocabSize
	const trigger = 8
	head := base[:3] // the reply "opens with the analysis header"
	endID := 0
	for slices.Contains(base[:trigger], endID) {
		endID++ // an id the model has not written, so the budget sees an analysis it did not close itself
	}
	force := []int{endID, (endID + 1) % vocab, (endID + 2) % vocab, (endID + 3) % vocab, (endID + 4) % vocab, (endID + 5) % vocab}
	for _, gated := range []bool{true, false} {
		b := chat.NewHarmonyBudget(head, endID, force, trigger+len(force))
		sp := decoder.SamplingParams{LogitProcessor: b.Process}
		if gated {
			sp.LogitProcessorGate = b.Gate
		}
		got := run(sp, 24)
		if !slices.Equal(got[:trigger], base[:trigger]) {
			t.Errorf("gated=%v: tokens changed before the trigger: %v vs %v", gated, got[:trigger], base[:trigger])
		}
		if len(got) < trigger+len(force) || !slices.Equal(got[trigger:trigger+len(force)], force) {
			t.Errorf("gated=%v: tokens %d..%d = %v, want the forced sequence %v", gated, trigger, trigger+len(force), got[min(trigger, len(got)):min(trigger+len(force), len(got))], force)
		}
		if len(got) <= trigger+len(force) {
			t.Errorf("gated=%v: the stream ended with the forced sequence (%d tokens): the model must be free after it", gated, len(got))
		}
	}
	// A budget the reply never reaches changes nothing.
	b := chat.NewHarmonyBudget(head, endID, force, 1000)
	if got := run(decoder.SamplingParams{LogitProcessor: b.Process, LogitProcessorGate: b.Gate}, 24); !slices.Equal(got, base) {
		t.Errorf("an unreached budget changed the stream: %v vs %v", got, base)
	}
}

// reasoning_effort low|medium|high is gpt-oss's real knob (its template writes it on the `Reasoning:` line); on every other family
// the field keeps meaning only "none". Both routes that carry it — chat completions' flat reasoning_effort and the Responses API's
// nested reasoning.effort — reach the same resolution.
func TestResolveThink_reasoningEffort(t *testing.T) {
	srv := &server{}
	for in, want := range map[string]string{"low": "low", "HIGH": "high", " medium": "medium", "none": "", "minimal": "", "bogus": "", "": ""} {
		ts, err := srv.resolveThink(thinkRequest{reasoningEffort: in})
		if err != nil || ts.effort != want {
			t.Errorf("reasoning_effort %q: effort=%q err=%v, want %q", in, ts.effort, err, want)
		}
	}
	// "none" is still the off switch of the families that have one, untouched by effort.
	if ts, _ := srv.resolveThink(thinkRequest{reasoningEffort: "none"}); !ts.explicit || ts.mode != chat.ThinkOff || ts.effort != "" {
		t.Errorf("none must still turn thinking off and set no effort: %+v", ts)
	}
	// An effort does not decide thinking on or off, and an explicit enable_thinking is not disturbed by it.
	ts, _ := srv.resolveThink(thinkRequest{reasoningEffort: "high"})
	if ts.explicit {
		t.Errorf("a bare effort must not make the request choose on/off: %+v", ts)
	}
	ts, _ = srv.resolveThink(thinkRequest{reasoningEffort: "low", kwargs: map[string]json.RawMessage{"enable_thinking": json.RawMessage("true")}})
	if !ts.explicit || ts.mode != chat.ThinkOn || ts.effort != "low" {
		t.Errorf("enable_thinking and effort are independent: %+v", ts)
	}

	var rr responseReq
	if err := json.Unmarshal([]byte(`{"reasoning":{"effort":"low"}}`), &rr); err != nil {
		t.Fatal(err)
	}
	if ts, _ := srv.resolveThink(rr.thinkRequest()); ts.effort != "low" {
		t.Errorf("the Responses API's reasoning.effort did not reach the resolution: %+v", ts)
	}
}

// templateFor: the effort lands in gpt-oss's system block and in no other model's prompt.
func TestTemplateFor_reasoningEffort(t *testing.T) {
	turns := []chat.Turn{{Role: "user", Content: "Hi"}}
	oss := &loadedModel{tmpl: chat.Harmony()}
	if got := oss.templateFor(thinkSettings{effort: "low"}).Render("", turns); !strings.Contains(got, "Reasoning: low\n\n") {
		t.Errorf("gpt-oss prompt with effort low:\n%s", got)
	}
	if got := oss.templateFor(thinkSettings{}).Render("", turns); !strings.Contains(got, "Reasoning: medium\n\n") {
		t.Errorf("no effort asked for: the template's own default (medium) must stand:\n%s", got)
	}
	// The effort survives -thinking off / on alongside it.
	if got := oss.templateFor(thinkSettings{effort: "high", explicit: true, mode: chat.ThinkOff}).Render("", turns); !strings.Contains(got, "Reasoning: high\n\n") {
		t.Errorf("effort lost when the request also chose a thinking mode:\n%s", got)
	}
	for _, tm := range []*chat.Template{chat.ChatML(), chat.Llama3()} {
		other := &loadedModel{tmpl: tm}
		if a, b := other.templateFor(thinkSettings{effort: "high"}).Render("", turns), tm.Render("", turns); a != b {
			t.Errorf("%s: a reasoning_effort moved a non-gpt-oss prompt:\n%s\nvs\n%s", tm.Name(), a, b)
		}
	}
	if got := (&loadedModel{}).templateFor(thinkSettings{effort: "high"}); got != nil {
		t.Error("a model with no template stays template-less")
	}
}

// settleCalls: where a finished turn's tool calls are read from. A gpt-oss call is a message the router takes out of the answer text, so
// it has to be asked; with no router (-reasoning-format none) the buffer is the raw reply and the template's parser reads it; and every
// other family's calls are in the buffer exactly as before.
func TestSettleCalls_harmony(t *testing.T) {
	tools := []chat.Tool{{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)}}
	call := `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}`
	feed := func(reply string, ts thinkSettings) (raw string, th *thinkOut) {
		th = newThinkOut(chat.Harmony(), nil, ts, func(string) {})
		var b strings.Builder
		for i := 0; i < len(reply); i += 6 { // token-sized pieces
			b.WriteString(th.feed(reply[i:min(i+6, len(reply))]))
		}
		b.WriteString(th.finish())
		return b.String(), th
	}
	ts := thinkSettings{mode: chat.ThinkTemplate, format: rfSplit}

	// With the router: the buffered text has no call in it, the router does.
	reply := "<|channel|>analysis<|message|>need the weather<|end|><|start|>assistant" + call
	raw, th := feed(reply, ts)
	if raw != "" {
		t.Fatalf("setup: the call leaked into the buffered text: %q", raw)
	}
	calls, lead := settleCalls(chat.Harmony(), raw, tools, th)
	if len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Paris"}` || lead != "" {
		t.Errorf("router path: calls=%+v lead=%q", calls, lead)
	}

	// A preamble before the call is the prose, trimmed as every parser trims it.
	raw, th = feed("<|channel|>commentary<|message|>Let me check.\n<|end|><|start|>assistant"+call, ts)
	if calls, lead = settleCalls(chat.Harmony(), raw, tools, th); len(calls) != 1 || lead != "Let me check." {
		t.Errorf("preamble path: calls=%+v lead=%q", calls, lead)
	}

	// No router (-reasoning-format none): the buffer is the raw reply and the template's own parser reads the call out of it.
	if calls, lead = settleCalls(chat.Harmony(), reply, tools, nil); len(calls) != 1 || calls[0].Name != "get_weather" {
		t.Errorf("raw path: calls=%+v lead=%q", calls, lead)
	}

	// A prose answer: no calls, the text as it is.
	raw, th = feed("<|channel|>analysis<|message|>t<|end|><|start|>assistant<|channel|>final<|message|>It is 18C.", ts)
	if calls, lead = settleCalls(chat.Harmony(), raw, tools, th); len(calls) != 0 || lead != "It is 18C." {
		t.Errorf("prose path: calls=%+v lead=%q", calls, lead)
	}

	// Another family is untouched: its call is in the buffer, and a router (or none) changes nothing.
	chatml := "Sure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	if calls, lead = settleCalls(chat.ChatML(), chatml, tools, nil); len(calls) != 1 || calls[0].Name != "get_weather" || lead != "Sure." {
		t.Errorf("chatml: calls=%+v lead=%q", calls, lead)
	}
	if (*thinkOut)(nil).toolCalls() != nil {
		t.Error("a nil router has no calls")
	}
}
