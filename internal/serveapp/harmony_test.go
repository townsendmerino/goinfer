package serveapp

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
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
