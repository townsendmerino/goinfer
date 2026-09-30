package serveapp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

func TestMessagesToTurns_reasoning(t *testing.T) {
	msgs := []chatMessage{
		{Role: "user", Content: rawStr("Hi")},
		{Role: "assistant", Content: rawStr("A"), ReasoningContent: "from reasoning_content", Reasoning: "ignored: reasoning_content wins"},
		{Role: "assistant", Content: rawStr("B"), Reasoning: "from reasoning"},
		{Role: "assistant", Content: rawStr("C")},
	}
	_, turns := messagesToTurns(msgs)
	want := []string{"", "from reasoning_content", "from reasoning", ""}
	for i, w := range want {
		if turns[i].Reasoning != w {
			t.Errorf("turn %d: Reasoning = %q, want %q", i, turns[i].Reasoning, w)
		}
	}
	var decoded chatMessage
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":"x","reasoning_content":"r1","reasoning":"r2"}`), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ReasoningContent != "r1" || decoded.Reasoning != "r2" {
		t.Errorf("the wire fields were not decoded: %+v", decoded)
	}
}

// anthropicReqFrom builds a request from raw messages JSON.
func anthropicReqFrom(t *testing.T, messages string) *anthropicReq {
	t.Helper()
	var req anthropicReq
	if err := json.Unmarshal([]byte(`{"model":"m","max_tokens":10,"messages":`+messages+`}`), &req); err != nil {
		t.Fatal(err)
	}
	return &req
}

func TestAnthropicTurns_thinkingBlocksAndToolLoop(t *testing.T) {
	// What Claude Code sends on the second request of a tool loop (recorded against serve 2026-09-30): the assistant message
	// replays its thinking block (empty signature) with text and the tool_use; the user message carries the tool_result AND a
	// reminder text block.
	req := anthropicReqFrom(t, `[
	  {"role":"user","content":[{"type":"text","text":"Weather in Paris?"}]},
	  {"role":"assistant","content":[
	    {"type":"thinking","thinking":"need the weather","signature":""},
	    {"type":"text","text":"Checking."},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
	  {"role":"user","content":[
	    {"type":"tool_result","tool_use_id":"toolu_1","content":"18C"},
	    {"type":"text","text":"<system-reminder>be brief</system-reminder>"}]}
	]`)
	_, turns, err := anthropicTurns(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 4 {
		t.Fatalf("got %d turns, want user, assistant, tool, reminder: %+v", len(turns), turns)
	}
	if turns[1].Role != "assistant" || turns[1].Reasoning != "need the weather" || turns[1].Content != "Checking." || len(turns[1].ToolCalls) != 1 {
		t.Errorf("assistant turn: %+v", turns[1])
	}
	if turns[2].Role != "tool" {
		t.Errorf("turn 2 should be the tool result: %+v", turns[2])
	}
	if turns[3].Role != "user" || !turns[3].ToolLoop {
		t.Errorf("reminder text beside a tool_result must be a user turn marked ToolLoop: %+v", turns[3])
	}
	if turns[0].ToolLoop {
		t.Error("the real query must not be marked as a tool-loop continuation")
	}

	// A user message that has text but NO tool_result is a real query, never marked.
	_, turns, _ = anthropicTurns(anthropicReqFrom(t, `[{"role":"user","content":[{"type":"text","text":"hello"}]}]`))
	if len(turns) != 1 || turns[0].ToolLoop {
		t.Errorf("plain user text: %+v", turns)
	}

	// An assistant message that is only a thinking block produces no turn (there is nothing to replay it on), as before.
	_, turns, _ = anthropicTurns(anthropicReqFrom(t, `[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":""}]}]`))
	if len(turns) != 1 {
		t.Errorf("a thinking-only assistant message must not become a turn: %+v", turns)
	}
}

func TestMergeAdjacent_carriesReasoningAndToolLoop(t *testing.T) {
	got := mergeAdjacent([]chat.Turn{
		{Role: "assistant", Content: "a", Reasoning: "r1"},
		{Role: "assistant", Content: "b", Reasoning: "r2"},
		{Role: "user", Content: "u1", ToolLoop: true},
		{Role: "user", Content: "u2", ToolLoop: false},
	})
	if len(got) != 2 || got[0].Reasoning != "r1\nr2" || got[0].Content != "a\nb" {
		t.Fatalf("assistant merge: %+v", got)
	}
	if got[1].ToolLoop {
		t.Errorf("a merged user turn is a continuation only if every part was: %+v", got[1])
	}
}

// The point of the work, end to end through the real Qwen3 template: an OpenAI-shaped client (reasoning_content on the assistant
// message, a `tool` message) and an Anthropic-shaped one (a thinking block, tool_result + reminder text) replaying the SAME tool
// loop must put the model's reasoning into the prompt the same way, and Claude Code's reminder must not cost it the reasoning.
func TestHistory_bothClientShapesRenderTheReasoning(t *testing.T) {
	tm := templateFromGolden(t, "qwen3-4b").WithThinking(chat.ThinkTemplate)
	tools := []chat.Tool{{Name: "get_weather", Description: "Weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}

	var oa []chatMessage
	if err := json.Unmarshal([]byte(`[
	  {"role":"user","content":"Weather in Paris?"},
	  {"role":"assistant","content":"","reasoning_content":"need the weather",
	   "tool_calls":[{"id":"toolu_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
	  {"role":"tool","content":"18C","name":"get_weather","tool_call_id":"toolu_1"}
	]`), &oa); err != nil {
		t.Fatal(err)
	}
	_, oaTurns := messagesToTurns(oa)
	oaPrompt := tm.RenderTools("", oaTurns, tools)

	an := anthropicReqFrom(t, `[
	  {"role":"user","content":[{"type":"text","text":"Weather in Paris?"}]},
	  {"role":"assistant","content":[{"type":"thinking","thinking":"need the weather","signature":""},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"18C"}]}
	]`)
	_, anTurns, _ := anthropicTurns(an)
	anPrompt := tm.RenderTools("", anTurns, tools)

	const block = "<|im_start|>assistant\n<think>\nneed the weather\n</think>\n\n<tool_call>"
	for name, p := range map[string]string{"OpenAI": oaPrompt, "Anthropic": anPrompt} {
		if !strings.Contains(p, block) {
			t.Errorf("%s-shaped replay lost the reasoning of the tool loop in progress:\n%s", name, p[max(0, len(p)-400):])
		}
	}
	if oaPrompt != anPrompt {
		t.Errorf("the two client shapes rendered different prompts for the same tool loop:\n OpenAI    %q\n Anthropic %q", oaPrompt, anPrompt)
	}

	// Claude Code's shape: the reminder text arrives IN the tool_result message. It must not strip the reasoning.
	withReminder := anthropicReqFrom(t, `[
	  {"role":"user","content":[{"type":"text","text":"Weather in Paris?"}]},
	  {"role":"assistant","content":[{"type":"thinking","thinking":"need the weather","signature":""},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"18C"},
	    {"type":"text","text":"<system-reminder>be brief</system-reminder>"}]}
	]`)
	_, rTurns, _ := anthropicTurns(withReminder)
	if p := tm.RenderTools("", rTurns, tools); !strings.Contains(p, block) {
		t.Errorf("a reminder beside the tool_result made the history rule treat it as a new query and drop the reasoning:\n%s", p[max(0, len(p)-500):])
	}

	// Before the last REAL query the reasoning is dropped (the template's own rule), however it arrived.
	next := append(append([]chat.Turn(nil), anTurns...), chat.Turn{Role: "assistant", Content: "It is 18C."}, chat.Turn{Role: "user", Content: "And Rome?"})
	if p := tm.RenderTools("", next, tools); strings.Contains(p, "need the weather") {
		t.Errorf("reasoning from a finished loop must not be replayed once a new user query arrives:\n%s", p[max(0, len(p)-500):])
	}

	// -thinking asis is the pre-thinking bytes: the reasoning is ignored there.
	if p := tm.WithThinking(chat.ThinkAsIs).RenderTools("", anTurns, tools); strings.Contains(p, "need the weather") {
		t.Error("-thinking asis must not render replayed reasoning")
	}
}

// Gemma 4 (byte-exact tool renderer): the tool loop in progress keeps each turn's reasoning as a thought channel.
func TestHistory_gemmaToolLoopKeepsReasoning(t *testing.T) {
	tm := templateFromGolden(t, "gemma-4-26b-a4b-it").WithThinking(chat.ThinkTemplate)
	tools := []chat.Tool{{Name: "get_weather", Description: "Weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}
	an := anthropicReqFrom(t, `[
	  {"role":"user","content":[{"type":"text","text":"Weather in Paris?"}]},
	  {"role":"assistant","content":[{"type":"thinking","thinking":"need the weather","signature":""},
	    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"18C"}]}
	]`)
	_, turns, _ := anthropicTurns(an)
	if p := tm.RenderTools("", turns, tools); !strings.Contains(p, "<|channel>thought\nneed the weather\n<channel|><|tool_call>") {
		t.Errorf("Gemma tool loop lost its reasoning channel:\n%s", p)
	}
}

// A reply to a Gemma 4 thinking-on prompt that ends after a tool response starts INSIDE an open thought channel, so the
// splitter must start forced-open or the reasoning leaks into content (found by the history goldens).
func TestSplitter_gemmaToolResponsePromptIsForcedOpen(t *testing.T) {
	on := templateFromGolden(t, "gemma-4-26b-a4b-it").WithThinking(chat.ThinkOn)
	toolLast := []chat.Turn{{Role: "user", Content: "q"}, {Role: "assistant", ToolCalls: []chat.ToolCall{{Name: "f"}}}, {Role: "tool", Content: "r"}}
	userLast := []chat.Turn{{Role: "user", Content: "q"}}
	if !on.PromptOpensThinkFor(toolLast) {
		t.Error("thinking on, last turn a tool response: the prompt ends inside an open thought channel")
	}
	if on.PromptOpensThinkFor(userLast) {
		t.Error("thinking on, last turn a user message: the model opens the channel itself")
	}
	off := templateFromGolden(t, "gemma-4-26b-a4b-it").WithThinking(chat.ThinkOff)
	if off.PromptOpensThinkFor(toolLast) {
		t.Error("thinking off never ends inside an open channel")
	}
	r, c := func() (string, string) {
		sp := on.NewReasoningSplitter(toolLast)
		r1, c1 := sp.Push("checking the units\n<channel|>It is 18C.")
		r2, c2 := sp.Flush()
		return r1 + r2, c1 + c2
	}()
	if r != "checking the units" || c != "It is 18C." {
		t.Errorf("forced-open Gemma split: reasoning=%q content=%q", r, c)
	}
}
