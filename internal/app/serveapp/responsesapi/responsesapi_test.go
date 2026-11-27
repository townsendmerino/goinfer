package responsesapi

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

func TestInputToMessagesAndToolOutputs(t *testing.T) {
	msgs, err := InputToMessages(json.RawMessage(`[
		{"role":"user","content":"weather?"},
		{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
		{"type":"function_call_output","call_id":"c1","output":{"temp":18}}
	]`))
	if err != nil || len(msgs) != 3 {
		t.Fatalf("InputToMessages = %+v, %v", msgs, err)
	}
	if msgs[1].Role != "assistant" || len(msgs[1].ToolCalls) != 1 || msgs[1].ToolCalls[0].ID != "c1" {
		t.Fatalf("assistant tool call = %+v", msgs[1])
	}
	if msgs[2].Role != "tool" || msgs[2].Text() != `{"temp":18}` {
		t.Fatalf("tool output = %+v", msgs[2])
	}

	out, calls := BuildToolOutputs("resp_1", []chat.ToolCall{{Name: "ping", Arguments: json.RawMessage(`{}`)}}, "", "checking", nil, func() string { return "gen1" })
	if len(out) != 2 || len(calls) != 1 || calls[0].ID != "call_gen1" {
		t.Fatalf("BuildToolOutputs = %+v, %+v", out, calls)
	}
}

func TestReqFastAndStatus(t *testing.T) {
	n := 64
	r := Req{
		MaxOutputTokens: &n,
		Reasoning: &struct {
			Effort string `json:"effort"`
		}{Effort: "high"},
	}
	r.ApplyFast()
	if tr := r.ThinkRequest(); tr.ReasoningEffort != "none" {
		t.Fatalf("ThinkRequest after ApplyFast = %+v", tr)
	}
	sm := r.SamplingSettings()
	if sm.MaxTokens == nil || *sm.MaxTokens != 64 || !sm.OutputByWindow {
		t.Fatalf("SamplingSettings = %+v", sm)
	}
	if Status("length") != "incomplete" || Status("cancelled") != "cancelled" || Status("stop") != "completed" {
		t.Fatalf("unexpected Status mapping")
	}
	obj := WithCachedTokens(Object("resp_1", "m", 10, "incomplete", []any{}, 5, 3), 2)
	if obj["incomplete_details"] == nil {
		t.Fatalf("missing incomplete_details on incomplete response: %+v", obj)
	}
}
