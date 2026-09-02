package serveapp

import (
	"encoding/json"
	"testing"
)

// M-18: A RESPONSES TOOL LOOP COULD NOT COMPLETE.
//
// `function_call` and `function_call_output` carry no `role` and no `content`, so decoding items as
// {role, content} turned both into `{Role:"user", Content:""}` — two empty user turns. The model
// never saw the tool result, so it answered without it or re-called the same tool forever, under
// HTTP 200. docs/server.md and responses.go both claim the round-trip; TestServe_responses step 4
// never feeds a result back, which is exactly why nothing caught it.
//
// This is the step-4 that was missing, at the decode layer where the loss happened.
func TestResponses_toolLoopItemsSurviveDecoding(t *testing.T) {
	input := `[
      {"role":"user","content":"weather in Paris?"},
      {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
      {"type":"function_call_output","call_id":"call_1","output":"18C and raining"}
    ]`
	msgs, err := responseInputToMessages(json.RawMessage(input))
	if err != nil {
		t.Fatalf("responseInputToMessages: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("got %d messages, want 3: %+v", len(msgs), msgs)
	}

	if msgs[0].Role != "user" || msgs[0].text() != "weather in Paris?" {
		t.Errorf("plain message turn changed: %+v", msgs[0])
	}

	// The model's own call must come back as an ASSISTANT turn carrying the tool call — that is
	// what the chat template renders. An empty user turn here is the before-state.
	call := msgs[1]
	if call.Role != "assistant" {
		t.Errorf("function_call became role %q, want assistant", call.Role)
	}
	if len(call.ToolCalls) != 1 {
		t.Fatalf("function_call carried %d tool calls, want 1 — the model's own call was dropped", len(call.ToolCalls))
	}
	if call.ToolCalls[0].ID != "call_1" || call.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool call lost its identity: %+v", call.ToolCalls[0])
	}
	if call.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("arguments = %q", call.ToolCalls[0].Function.Arguments)
	}

	// THE RESULT ITSELF. This is the turn whose loss made the loop livelock: the model asks, gets
	// nothing back, and asks again.
	out := msgs[2]
	if out.Role != "tool" {
		t.Errorf("function_call_output became role %q, want tool", out.Role)
	}
	if out.ToolCallID != "call_1" {
		t.Errorf("tool result lost its call_id: %q", out.ToolCallID)
	}
	if got := out.text(); got != "18C and raining" {
		t.Errorf("tool result text = %q, want %q — an empty tool turn IS the defect", got, "18C and raining")
	}
}

// The output field is typed loosely enough to arrive as something other than a string. Whatever it
// is, the turn must not come out empty: an empty tool turn is indistinguishable from the bug.
func TestResponses_toolOutputNeverDecodesToNothing(t *testing.T) {
	for name, raw := range map[string]string{
		"string":        `"plain text"`,
		"content parts": `[{"type":"text","text":"in parts"}]`,
		"object":        `{"temp_c":18}`,
		"number":        `42`,
	} {
		if got := toolOutputText(json.RawMessage(raw)); got == "" {
			t.Errorf("%s: output %s flattened to empty — the model would see a blank tool result", name, raw)
		}
	}
	if got := toolOutputText(nil); got != "" {
		t.Errorf("absent output should stay empty, got %q", got)
	}
}
