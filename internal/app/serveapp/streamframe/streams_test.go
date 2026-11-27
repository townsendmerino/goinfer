package streamframe

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRespStream_lifecycleAndSequenceNumbers(t *testing.T) {
	rec := httptest.NewRecorder()
	ss := NewSSEWriter(rec, rec)
	rs := NewRespStream(ss, "resp_1", nil)
	rs.ReasoningDelta("step 1")
	rs.Delta("hello ")
	msg := rs.CloseMessage("world")
	if msg["id"] != "resp_1-msg" {
		t.Fatalf("msg id = %v, want resp_1-msg", msg["id"])
	}
	rsn := rs.CloseReasoning()
	if rsn == nil || rsn["id"] != "resp_1-rs" {
		t.Fatalf("rsn = %v, want resp_1-rs", rsn)
	}
	text, err := ReasoningItemText(rsn["encrypted_content"].(string), nil, nil)
	if err != nil || text != "step 1" {
		t.Fatalf("ReasoningItemText = %q, %v; want 'step 1'", text, err)
	}
	rs.FunctionCall(map[string]any{"id": "fc_1", "type": "function_call", "name": "fn", "arguments": `{"a":1}`})
	body := rec.Body.String()
	for _, want := range []string{
		"event: response.output_item.added",
		"event: response.reasoning_text.delta",
		"event: response.reasoning_text.done",
		"event: response.output_text.delta",
		"event: response.function_call_arguments.delta",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("stream body missing %q", want)
		}
	}
}

func TestAnthropicStreams_thinkAndTextOrdering(t *testing.T) {
	rec := httptest.NewRecorder()
	ss := NewSSEWriter(rec, rec)
	th := NewAnthropicThink(ss, true, func(text string) string { return "sig:" + text })
	ts := NewAnthropicTextStream(ss, th)

	th.Push("hmm")
	ts.Push("   ") // leading whitespace held
	if ts.Open() {
		t.Fatal("leading whitespace should not open text block")
	}
	ts.Push("answer")
	if !ts.Open() || th.Offset() != 1 {
		t.Fatalf("expected open text block at offset 1, got open=%v offset=%d", ts.Open(), th.Offset())
	}
	nextIdx := ts.Finish("!", false)
	if nextIdx != 2 {
		t.Fatalf("nextIdx = %d, want 2", nextIdx)
	}
	if !strings.Contains(rec.Body.String(), `"signature":"sig:hmm"`) {
		t.Errorf("missing minted signature in stream: %s", rec.Body.String())
	}
}

func TestEmptyAnswerOf_classifications(t *testing.T) {
	out := &TurnOutcome{}
	if got := EmptyAnswerOf(out, nil, "stop", "<END>", 0, 3); got == nil || got.Reason != "stop_sequence" {
		t.Errorf("stop_sequence = %+v", got)
	}
	if got := EmptyAnswerOf(out, nil, "stop", "", 0, 0); got == nil || got.Reason != "no_output" {
		t.Errorf("no_output (0 tokens) = %+v", got)
	}
	if got := EmptyAnswerOf(&TurnOutcome{Answered: true}, nil, "stop", "", 0, 5); got != nil {
		t.Errorf("answered turn returned non-nil EmptyAnswer: %+v", got)
	}
	resp := MarkEmptyResponse(map[string]any{"status": "completed"}, &EmptyAnswer{Reason: "no_output", Detail: "none"})
	if resp["status"] != "incomplete" {
		t.Errorf("MarkEmptyResponse status = %v, want incomplete", resp["status"])
	}
	if _, err := ReasoningItemText(ReasoningBlobPrefix+"!!!", json.RawMessage(`[]`), nil); err == nil {
		t.Error("expected malformed v1 encrypted_content to return error")
	}
}
