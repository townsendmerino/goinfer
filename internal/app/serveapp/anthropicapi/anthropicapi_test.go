package anthropicapi

import (
	"encoding/json"
	"testing"
)

func TestAnthropicAPI(t *testing.T) {
	req := &Req{
		System: json.RawMessage(`"be concise"`),
		Messages: []Message{
			{Role: "user", Content: json.RawMessage(`"first"`)},
			{Role: "system", Content: json.RawMessage(`"mid-turn note"`)},
			{Role: "assistant", Content: json.RawMessage(`"ok"`)},
		},
	}
	sys, turns, aerr := Turns(req)
	if aerr != nil {
		t.Fatalf("Turns: %v", aerr)
	}
	if sys != "be concise" || len(turns) != 2 || turns[0].Content != "first\nmid-turn note" || turns[1].Content != "ok" {
		t.Fatalf("Turns = (%q, %+v)", sys, turns)
	}

	if reason, seq := StopReason("stop", "END"); reason != "stop_sequence" || seq != "END" {
		t.Fatalf("StopReason = (%q, %v)", reason, seq)
	}
	u := Usage(100, 40, 25)
	if u["input_tokens"] != 60 || u["cache_read_input_tokens"] != 40 || u["output_tokens"] != 25 {
		t.Fatalf("Usage = %+v", u)
	}
}
