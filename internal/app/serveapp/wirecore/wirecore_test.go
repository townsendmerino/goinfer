package wirecore

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestWireCore(t *testing.T) {
	// Stop holdback and UTF-8 rune boundary.
	if got := StopTailHold("hello EN", []string{"END"}); got != 2 {
		t.Fatalf("StopTailHold = %d, want 2", got)
	}
	if i, which, ok := FirstStop("a<|end|>b", []string{"<|end|>"}); !ok || i != 1 || which != "<|end|>" {
		t.Fatalf("FirstStop = (%d, %q, %v), want (1, <|end|>, true)", i, which, ok)
	}
	euro := "a€"
	if got := CompleteUTF8(euro[:len(euro)-1]); got != 1 {
		t.Fatalf("CompleteUTF8(partial euro) = %d, want 1", got)
	}
	if got := CompleteUTF8(euro); got != len(euro) {
		t.Fatalf("CompleteUTF8(full euro) = %d, want %d", got, len(euro))
	}

	// Developer/system role handling in MessagesToTurns.
	sys, turns := MessagesToTurns([]ChatMessage{
		{Role: "developer", Content: RawStr("rule 1")},
		{Role: "system", Content: RawStr("rule 2")},
		{Role: "user", Content: RawStr("hi")},
		{Role: "system", Content: RawStr("mid reminder")},
	})
	if sys != "rule 1\n\nrule 2" || len(turns) != 2 || !turns[1].ToolLoop {
		t.Fatalf("MessagesToTurns = (%q, %+v)", sys, turns)
	}

	// Flat vs nested ToolSpec unmarshaling.
	var flat ToolSpec
	if err := json.Unmarshal([]byte(`{"type":"function","name":"exec","description":"run"}`), &flat); err != nil || flat.Function.Name != "exec" {
		t.Fatalf("flat ToolSpec = %+v, err %v", flat, err)
	}

	// Overflow helpless vs compactable messaging.
	compact := &Overflow{Prompt: 500, Ctx: 400, ModelWindow: 1024, Fixed: 100, Reserved: 100}
	if compact.Helpless() || !strings.Contains(compact.Message("openai"), "context_length_exceeded") {
		t.Fatalf("compactable overflow: helpless=%v msg=%q", compact.Helpless(), compact.Message("openai"))
	}
	helpless := &Overflow{Prompt: 500, Ctx: 400, ModelWindow: 400, Fixed: 350, Reserved: 100}
	if !helpless.Helpless() || strings.Contains(helpless.Message("openai"), "context_length_exceeded") {
		t.Fatalf("helpless overflow: helpless=%v msg=%q", helpless.Helpless(), helpless.Message("openai"))
	}
}
