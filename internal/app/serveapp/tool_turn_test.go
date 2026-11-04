package serveapp

import (
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// The shared tool turn's reconciliation, driven by the real prose streamer and the real ChatML
// parser over whole outputs fed in chunks — the path /v1/chat/completions, /v1/responses and
// /v1/messages all take (tool_turn.go). For every output: what streamed plus rest is the prose the
// turn settles on, and no byte after the call opener ever streamed.
func TestReconcileProse_streamedPlusRestIsTheProse(t *testing.T) {
	tmpl := chat.ChatML()
	opener, ok := tmpl.ToolCallOpener()
	if !ok {
		t.Fatal("chatml is expected to stream prose (ToolCallOpener)")
	}
	call := "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	for _, c := range []struct{ name, out string }{
		{"prose only", "It is sunny in Paris today."},
		{"prose opening with a newline (no call)", "\nHello there."},
		{"prose then call", "Let me check that.\n" + call},
		{"whitespace-padded prose then call", "  \n Let me check.  \n\n" + call},
		{"call only", call},
		{"call only after a newline", "\n" + call},
	} {
		for _, chunk := range []int{1, 3, 7, 1000} {
			ps := chat.NewProseStreamer(opener)
			var streamed strings.Builder
			for i := 0; i < len(c.out); i += chunk {
				streamed.WriteString(ps.Push(c.out[i:min(i+chunk, len(c.out))]))
			}
			calls, parsed := tmpl.ParseToolCalls(c.out)
			lead, rest, err := reconcileProse(c.out, len(calls) > 0, parsed, streamed.String())
			if err != nil {
				t.Fatalf("%s (chunk %d): %v — streamed %q", c.name, chunk, err, streamed.String())
			}
			if strings.Contains(streamed.String(), "<tool_call>") {
				t.Errorf("%s (chunk %d): call syntax streamed as prose: %q", c.name, chunk, streamed.String())
			}
			sent := streamed.String() + rest
			if len(calls) > 0 && sent != lead {
				t.Errorf("%s (chunk %d): streamed %q + rest %q != lead %q", c.name, chunk, streamed.String(), rest, lead)
			}
			if len(calls) == 0 && strings.TrimSpace(sent) != strings.TrimSpace(c.out) {
				t.Errorf("%s (chunk %d): streamed+rest %q lost prose from %q", c.name, chunk, sent, c.out)
			}
		}
	}
}

func TestReconcileProse_divergenceIsReported(t *testing.T) {
	if _, _, err := reconcileProse("abc", true, "abc", "xyz"); !errors.Is(err, errProseDiverged) {
		t.Errorf("streamed bytes that are not a prefix of the lead must report errProseDiverged, got %v", err)
	}
	lead, rest, err := reconcileProse("\n raw", false, "raw", "")
	if err != nil || lead != "\n raw" || rest != "\n raw" {
		t.Errorf("nothing streamed: lead and rest are the raw output unchanged, got %q / %q / %v", lead, rest, err)
	}
}

// -lenient-tool-calls through the shared tool turn: the config flag reaches the template (tuneTemplate), the fence-aware
// streamer holds the fence back, and what streamed plus rest is exactly the prose the turn settles on, for the two real
// cold-user replies (a call), for a demonstration (prose: the fence must still be DELIVERED, at the end), and with the
// flag off (the same fenced reply is prose and streams whole).
func TestReconcileProse_lenientFencedCalls(t *testing.T) {
	tools := []chat.Tool{{Name: "edit", Parameters: []byte(`{"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"}},"required":["filePath","oldString","newString"]}`)}}
	const fence = "```json\n{\"name\": \"edit\", \"arguments\": {\"filePath\": \"calc.go\", \"oldString\": \"return a - b\", \"newString\": \"return a + b\"}}\n```"
	const lead1 = "The `add` function should subtract instead of add. I'll update the function and verify it."
	cases := []struct {
		name, out  string
		on, isCall bool
		wantLead   string
	}{
		{"attempt 2: the whole reply is the fence", fence, true, true, ""},
		{"attempt 1: prose, then the fence", lead1 + "\n" + fence, true, true, lead1},
		{"a demonstration with an explanation after the fence", "Here is an example:\n" + fence + "\nThis edits the file.", true, false, ""},
		{"the same fenced reply with the flag off", lead1 + "\n" + fence, false, false, ""},
	}
	for _, c := range cases {
		cfg := config{lenientToolCalls: c.on}
		tmpl := cfg.tuneTemplate(chat.ChatML())
		if tmpl.LenientToolCalls() != c.on {
			t.Fatalf("%s: tuneTemplate(lenientToolCalls=%v) gave LenientToolCalls()=%v", c.name, c.on, tmpl.LenientToolCalls())
		}
		opener, _ := tmpl.ToolCallOpener()
		for _, chunk := range []int{1, 4, 11, 1000} {
			ps := chat.NewBareAwareProseStreamer(opener)
			if tmpl.LenientToolCalls() {
				ps.FenceAware()
			}
			var streamed strings.Builder
			for i := 0; i < len(c.out); i += chunk {
				streamed.WriteString(ps.Push(c.out[i:min(i+chunk, len(c.out))]))
			}
			calls, parsed := tmpl.ParseToolCallsFor(c.out, tools)
			if (len(calls) > 0) != c.isCall {
				t.Fatalf("%s: calls = %+v, want a call: %v", c.name, calls, c.isCall)
			}
			lead, rest, err := reconcileProse(c.out, len(calls) > 0, parsed, streamed.String())
			if err != nil {
				t.Fatalf("%s (chunk %d): %v — streamed %q", c.name, chunk, err, streamed.String())
			}
			sent := streamed.String() + rest
			if c.isCall {
				if sent != c.wantLead || lead != c.wantLead {
					t.Errorf("%s (chunk %d): streamed %q + rest %q, lead %q; want the lead %q", c.name, chunk, streamed.String(), rest, lead, c.wantLead)
				}
				if strings.Contains(sent, "```") {
					t.Errorf("%s (chunk %d): the fence reached the client as prose: %q", c.name, chunk, sent)
				}
			} else if strings.TrimSpace(sent) != strings.TrimSpace(c.out) {
				t.Errorf("%s (chunk %d): streamed %q + rest %q lost prose from %q", c.name, chunk, streamed.String(), rest, c.out)
			}
		}
	}
}
