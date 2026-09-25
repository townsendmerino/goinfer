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
