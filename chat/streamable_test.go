package chat

import (
	"strings"
	"testing"
)

func TestProseStreamerMatchesParser(t *testing.T) {
	families := map[string]*Template{"chatml": ChatML(), "mellum2": Mellum2(), "gemma4": Gemma4(), "flint": Flint()}
	for name, tmpl := range families {
		opener, ok := tmpl.ToolCallOpener()
		if !ok {
			t.Fatalf("%s: expected a streamable family", name)
		}
		body := `call:f{"a":1}`
		outs := []string{
			"plain prose, no call at all",
			"prose before " + opener + body,
			"prose before   " + opener + body,
			"   leading space then prose",
			"   leading and trailing before a call   " + opener + body,
			opener + body,
			"interior  double  spaces stay",
			"",
		}
		for _, out := range outs {
			_, lead := tmpl.ParseToolCalls(out)
			ps := NewProseStreamer(opener)
			var released strings.Builder
			for i := 0; i < len(out); i++ {
				released.WriteString(ps.Push(out[i : i+1]))
			}
			got := released.String()
			if !strings.HasPrefix(lead, got) {
				t.Errorf("%s: streamed %q is NOT a prefix of the parsed lead %q (out=%q)", name, got, lead, out)
			}
			if got != lead {
				if strings.TrimSpace(lead[len(got):]) != "" {
					t.Errorf("%s: streamer withheld non-whitespace %q (out=%q)", name, lead[len(got):], out)
				}
			}
		}
	}
	for name, tmpl := range map[string]*Template{"mistral": Mistral(), "llama3": Llama3()} {
		if _, ok := tmpl.ToolCallOpener(); ok {
			t.Errorf("%s is declared streamable, but its parser normalizes the lead beyond a prefix", name)
		}
	}
}
