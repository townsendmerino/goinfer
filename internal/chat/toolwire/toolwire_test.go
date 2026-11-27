package toolwire

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

func TestStreamableLen_holdsBackPartialOpeners(t *testing.T) {
	const opener = "<tool_call>"
	cases := []struct {
		pending string
		want    int
		why     string
	}{
		{"hello", 5, "no opener byte in sight"},
		{"hello<", 5, "trailing '<' could begin the opener"},
		{"hello<tool_", 5, "a longer partial opener is held too"},
		{"hello<tool_call", 5, "one byte short of the opener"},
		{"hello<tool_call>", 5, "opener complete: prose ends at it"},
		{"hello<tool_call>{\"n\":1}", 5, "content after the opener is never prose"},
		{"<tool_call>", 0, "opener at position 0"},
		{"", 0, "empty"},
		{"a<b", 3, "'<' followed by a non-matching byte is ordinary text"},
		{"<<tool_call>", 1, "the first '<' is prose; the opener starts at 1"},
	}
	for _, c := range cases {
		if got := StreamableLen(c.pending, opener); got != c.want {
			t.Errorf("StreamableLen(%q) = %d, want %d — %s", c.pending, got, c.want, c.why)
		}
	}
	if got := StreamableLen("anything at all", ""); got != len("anything at all") {
		t.Errorf("empty opener must never hold back: got %d", got)
	}
}

func TestStreamableLen_bytewiseNeverOverruns(t *testing.T) {
	for _, tc := range []struct{ out, opener string }{
		{"some prose<tool_call>{\"name\":\"f\"}</tool_call>", "<tool_call>"},
		{"<tool_call>{}", "<tool_call>"},
		{"no calls here at all", "<tool_call>"},
		{"tricky < < <t <to <tool_call not yet<tool_call>x", "<tool_call>"},
		{"gemma<|tool_call>call:f{}", "<|tool_call>"},
		{"partial <|tool_ and more", "<|tool_call>"},
	} {
		want := tc.out
		if i := strings.Index(tc.out, tc.opener); i >= 0 {
			want = tc.out[:i]
		}
		var released, pending strings.Builder
		started := false
		for i := 0; i < len(tc.out); i++ {
			if started {
				break
			}
			pending.WriteByte(tc.out[i])
			p := pending.String()
			n := StreamableLen(p, tc.opener)
			released.WriteString(p[:n])
			if strings.Contains(p, tc.opener) {
				started = true
				continue
			}
			pending.Reset()
			pending.WriteString(p[n:])
		}
		if released.String() != want {
			t.Errorf("bytewise over %q: released %q, want %q", tc.out, released.String(), want)
		}
	}
}

func TestFencedToolCall_schemaValidated(t *testing.T) {
	tools := []chatcore.Tool{{
		Name:       "read",
		Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}}
	call, lead, ok := FencedToolCall("Let me read that.\n```json\n{\"name\":\"read\",\"arguments\":{\"path\":\"main.go\"}}\n```\n", tools)
	if !ok || call.Name != "read" || lead != "Let me read that." {
		t.Fatalf("FencedToolCall = %+v, %q, %v", call, lead, ok)
	}
	if _, _, ok := FencedToolCall("```json\n{\"name\":\"read\",\"arguments\":{}}\n```\nMore text after.", tools); ok {
		t.Fatal("trailing prose or missing required property must not parse as a fenced call")
	}
}
