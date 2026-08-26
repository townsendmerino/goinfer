package chat

import (
	"strings"
	"testing"
)

// G21 — StreamableLen is the whole safety argument for incremental tool-call
// streaming, so it is tested as a property, not by example alone: whatever it
// releases must be a prefix of the prose the parser will later call the "lead",
// no matter how the output is chopped into chunks.

func TestStreamableLen_holdsBackPartialOpeners(t *testing.T) {
	const opener = "<tool_call>"
	cases := []struct {
		pending string
		want    int
		why     string
	}{
		{"hello", 5, "no opener byte in sight — all prose"},
		{"hello<", 5, "trailing '<' could begin the opener"},
		{"hello<tool_", 5, "a longer partial opener is held too"},
		{"hello<tool_call", 5, "one byte short of the opener — still held"},
		{"hello<tool_call>", 5, "opener complete: prose ends exactly at it"},
		{"hello<tool_call>{\"n\":1}", 5, "content after the opener is never prose"},
		{"<tool_call>", 0, "opener at position 0: no prose at all"},
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

// The property that matters: feed an output one byte at a time (the worst case
// for a multi-byte opener) and the concatenation of everything released must be
// exactly the prose before the opener — never more, never reordered.
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
		if !strings.HasPrefix(want, released.String()) {
			t.Errorf("bytewise over %q: released is not a prefix of the lead", tc.out)
		}
	}
}

// THE gate: for every streamable family, streaming a generation one byte at a
// time must release exactly a PREFIX of the lead ParseToolCalls computes over
// the whole output — for prose-only outputs, prose-then-call, and the whitespace
// shapes that made the naive "raw prefix" design wrong.
//
// This test is why the design changed. An earlier version declared these families
// streamable on the assumption that lead was the raw untrimmed prefix; it is
// strings.TrimSpace(lead), and this caught that before any byte could be emitted
// that the parser would later disagree with.
func TestProseStreamerMatchesParser(t *testing.T) {
	families := map[string]*Template{"chatml": ChatML(), "mellum2": Mellum2(), "gemma4": Gemma4()}
	for name, tmpl := range families {
		opener, ok := tmpl.ToolCallOpener()
		if !ok {
			t.Fatalf("%s: expected a streamable family", name)
		}
		body := `call:f{"a":1}`
		outs := []string{
			"plain prose, no call at all",
			"prose before " + opener + body,
			"prose before   " + opener + body, // trailing run before the opener
			"   leading space then prose",
			"   leading and trailing before a call   " + opener + body,
			opener + body, // no prose at all
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
				// Held-back bytes are legitimate (a trailing run with no follower),
				// but the remainder must be whitespace only — never dropped content.
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
