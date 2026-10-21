package serveapp

import "testing"

// The closed list (idsFormat) on its own, with no tokenizer: what it lets through and what stays content. The Mellum2.1
// cases themselves are TestIDsCarried_templateFormat, which needs the per-machine tokenizer.
func TestIDsFormat_align(t *testing.T) {
	f := idsFormat{rOpen: "<think>", rClose: "</think>", cOpen: "<tool_call>", cClose: "</tool_call>"}
	call := func(obj string) string { return "<tool_call>\n" + obj + "\n</tool_call>" }
	for _, tc := range []struct {
		name, d, t string
		ok         bool
	}{
		{"a newline the template adds before the closing delimiter", "<think>\nNow</think>\n\nok", "<think>\nNow\n</think>\n\nok", true},
		{"a newline the template drops after the opening delimiter", "<think>\n\nNow\n</think>", "<think>\nNow\n</think>", true},
		{"the template's space after a call's key", call(`{"name": "f", "arguments":{"cmd":"a b"}}`), call(`{"name": "f", "arguments": {"cmd":"a b"}}`), true},
		{"separators re-spaced throughout a call", call(`{"name":"f","arguments":{"a":[1,2]}}`), call(`{"name": "f", "arguments": {"a": [1, 2]}}`), true},
		{"identical", "<think>\nx\n</think>", "<think>\nx\n</think>", true},

		{"a space inside an argument string", call(`{"arguments":{"cmd":"a b"}}`), call(`{"arguments":{"cmd":"a  b"}}`), false},
		{"a newline inside an argument string", call(`{"arguments":{"cmd":"ab"}}`), call(`{"arguments":{"cmd":"a\nb"}}`), false},
		{"a number split by a space", call(`{"arguments":{"n":12}}`), call(`{"arguments":{"n":1 2}}`), false},
		{"a space, not a newline, before the closing delimiter", "<think>\nNow</think>", "<think>\nNow </think>", false},
		{"a newline inside the reasoning, away from the delimiters", "<think>\na b\n</think>", "<think>\na\nb\n</think>", false},
		{"JSON spacing outside a call", `{"a":1}`, `{"a": 1}`, false},
		{"JSON spacing after a call has closed", call(`{"a":1}`) + `{"b":2}`, call(`{"a":1}`) + `{"b": 2}`, false},
		{"a one-character content edit", "<think>\nNow</think>", "<think>\nNox\n</think>", false},
		{"a difference before the first byte", "<think>\nNow</think>", "\n<think>\nNow</think>", false},
	} {
		end, at, differs, ok := f.align(tc.d, tc.t)
		if ok != tc.ok {
			t.Errorf("%s: ok=%v, want %v", tc.name, ok, tc.ok)
			continue
		}
		if !ok {
			continue
		}
		if end != len(tc.t) || differs != (tc.d != tc.t) || len(at) != len(tc.d)+1 || at[len(tc.d)] != end {
			t.Errorf("%s: end %d of %d, differs %v, at %d entries", tc.name, end, len(tc.t), differs, len(at))
		}
		for i := range len(tc.d) { // every byte stands for an equal byte of t, or for a skipped whitespace run
			if c := tc.d[i]; tc.t[at[i]] != c && c != ' ' && c != '\n' {
				t.Errorf("%s: d[%d]=%q stands for t[%d]=%q", tc.name, i, c, at[i], tc.t[at[i]])
			}
		}
	}
}
