package chat

import "testing"

// Text that mentions or imitates a tool call must come back as prose, not a call. These are the cases a tolerant reader of a
// `<tool_call>` block (a "function" key for "name", raw newlines inside a string) must not widen the parser into: the reader
// is strict today, and each case here stays prose under it. Written for the repair that was not needed (the tool-call constraint
// armed on every request in the 2026-10-17 pilot re-run); they stand as the guard for whoever adds that tolerance later.

var proseTools = []Tool{
	{Name: "task", Description: "Launch a subagent", Parameters: []byte(`{"type":"object"}`)},
	{Name: "read", Description: "Read a file", Parameters: []byte(`{"type":"object"}`)},
}

func TestToolCalls_proseStaysProse(t *testing.T) {
	tm := &Template{name: "chatml"}
	for _, c := range []struct{ name, in string }{
		{"a prose example of a call", "You can read a file like this: {\"name\": \"read\", \"arguments\": {\"filePath\": \"a.go\"}} and then continue."},
		{"a prose example with the function key", "Call it as {\"function\": \"read\", \"arguments\": {\"filePath\": \"a.go\"}}."},
		{"a bare call with a raw newline, no tags", "{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\nb.go\"}}"},
		{"a bare function-key call, no tags", "{\"function\": \"read\", \"arguments\": {}}"},
		{"a fenced example", "Use:\n```json\n{\"function\": \"read\", \"arguments\": {\"filePath\": \"a.go\"}}\n```\n"},
		{"the tag named in prose with a placeholder body", "Wrap it as <tool_call>{...}</tool_call> and I will run it."},
		{"an unoffered tool, function key, in tags", "<tool_call>\n{\"function\": \"rm_rf\", \"arguments\": {}}\n</tool_call>"},
		{"an unoffered tool, raw newline, in tags", "<tool_call>\n{\"name\": \"rm_rf\", \"arguments\": {\"p\": \"a\nb\"}}\n</tool_call>"},
		{"a truncated body", "<tool_call>\n{\"function\": \"read\", \"arguments\": {\"filePath\": \"a.go"},
		{"a trailing comma", "<tool_call>\n{\"function\": \"read\", \"arguments\": {\"filePath\": \"a.go\",}}\n</tool_call>"},
		{"arguments that are not an object", "<tool_call>\n{\"function\": \"read\", \"arguments\": [1, 2]}\n</tool_call>"},
		{"a function key that is neither a string nor a call", "<tool_call>\n{\"function\": 7, \"arguments\": {}}\n</tool_call>"},
	} {
		if calls, _ := tm.ParseToolCallsFor(c.in, proseTools); len(calls) != 0 {
			t.Errorf("%s: read as a call: %+v", c.name, calls)
		}
	}
}
