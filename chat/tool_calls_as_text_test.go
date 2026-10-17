package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The three replies of tack's E1 pilot job 1 that came back as text (chat/testdata/tool_calls_as_text/README.md), and the plants that keep a
// prose example of a call prose. Qwen2.5's own template specifies {"name": <function-name>, "arguments": <args-json-object>} inside <tool_call>.

var pilotTools = []Tool{
	{Name: "task", Description: "Launch a subagent", Parameters: []byte(`{"type":"object"}`)},
	{Name: "read", Description: "Read a file", Parameters: []byte(`{"type":"object"}`)},
}

func pilotReply(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "tool_calls_as_text", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func chatMLTemplate() *Template { return &Template{name: "chatml"} }

func TestToolCallsAsText_realReplies(t *testing.T) {
	tm := chatMLTemplate()

	calls, lead := tm.ParseToolCallsFor(pilotReply(t, "pilot_task01_raw_newlines.txt"), pilotTools)
	if len(calls) != 1 || calls[0].Name != "task" || lead != "" {
		t.Fatalf("task 1 (raw newlines in a string): calls %+v lead %q; want one call to task", calls, lead)
	}
	var args struct {
		Description  string `json:"description"`
		Prompt       string `json:"prompt"`
		SubagentType string `json:"subagent_type"`
	}
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("task 1 arguments are not JSON: %v: %s", err, calls[0].Arguments)
	}
	if !strings.Contains(args.Prompt, ":\n\n- WrapInt8(q8 []int8") {
		t.Errorf("task 1: the prompt's own newlines were not kept: %q", args.Prompt[:min(len(args.Prompt), 300)])
	}
	if args.SubagentType != "general" {
		t.Errorf("task 1: subagent_type %q", args.SubagentType)
	}

	calls, _ = tm.ParseToolCallsFor(pilotReply(t, "pilot_task02_function_key.txt"), pilotTools)
	if len(calls) != 1 || calls[0].Name != "task" {
		t.Fatalf("task 2 (\"function\" for \"name\"): calls %+v; want one call to task", calls)
	}
	if !strings.Contains(string(calls[0].Arguments), "case-insensitive") {
		t.Errorf("task 2 arguments: %s", calls[0].Arguments)
	}

	// Task 5: the markers are other tokens, so there is no block. The call between them is prose with a call in it, and stays prose.
	raw := pilotReply(t, "pilot_task05_garbled_markers.txt")
	calls, lead = tm.ParseToolCallsFor(raw, pilotTools)
	if len(calls) != 0 || lead != strings.TrimSpace(raw) {
		t.Errorf("task 5 (garbled markers): calls %+v, lead changed=%v; want no call and the text untouched", calls, lead != strings.TrimSpace(raw))
	}
}

// Without the request's tool list nothing can vouch for a repaired name, so ParseToolCalls (no list) reads only what the strict reader read before.
func TestToolCallsAsText_needsTheToolList(t *testing.T) {
	for _, f := range []string{"pilot_task01_raw_newlines.txt", "pilot_task02_function_key.txt"} {
		if calls, _ := chatMLTemplate().ParseToolCalls(pilotReply(t, f)); len(calls) != 0 {
			t.Errorf("%s: %d call(s) from ParseToolCalls without a tool list", f, len(calls))
		}
	}
}

func TestToolCallsAsText_variants(t *testing.T) {
	tm := chatMLTemplate()
	for _, c := range []struct{ name, in, tool string }{
		{"function as an OpenAI-shaped object", "<tool_call>\n{\"function\": {\"name\": \"read\", \"arguments\": {\"filePath\": \"a.go\"}}}\n</tool_call>", "read"},
		{"typed OpenAI call", "<tool_call>\n{\"type\": \"function\", \"function\": {\"name\": \"read\", \"arguments\": \"{\\\"filePath\\\": \\\"a.go\\\"}\"}}\n</tool_call>", "read"},
		{"function string, parameters key", "<tool_call>\n{\"function\": \"read\", \"parameters\": {\"filePath\": \"a.go\"}}\n</tool_call>", "read"},
		{"no closing tag, raw newline", "<tool_call>\n{\"function\": \"read\", \"arguments\": {\"filePath\": \"a\nb.go\"}}", "read"},
		{"prose lead", "Reading it.\n<tool_call>\n{\"function\": \"read\", \"arguments\": {}}\n</tool_call>", "read"},
		{"tab and carriage return in a string", "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\t\rb\"}}\n</tool_call>", "read"},
	} {
		calls, _ := tm.ParseToolCallsFor(c.in, pilotTools)
		if len(calls) != 1 || calls[0].Name != c.tool {
			t.Errorf("%s: calls %+v; want one call to %s", c.name, calls, c.tool)
			continue
		}
		var m map[string]any
		if err := json.Unmarshal(calls[0].Arguments, &m); err != nil {
			t.Errorf("%s: arguments are not a JSON object: %s", c.name, calls[0].Arguments)
		}
	}
}

// The plants: every one of these is text that mentions or imitates a call, and none may become one. They are the check that the repair stays inside a
// `<tool_call>` block, on an offered name, on a body that is one JSON object.
func TestToolCallsAsText_proseStaysProse(t *testing.T) {
	tm := chatMLTemplate()
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
		if calls, _ := tm.ParseToolCallsFor(c.in, pilotTools); len(calls) != 0 {
			t.Errorf("%s: read as a call: %+v", c.name, calls)
		}
	}
}

func TestEscapeControlInStrings(t *testing.T) {
	for in, want := range map[string]string{
		"{\"a\": \"x\ny\"}":      "{\"a\": \"x\\ny\"}",
		"{\n\"a\": 1}":           "{\n\"a\": 1}", // whitespace outside a string is left alone
		`{"a": "q\"\n"}`:         `{"a": "q\"\n"}`,
		"{\"a\": \"\x01\"}":      `{"a": "\u0001"}`,
		`{"a": "\\"}` + "\n":     `{"a": "\\"}` + "\n", // an escaped backslash does not escape the quote after it
		"{\"a\": \"t\tr\rn\n\"}": `{"a": "t\tr\rn\n"}`,
	} {
		if got := escapeControlInStrings(in); got != want {
			t.Errorf("escapeControlInStrings(%q) = %q, want %q", in, got, want)
		}
	}
}
