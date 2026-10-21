package constrain

import (
	"encoding/json"
	"os"
	"testing"
)

// opencodeKenTools is the tool list a real agent sends: opencode 1.19.0's built-in tools and ken-mcp v1.9.0's, as tack's
// E1 pilot captured them. Several use keywords the strict compiler refuses (maximum, exclusiveMinimum, an untyped
// property), which used to fail the whole union and leave every call unconstrained.
func opencodeKenTools(t *testing.T) []ToolSpec {
	t.Helper()
	raw, err := os.ReadFile("testdata/opencode-ken-tools.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Tools []struct {
			Function struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var specs []ToolSpec
	for _, tl := range doc.Tools {
		specs = append(specs, ToolSpec{Name: tl.Function.Name, Parameters: tl.Function.Parameters})
	}
	return specs
}

// TestToolCallsGrammar_opencodeTools: the union compiles over opencode's and ken-mcp's tools, so a call to them is
// constrained. It accepts well-formed calls, including to a tool whose schema the strict compiler refuses, and refuses
// the two malformed calls tack's E1 pilot received: a raw newline inside a string argument, and "function" where the
// call's key is "name".
func TestToolCallsGrammar_opencodeTools(t *testing.T) {
	specs := opencodeKenTools(t)
	strictFails := 0
	for _, s := range specs {
		if _, err := JSONSchema(s.Parameters); err != nil {
			strictFails++
		}
	}
	if strictFails == 0 {
		t.Fatal("no tool in the list fails the strict compiler: the test no longer covers what it was written for")
	}
	u, err := ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, specs)
	if err != nil {
		t.Fatalf("the union over %d tools (%d refused by the strict compiler): %v", len(specs), strictFails, err)
	}
	call := func(body string) string { return "<tool_call>\n" + body + "\n</tool_call>" }
	for _, ok := range []string{
		`{"name": "read", "arguments": {"filePath": "/repo/cmd/serve/main.go", "limit": 2000}}`,
		`{"name": "bash", "arguments": {"command": "go test ./...", "timeout": 120000, "workdir": "/repo"}}`,
		`{"name": "task", "arguments": {"description": "d", "prompt": "line one\nline two", "subagent_type": "general"}}`,
		`{"name": "ken_search", "arguments": {"query": "prepare", "languages": ["go"]}}`,
	} {
		if !accepts(u, call(ok)) {
			t.Errorf("a well-formed call was refused: %s", ok)
		}
	}
	for what, bad := range map[string]string{
		"a raw newline inside a string": "{\"name\": \"task\", \"arguments\": {\"description\": \"d\", \"prompt\": \"line one\nline two\", \"subagent_type\": \"general\"}}",
		`"function" for "name"`:         `{"function": "task", "arguments": {"description": "d", "prompt": "p", "subagent_type": "general"}}`,
		"an undeclared tool":            `{"name": "rm_rf", "arguments": {}}`,
	} {
		if accepts(u, call(bad)) {
			t.Errorf("%s was accepted", what)
		}
	}
}
