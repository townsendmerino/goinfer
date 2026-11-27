package chat

import (
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

// WithLenientToolCalls returns a copy of the template that also reads ONE fenced JSON call at the end of a reply
// as a tool call. Off by default, and a no-op for every family that does not accept a
// bare call (chatml, mellum2). It is a separate, opt-in rule because the shape it accepts is also what a model writes when
// asked to SHOW an example call; see fencedToolCall for how narrow it is made.
func (t *Template) WithLenientToolCalls(on bool) *Template {
	if t == nil || t.lenientFenced == on {
		return t
	}
	c := *t
	c.lenientFenced = on
	return &c
}

// LenientToolCalls reports whether WithLenientToolCalls is on for a family that honours it.
func (t *Template) LenientToolCalls() bool {
	return t != nil && t.lenientFenced && t.AcceptsBareToolCall()
}

// fencedToolCall reads out as a call when ALL of these hold (pre-registered in fenced_tool_call_test.go):
//   - the reply has exactly one fenced block and it is the last thing in it, so a demonstration that goes on to
//     explain itself, or any second block, is prose; the text before the fence is returned as the lead;
//   - the fence's tag is empty or json;
//   - it holds exactly one JSON object whose only keys are name, arguments (or parameters) and id;
//   - name is exactly a supplied tool's name, and arguments is an object that validates against that tool's schema.
//
// It cannot tell a shown call from a meant one when a demonstration ENDS on a single valid call; that residual is why
// the option is off by default.
func fencedToolCall(out string, tools []Tool) (ToolCall, string, bool) {
	trimmed := strings.TrimRight(out, " \t\r\n")
	if !strings.HasSuffix(trimmed, "```") || strings.Count(trimmed, "```") != 2 {
		return ToolCall{}, "", false
	}
	open := strings.Index(trimmed, "```")
	body := trimmed[open+3 : len(trimmed)-3]
	before, after, ok0 := strings.Cut(body, "\n")
	if !ok0 {
		return ToolCall{}, "", false
	}
	if tag := strings.TrimSpace(before); tag != "" && !strings.EqualFold(tag, "json") {
		return ToolCall{}, "", false
	}
	content := after
	var raw map[string]json.RawMessage
	dec := json.NewDecoder(strings.NewReader(content))
	if dec.Decode(&raw) != nil || raw == nil || strings.TrimSpace(content[dec.InputOffset():]) != "" {
		return ToolCall{}, "", false
	}
	for k := range raw {
		if k != "name" && k != "arguments" && k != "parameters" && k != "id" {
			return ToolCall{}, "", false
		}
	}
	var name string
	if json.Unmarshal(raw["name"], &name) != nil {
		return ToolCall{}, "", false
	}
	var tool *Tool
	for i := range tools {
		if tools[i].Name == name {
			tool = &tools[i]
			break
		}
	}
	if tool == nil {
		return ToolCall{}, "", false
	}
	args, ok := raw["arguments"]
	if !ok {
		if args, ok = raw["parameters"]; !ok {
			return ToolCall{}, "", false
		}
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(args, &obj) != nil || obj == nil || !chatcore.ArgsMatchSchema(obj, tool.Parameters) {
		return ToolCall{}, "", false
	}
	var id string
	if v, ok := raw["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	return ToolCall{ID: id, Name: name, Arguments: args}, strings.TrimSpace(trimmed[:open]), true
}
