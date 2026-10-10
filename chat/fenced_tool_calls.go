package chat

import (
	"bytes"
	"encoding/json"
	"strings"
)

// WithLenientToolCalls returns a copy of the template that also reads ONE fenced JSON call at the end of a reply
// as a tool call (docs/queue-correctness.md G39). Off by default, and a no-op for every family that does not accept a
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
		return ToolCall{}, "", false // not JSON, not an object, or anything but whitespace after the one object
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
	if json.Unmarshal(args, &obj) != nil || obj == nil || !argsMatchSchema(obj, tool.Parameters) {
		return ToolCall{}, "", false
	}
	var id string
	if v, ok := raw["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	return ToolCall{ID: id, Name: name, Arguments: args}, strings.TrimSpace(trimmed[:open]), true
}

// argsMatchSchema is the part of JSON Schema that a tool's `parameters` object uses in practice: every `required`
// property is present, every present property named in `properties` has the declared `type` (string, number,
// integer, boolean, array, object, null, or a list of them) and sits inside its `enum`, and a property `properties`
// does not name is refused only when `additionalProperties` is false. A schema with none of these constrains nothing,
// which is what an absent or empty `parameters` means. Nested objects are checked one level (type), not recursively.
func argsMatchSchema(obj map[string]json.RawMessage, schema json.RawMessage) bool {
	if len(bytes.TrimSpace(schema)) == 0 {
		return true
	}
	var sc struct {
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
		AdditionalProperties json.RawMessage            `json:"additionalProperties"`
	}
	if json.Unmarshal(schema, &sc) != nil {
		return true // not an object schema we can read: do not refuse on it
	}
	for _, r := range sc.Required {
		if _, ok := obj[r]; !ok {
			return false
		}
	}
	forbidExtra := strings.TrimSpace(string(sc.AdditionalProperties)) == "false"
	for k, v := range obj {
		ps, declared := sc.Properties[k]
		if !declared {
			if forbidExtra {
				return false
			}
			continue
		}
		var p struct {
			Type json.RawMessage   `json:"type"`
			Enum []json.RawMessage `json:"enum"`
		}
		if json.Unmarshal(ps, &p) != nil {
			continue
		}
		if !valueHasType(v, p.Type) {
			return false
		}
		if len(p.Enum) > 0 {
			in := false
			for _, e := range p.Enum {
				if fencedJSONEqual(e, v) {
					in = true
					break
				}
			}
			if !in {
				return false
			}
		}
	}
	return true
}

func fencedJSONEqual(a, b json.RawMessage) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	xb, _ := json.Marshal(x)
	yb, _ := json.Marshal(y)
	return bytes.Equal(xb, yb)
}

// valueHasType checks v against a schema `type`, which is a name or a list of names; an absent type matches anything.
func valueHasType(v, typ json.RawMessage) bool {
	if len(bytes.TrimSpace(typ)) == 0 {
		return true
	}
	var names []string
	var one string
	if json.Unmarshal(typ, &one) == nil {
		names = []string{one}
	} else if json.Unmarshal(typ, &names) != nil {
		return true
	}
	var x any
	if json.Unmarshal(v, &x) != nil {
		return false
	}
	for _, n := range names {
		switch n {
		case "string":
			if _, ok := x.(string); ok {
				return true
			}
		case "number":
			if _, ok := x.(float64); ok {
				return true
			}
		case "integer":
			if f, ok := x.(float64); ok && f == float64(int64(f)) {
				return true
			}
		case "boolean":
			if _, ok := x.(bool); ok {
				return true
			}
		case "array":
			if _, ok := x.([]any); ok {
				return true
			}
		case "object":
			if _, ok := x.(map[string]any); ok {
				return true
			}
		case "null":
			if x == nil {
				return true
			}
		default:
			return true // a type name this checker does not know: do not refuse on it
		}
	}
	return false
}
