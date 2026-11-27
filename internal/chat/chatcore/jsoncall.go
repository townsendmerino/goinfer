package chatcore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// ApostropheEscape is Jinja htmlsafe_json_dumps's 6-byte \u0027 escape for an apostrophe.
var ApostropheEscape = fmt.Sprintf(`\u%04x`, '\'')

// ToJSONEscape applies Jinja htmlsafe_json_dumps apostrophe escaping atop encoding/json output.
func ToJSONEscape(s string) string {
	return strings.ReplaceAll(s, "'", ApostropheEscape)
}

// JSONStr marshals s as a JSON string with Jinja htmlsafe escaping.
func JSONStr(s string) string {
	b, _ := json.Marshal(s)
	return ToJSONEscape(string(b))
}

// FuncDefJSON formats t as an OpenAI function-definition JSON object with Jinja htmlsafe escaping.
func FuncDefJSON(t Tool) string {
	m := map[string]any{"type": "function", "function": map[string]any{
		"name": t.Name, "description": t.Description, "parameters": json.RawMessage(t.Parameters),
	}}
	b, _ := json.Marshal(m)
	return ToJSONEscape(string(b))
}

// CallObjectJSON formats {"name": ..., "<argsKey>": {...}} for assistant tool-call history.
func CallObjectJSON(c ToolCall, argsKey string) string {
	args := string(c.Arguments)
	if args == "" {
		args = "{}"
	}
	return `{"name": ` + JSONStr(c.Name) + `, "` + argsKey + `": ` + args + `}`
}

// NormalizeArgs ensures arguments are a JSON object, unwrapping a stringified object when present.
func NormalizeArgs(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		var obj map[string]any
		if json.Unmarshal([]byte(s), &obj) == nil {
			return json.RawMessage(s)
		}
	}
	return raw
}

// CallFromJSON parses the first JSON object in s as {"name": ..., "<argsKey>": {...}}.
func CallFromJSON(s, argsKey string) (ToolCall, bool) {
	var raw map[string]json.RawMessage
	if json.NewDecoder(strings.NewReader(s)).Decode(&raw) != nil {
		return ToolCall{}, false
	}
	var name string
	if json.Unmarshal(raw["name"], &name) != nil || name == "" {
		return ToolCall{}, false
	}
	args := raw[argsKey]
	if args == nil {
		if a, ok := raw["arguments"]; ok {
			args = a
		} else {
			args = raw["parameters"]
		}
	}
	var id string
	if v, ok := raw["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	return ToolCall{ID: id, Name: name, Arguments: NormalizeArgs(args)}, true
}

// CallsFromArray parses the first JSON array in s into ToolCalls.
func CallsFromArray(s, argsKey string) []ToolCall {
	var arr []json.RawMessage
	if json.NewDecoder(strings.NewReader(s)).Decode(&arr) != nil {
		return nil
	}
	var calls []ToolCall
	for _, e := range arr {
		if c, ok := CallFromJSON(string(e), argsKey); ok {
			calls = append(calls, c)
		}
	}
	return calls
}

// BareToolCall parses an unwrapped leading JSON call object when its name matches one of tools.
func BareToolCall(out string, tools []Tool) (ToolCall, bool) {
	s := strings.TrimLeft(out, " \t\r\n")
	if !strings.HasPrefix(s, "{") {
		return ToolCall{}, false
	}
	var raw map[string]json.RawMessage
	if json.NewDecoder(strings.NewReader(s)).Decode(&raw) != nil {
		return ToolCall{}, false
	}
	var name string
	if json.Unmarshal(raw["name"], &name) != nil {
		return ToolCall{}, false
	}
	known := false
	for _, tl := range tools {
		if tl.Name == name {
			known = true
			break
		}
	}
	if !known {
		return ToolCall{}, false
	}
	args, ok := raw["arguments"]
	if !ok {
		args, ok = raw["parameters"]
	}
	if !ok {
		return ToolCall{}, false
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(args, &obj) != nil || obj == nil {
		return ToolCall{}, false
	}
	var id string
	if v, ok := raw["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	return ToolCall{ID: id, Name: name, Arguments: args}, true
}

// ParamTypes returns the declared JSON Schema type of each property for the named tool.
func ParamTypes(name string, tools []Tool) map[string]string {
	out := map[string]string{}
	for _, t := range tools {
		if t.Name != name {
			continue
		}
		schema, err := ParseJV(t.Parameters)
		if err != nil {
			return out
		}
		props, _ := schema.Get("properties")
		for _, k := range props.Keys {
			if ty, ok := props.Vals[k].Get("type"); ok && ty.Kind == JStr {
				out[k] = ty.S
			}
		}
	}
	return out
}

// TypedParam encodes a text parameter value as JSON according to its declared schema type ("" when undeclared).
func TypedParam(val, typ string) []byte {
	asString := func() []byte { b, _ := json.Marshal(val); return b }
	t := strings.TrimSpace(val)
	switch typ {
	case "string":
		return asString()
	case "integer", "number":
		if json.Valid([]byte(t)) {
			var n json.Number
			if json.Unmarshal([]byte(t), &n) == nil {
				return []byte(n.String())
			}
		}
		return asString()
	case "boolean":
		switch strings.ToLower(t) {
		case "true":
			return []byte("true")
		case "false":
			return []byte("false")
		}
		return asString()
	case "null":
		if t == "None" || t == "null" {
			return []byte("null")
		}
		return asString()
	case "array", "object":
		if json.Valid([]byte(t)) {
			return []byte(t)
		}
		return asString()
	}
	switch t {
	case "True":
		return []byte("true")
	case "False":
		return []byte("false")
	case "None":
		return []byte("null")
	}
	if t != "" && json.Valid([]byte(t)) {
		var v any
		if json.Unmarshal([]byte(t), &v) == nil {
			switch v.(type) {
			case float64, bool, nil, map[string]any, []any:
				return []byte(t)
			}
		}
	}
	return asString()
}

// ArgsMatchSchema checks obj against the required, properties, enum, and additionalProperties fields of schema.
func ArgsMatchSchema(obj map[string]json.RawMessage, schema json.RawMessage) bool {
	if len(bytes.TrimSpace(schema)) == 0 {
		return true
	}
	var sc struct {
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
		AdditionalProperties json.RawMessage            `json:"additionalProperties"`
	}
	if json.Unmarshal(schema, &sc) != nil {
		return true
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
		if !ValueHasType(v, p.Type) {
			return false
		}
		if len(p.Enum) > 0 {
			in := false
			for _, e := range p.Enum {
				if jsonEqual(e, v) {
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

func jsonEqual(a, b json.RawMessage) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	xb, _ := json.Marshal(x)
	yb, _ := json.Marshal(y)
	return bytes.Equal(xb, yb)
}

// ValueHasType reports whether JSON value v satisfies the JSON Schema type declaration typ.
func ValueHasType(v, typ json.RawMessage) bool {
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
			return true
		}
	}
	return false
}
