package chat

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// Gemma 4 tool calling — a bespoke non-JSON micro-language, distinct from the
// other families. Tools are declared as
//
//	<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:<gschema>}<tool|>
//
// a call is
//
//	<|tool_call>call:NAME{key:<|"|>val<|"|>,…}<tool_call|>
//
// and a result is <|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>.
// Strings are wrapped in the <|"|> quote marker; types are upper-cased; schema
// keys are emitted alphabetically.

const gq = `<|"|>` // Gemma's string-quote marker

func renderGemma4Tools(system string, turns []Turn, tools []Tool) string {
	var b strings.Builder
	b.WriteString("<bos><|turn>system\n")
	if s := strings.TrimSpace(system); s != "" {
		b.WriteString(s + "\n")
	}
	for _, tl := range tools {
		b.WriteString("<|tool>declaration:" + tl.Name + "{description:" + gq + tl.Description + gq +
			",parameters:" + gemmaSchema(tl.Parameters) + "}<tool|>")
	}
	b.WriteString("<turn|>\n")
	for _, m := range turns {
		switch m.Role {
		case "assistant":
			b.WriteString("<|turn>model\n" + m.Content)
			for _, c := range m.ToolCalls {
				b.WriteString("<|tool_call>call:" + c.Name + "{" + gemmaArgs(c.Arguments) + "}<tool_call|>")
			}
			b.WriteString("<turn|>\n")
		case "tool":
			b.WriteString("<|tool_response>response:" + m.ToolName + "{value:" + gq + m.Content + gq + "}<tool_response|>")
		default:
			b.WriteString("<|turn>user\n" + m.Content + "<turn|>\n")
		}
	}
	b.WriteString("<|turn>model\n<|channel>thought\n<channel|>")
	return b.String()
}

func parseGemma4Tools(out string) ([]ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<|tool_call>"); ok {
		lead = before
	}
	var calls []ToolCall
	for rest := out; ; {
		i := strings.Index(rest, "<|tool_call>call:")
		if i < 0 {
			break
		}
		rest = rest[i+len("<|tool_call>call:"):]
		brace := strings.IndexByte(rest, '{')
		if brace < 0 {
			break
		}
		name := strings.TrimSpace(rest[:brace])
		rest = rest[brace+1:]
		end := strings.Index(rest, "}<tool_call|>")
		body := rest
		if end >= 0 {
			body = rest[:end]
			rest = rest[end+len("}<tool_call|>"):]
		}
		calls = append(calls, ToolCall{Name: name, Arguments: gemmaBodyToJSON(body)})
		if end < 0 {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// gemmaSchema renders a JSON Schema in Gemma's parameter syntax (keys alphabetical,
// types upper-cased, strings in <|"|> quotes).
func gemmaSchema(raw json.RawMessage) string {
	var s map[string]any
	if json.Unmarshal(raw, &s) != nil {
		return "{type:" + gq + "OBJECT" + gq + "}"
	}
	keys := make([]string, 0, len(s))
	for k := range s {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		v := s[k]
		switch k {
		case "type":
			parts = append(parts, "type:"+gq+strings.ToUpper(toStr(v))+gq)
		case "description":
			parts = append(parts, "description:"+gq+toStr(v)+gq)
		case "enum":
			parts = append(parts, "enum:"+gemmaStrList(v))
		case "required":
			parts = append(parts, "required:"+gemmaStrList(v))
		case "properties":
			parts = append(parts, "properties:"+gemmaProps(v))
		case "items":
			if sub, _ := json.Marshal(v); sub != nil {
				parts = append(parts, "items:"+gemmaSchema(sub))
			}
		}
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func gemmaProps(v any) string {
	m, ok := v.(map[string]any)
	if !ok {
		return "{}"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		sub, _ := json.Marshal(m[k])
		parts = append(parts, k+":"+gemmaSchema(sub))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func gemmaStrList(v any) string {
	arr, ok := v.([]any)
	if !ok {
		return "[]"
	}
	var parts []string
	for _, e := range arr {
		parts = append(parts, gq+toStr(e)+gq)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// gemmaArgs renders a JSON arguments object as Gemma's key:value body.
func gemmaArgs(raw json.RawMessage) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, k+":"+gemmaValue(m[k]))
	}
	return strings.Join(parts, ",")
}

func gemmaValue(v any) string {
	switch x := v.(type) {
	case string:
		return gq + x + gq
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// gemmaBodyToJSON parses a "key:value,…" call body into a JSON object. String
// values are <|"|>-quoted; bare values are parsed as numbers/bools (else string).
func gemmaBodyToJSON(body string) json.RawMessage {
	out := map[string]any{}
	for _, kv := range splitGemmaPairs(body) {
		before, after, ok := strings.Cut(kv, ":")
		if !ok {
			continue
		}
		key := strings.TrimSpace(before)
		val := strings.TrimSpace(after)
		switch {
		case strings.HasPrefix(val, gq):
			out[key] = strings.TrimSuffix(strings.TrimPrefix(val, gq), gq)
		case val == "true" || val == "false":
			out[key] = val == "true"
		default:
			if f, err := strconv.ParseFloat(val, 64); err == nil {
				out[key] = f
			} else {
				out[key] = val
			}
		}
	}
	b, _ := json.Marshal(out)
	return b
}

// splitGemmaPairs splits on commas that are not inside a <|"|>…<|"|> string.
func splitGemmaPairs(body string) []string {
	var pairs []string
	depth := 0 // inside a quoted string?
	start := 0
	for i := 0; i < len(body); i++ {
		if strings.HasPrefix(body[i:], gq) {
			depth ^= 1
			i += len(gq) - 1
			continue
		}
		if body[i] == ',' && depth == 0 {
			pairs = append(pairs, body[start:i])
			start = i + 1
		}
	}
	if start < len(body) {
		pairs = append(pairs, body[start:])
	}
	return pairs
}

func toStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return string(b)
}
