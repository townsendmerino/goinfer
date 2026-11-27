package chat

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

type jv = chatcore.JV

const (
	jNull = chatcore.JNull
	jBool = chatcore.JBool
	jNum  = chatcore.JNum
	jStr  = chatcore.JStr
	jArr  = chatcore.JArr
	jObj  = chatcore.JObj
)

var (
	parseJV = chatcore.ParseJV
	pyDumps = chatcore.PyDumps
	pyQuote = chatcore.PyQuote
	pyArgs  = chatcore.PyArgs
)

// tsType is the template's render_typescript_type macro, branch for branch and in the macro's order.
func tsType(p jv) string {
	typ, hasType := p.Get("type")
	if hasType && typ.Kind == jStr && typ.S == "array" {
		var out string
		if items, ok := p.Get("items"); ok && items.Truthy() {
			it, _ := items.Get("type")
			switch {
			case it.Kind == jStr && it.S == "string":
				out = "string[]"
			case it.Kind == jStr && (it.S == "number" || it.S == "integer"):
				out = "number[]"
			case it.Kind == jStr && it.S == "boolean":
				out = "boolean[]"
			default:
				inner := tsType(items)
				if inner == "object | object" || utf8.RuneCountInString(inner) > 50 {
					out = "any[]"
				} else {
					out = inner + "[]"
				}
			}
		} else {
			out = "any[]"
		}
		if n, _ := p.Get("nullable"); n.Truthy() {
			out += " | null"
		}
		return out
	}
	if hasType && typ.Kind == jArr && len(typ.Arr) > 0 {
		if len(typ.Arr) > 1 {
			parts := make([]string, len(typ.Arr))
			for i, t := range typ.Arr {
				parts[i] = t.Text()
			}
			return strings.Join(parts, " | ")
		}
		return typ.Arr[0].Text()
	}
	if one, ok := p.Get("oneOf"); ok && one.Truthy() {
		hasObject := false
		for _, v := range one.Arr {
			if t, _ := v.Get("type"); t.Kind == jStr && t.S == "object" {
				hasObject = true
			}
		}
		if hasObject && len(one.Arr) > 1 {
			return "any"
		}
		var b strings.Builder
		for i, v := range one.Arr {
			b.WriteString(tsType(v))
			if d, _ := v.Get("description"); d.Truthy() {
				b.WriteString("// " + d.Text())
			}
			if d, ok := v.Get("default"); ok {
				b.WriteString(strings.Repeat(" ", 20) + "// default: " + pyDumps(d))
			}
			if i < len(one.Arr)-1 {
				b.WriteString(" | \n")
			}
		}
		return b.String()
	}
	switch {
	case hasType && typ.Kind == jStr && typ.S == "string":
		if e, ok := p.Get("enum"); ok && e.Truthy() {
			parts := make([]string, len(e.Arr))
			for i, v := range e.Arr {
				parts[i] = v.Text()
			}
			return `"` + strings.Join(parts, `" | "`) + `"`
		}
		if n, _ := p.Get("nullable"); n.Truthy() {
			return "string | null"
		}
		return "string"
	case hasType && typ.Kind == jStr && (typ.S == "number" || typ.S == "integer"):
		return "number"
	case hasType && typ.Kind == jStr && typ.S == "boolean":
		return "boolean"
	case hasType && typ.Kind == jStr && typ.S == "object":
		props, ok := p.Get("properties")
		if !ok || !props.Truthy() {
			return "object"
		}
		required := requiredSet(p)
		var b strings.Builder
		b.WriteString("{\n")
		for i, name := range props.Keys {
			b.WriteString(name)
			if !required[name] {
				b.WriteString("?")
			}
			b.WriteString(": \n" + strings.Repeat(" ", 16) + tsType(props.Vals[name]))
			if i < len(props.Keys)-1 {
				b.WriteString(", ")
			}
		}
		b.WriteString("}")
		return b.String()
	}
	return "any"
}

func requiredSet(schema jv) map[string]bool {
	set := map[string]bool{}
	if r, ok := schema.Get("required"); ok && r.Kind == jArr {
		for _, v := range r.Arr {
			set[v.Text()] = true
		}
	}
	return set
}

// harmonyNamespace is the template's render_tool_namespace("functions", tools).
func harmonyNamespace(tools []Tool) string {
	var b strings.Builder
	b.WriteString("## functions\n\nnamespace functions {\n\n")
	for _, t := range tools {
		if t.Description != "" {
			b.WriteString("// " + t.Description + "\n")
		}
		b.WriteString("type " + t.Name + " = ")
		schema, err := parseJV(t.Parameters)
		props, _ := schema.Get("properties")
		if err != nil || !props.Truthy() {
			b.WriteString("() => any;\n\n")
			continue
		}
		required := requiredSet(schema)
		b.WriteString("(_: {\n")
		for _, name := range props.Keys {
			spec := props.Vals[name]
			if d, _ := spec.Get("description"); d.Truthy() {
				b.WriteString("// " + d.Text() + "\n")
			}
			b.WriteString(name)
			if !required[name] {
				b.WriteString("?")
			}
			b.WriteString(": " + tsType(spec))
			if def, ok := spec.Get("default"); ok {
				enum, _ := spec.Get("enum")
				oneOf, _ := spec.Get("oneOf")
				switch {
				case enum.Truthy():
					b.WriteString(", // default: " + def.Text())
				case oneOf.Truthy():
					b.WriteString("// default: " + def.Text())
				default:
					b.WriteString(", // default: " + pyDumps(def))
				}
			}
			b.WriteString(",\n")
		}
		b.WriteString("}) => any;\n\n")
	}
	b.WriteString("} // namespace functions")
	return b.String()
}

// harmonySegments renders system + turns (+ tool declarations when tools is non-empty) in Harmony, ending with the generation prompt.
func harmonySegments(effort, system string, turns []Turn, tools []Tool) []Segment {
	var b segBuf
	b.sp("<|start|>")
	b.ct("system")
	b.sp("<|message|>")
	preamble := "You are ChatGPT, a large language model trained by OpenAI.\n" +
		"Knowledge cutoff: 2024-06\n" +
		"Current date: " + timeNow().Format("2006-01-02") + "\n\n" +
		"Reasoning: " + effort + "\n\n" +
		"# Valid channels: analysis, commentary, final. Channel must be included for every message."
	if len(tools) > 0 {
		preamble += "\nCalls to these tools must go to the commentary channel: 'functions'."
	}
	b.ct(preamble)
	b.sp("<|end|>")
	if system != "" || len(tools) > 0 {
		b.sp("<|start|>")
		b.ct("developer")
		b.sp("<|message|>")
		var dev strings.Builder
		if system != "" {
			dev.WriteString("# Instructions\n\n" + system + "\n\n")
		}
		if len(tools) > 0 {
			dev.WriteString("# Tools\n\n" + harmonyNamespace(tools))
		}
		b.ct(dev.String())
		b.sp("<|end|>")
	}

	callNames := map[string]string{}
	lastCall := ""
	for i, t := range turns {
		switch {
		case t.Role == "assistant" && len(t.ToolCalls) > 0:
			answered := false
			for _, later := range turns[i+1:] {
				if later.Role == "assistant" && len(later.ToolCalls) == 0 {
					answered = true
				}
			}
			if !answered {
				text := t.Content
				if t.Reasoning != "" {
					text = t.Reasoning
				}
				if text != "" {
					b.sp("<|start|>")
					b.ct("assistant")
					b.sp("<|channel|>")
					b.ct("analysis")
					b.sp("<|message|>")
					b.ct(text)
					b.sp("<|end|>")
				}
				if t.Reasoning != "" && t.Content != "" {
					b.sp("<|start|>")
					b.ct("assistant")
					b.sp("<|channel|>")
					b.ct("commentary")
					b.sp("<|message|>")
					b.ct(t.Content)
					b.sp("<|end|>")
				}
			}
			for _, c := range t.ToolCalls {
				b.sp("<|start|>")
				b.ct("assistant to=functions." + c.Name)
				b.sp("<|channel|>")
				b.ct("commentary json")
				b.sp("<|message|>")
				b.ct(pyArgs(c.Arguments))
				b.sp("<|call|>")
				lastCall = c.Name
				if c.ID != "" {
					callNames[c.ID] = c.Name
				}
			}
		case t.Role == "assistant":
			b.sp("<|start|>")
			b.ct("assistant")
			b.sp("<|channel|>")
			b.ct("final")
			b.sp("<|message|>")
			b.ct(t.Content)
			b.sp("<|end|>")
			lastCall = ""
		case t.Role == "tool":
			name := lastCall
			if n, ok := callNames[t.ToolCallID]; ok {
				name = n
			}
			if name == "" {
				name = t.ToolName
			}
			b.sp("<|start|>")
			b.ct("functions." + name + " to=assistant")
			b.sp("<|channel|>")
			b.ct("commentary")
			b.sp("<|message|>")
			b.ct(pyQuote(t.Content))
			b.sp("<|end|>")
		default:
			b.sp("<|start|>")
			b.ct("user")
			b.sp("<|message|>")
			b.ct(t.Content)
			b.sp("<|end|>")
		}
	}
	b.sp("<|start|>")
	b.ct("assistant")
	return b.segs
}

func harmonyToolCalls(calls []harmonyCall) []ToolCall {
	var out []ToolCall
	for _, c := range calls {
		name, ok := strings.CutPrefix(c.Recipient, "functions.")
		if !ok || name == "" {
			continue
		}
		args := bytes.TrimSpace([]byte(c.Args))
		if len(args) == 0 {
			args = []byte("{}")
		}
		var obj map[string]json.RawMessage
		if json.Unmarshal(args, &obj) != nil || obj == nil {
			continue
		}
		var compact bytes.Buffer
		if json.Compact(&compact, args) != nil {
			continue
		}
		out = append(out, ToolCall{Name: name, Arguments: json.RawMessage(compact.Bytes())})
	}
	return out
}

func parseHarmonyTools(out string) ([]ToolCall, string) {
	sp := newHarmonySplitter()
	_, c1 := sp.Push(out)
	_, c2 := sp.Flush()
	content := c1 + c2
	calls := harmonyToolCalls(sp.calls)
	if len(calls) == 0 {
		return nil, out
	}
	return calls, strings.TrimSpace(content)
}
