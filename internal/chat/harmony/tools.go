package harmony

import (
	"bytes"
	"encoding/json"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

func tsType(p chatcore.JV) string {
	typ, hasType := p.Get("type")
	if hasType && typ.Kind == chatcore.JStr && typ.S == "array" {
		var out string
		if items, ok := p.Get("items"); ok && items.Truthy() {
			it, _ := items.Get("type")
			switch {
			case it.Kind == chatcore.JStr && it.S == "string":
				out = "string[]"
			case it.Kind == chatcore.JStr && (it.S == "number" || it.S == "integer"):
				out = "number[]"
			case it.Kind == chatcore.JStr && it.S == "boolean":
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
	if hasType && typ.Kind == chatcore.JArr && len(typ.Arr) > 0 {
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
			if t, _ := v.Get("type"); t.Kind == chatcore.JStr && t.S == "object" {
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
				b.WriteString(strings.Repeat(" ", 20) + "// default: " + chatcore.PyDumps(d))
			}
			if i < len(one.Arr)-1 {
				b.WriteString(" | \n")
			}
		}
		return b.String()
	}
	switch {
	case hasType && typ.Kind == chatcore.JStr && typ.S == "string":
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
	case hasType && typ.Kind == chatcore.JStr && (typ.S == "number" || typ.S == "integer"):
		return "number"
	case hasType && typ.Kind == chatcore.JStr && typ.S == "boolean":
		return "boolean"
	case hasType && typ.Kind == chatcore.JStr && typ.S == "object":
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

func requiredSet(schema chatcore.JV) map[string]bool {
	set := map[string]bool{}
	if r, ok := schema.Get("required"); ok && r.Kind == chatcore.JArr {
		for _, v := range r.Arr {
			set[v.Text()] = true
		}
	}
	return set
}

// Namespace formats tool declarations as Harmony's TypeScript `namespace functions` block.
func Namespace(tools []chatcore.Tool) string {
	var b strings.Builder
	b.WriteString("## functions\n\nnamespace functions {\n\n")
	for _, t := range tools {
		if t.Description != "" {
			b.WriteString("// " + t.Description + "\n")
		}
		b.WriteString("type " + t.Name + " = ")
		schema, err := chatcore.ParseJV(t.Parameters)
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
					b.WriteString(", // default: " + chatcore.PyDumps(def))
				}
			}
			b.WriteString(",\n")
		}
		b.WriteString("}) => any;\n\n")
	}
	b.WriteString("} // namespace functions")
	return b.String()
}

// Segments renders system + turns (+ optional tool declarations) in Harmony format, ending with the generation prompt.
func Segments(now time.Time, effort, system string, turns []chatcore.Turn, tools []chatcore.Tool) []chatcore.Segment {
	var b chatcore.SegBuf
	b.Sp("<|start|>")
	b.Ct("system")
	b.Sp("<|message|>")
	preamble := "You are ChatGPT, a large language model trained by OpenAI.\n" +
		"Knowledge cutoff: 2024-06\n" +
		"Current date: " + now.Format("2006-01-02") + "\n\n" +
		"Reasoning: " + effort + "\n\n" +
		"# Valid channels: analysis, commentary, final. Channel must be included for every message."
	if len(tools) > 0 {
		preamble += "\nCalls to these tools must go to the commentary channel: 'functions'."
	}
	b.Ct(preamble)
	b.Sp("<|end|>")
	if system != "" || len(tools) > 0 {
		b.Sp("<|start|>")
		b.Ct("developer")
		b.Sp("<|message|>")
		var dev strings.Builder
		if system != "" {
			dev.WriteString("# Instructions\n\n" + system + "\n\n")
		}
		if len(tools) > 0 {
			dev.WriteString("# Tools\n\n" + Namespace(tools))
		}
		b.Ct(dev.String())
		b.Sp("<|end|>")
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
					b.Sp("<|start|>")
					b.Ct("assistant")
					b.Sp("<|channel|>")
					b.Ct("analysis")
					b.Sp("<|message|>")
					b.Ct(text)
					b.Sp("<|end|>")
				}
				if t.Reasoning != "" && t.Content != "" {
					b.Sp("<|start|>")
					b.Ct("assistant")
					b.Sp("<|channel|>")
					b.Ct("commentary")
					b.Sp("<|message|>")
					b.Ct(t.Content)
					b.Sp("<|end|>")
				}
			}
			for _, c := range t.ToolCalls {
				b.Sp("<|start|>")
				b.Ct("assistant to=functions." + c.Name)
				b.Sp("<|channel|>")
				b.Ct("commentary json")
				b.Sp("<|message|>")
				b.Ct(chatcore.PyArgs(c.Arguments))
				b.Sp("<|call|>")
				lastCall = c.Name
				if c.ID != "" {
					callNames[c.ID] = c.Name
				}
			}
		case t.Role == "assistant":
			b.Sp("<|start|>")
			b.Ct("assistant")
			b.Sp("<|channel|>")
			b.Ct("final")
			b.Sp("<|message|>")
			b.Ct(t.Content)
			b.Sp("<|end|>")
			lastCall = ""
		case t.Role == "tool":
			name := lastCall
			if n, ok := callNames[t.ToolCallID]; ok {
				name = n
			}
			if name == "" {
				name = t.ToolName
			}
			b.Sp("<|start|>")
			b.Ct("functions." + name + " to=assistant")
			b.Sp("<|channel|>")
			b.Ct("commentary")
			b.Sp("<|message|>")
			b.Ct(chatcore.PyQuote(t.Content))
			b.Sp("<|end|>")
		default:
			b.Sp("<|start|>")
			b.Ct("user")
			b.Sp("<|message|>")
			b.Ct(t.Content)
			b.Sp("<|end|>")
		}
	}
	b.Sp("<|start|>")
	b.Ct("assistant")
	return b.Segs
}

// ToolCalls converts parsed recipient messages into structured ToolCalls for functions.* recipients.
func ToolCalls(calls []Call) []chatcore.ToolCall {
	var out []chatcore.ToolCall
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
		out = append(out, chatcore.ToolCall{Name: name, Arguments: json.RawMessage(compact.Bytes())})
	}
	return out
}

// ParseTools parses a complete Harmony reply into structured tool calls and leading prose.
func ParseTools(out string) ([]chatcore.ToolCall, string) {
	sp := NewSplitter()
	_, c1 := sp.Push(out)
	_, c2 := sp.Flush()
	content := c1 + c2
	calls := sp.ToolCalls()
	if len(calls) == 0 {
		return nil, out
	}
	return calls, strings.TrimSpace(content)
}
