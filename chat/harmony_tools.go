package chat

// gpt-oss (Harmony) tool calling: the prompt half (declarations, and how a call and its result replay) and the parse half (the model's
// call message back into a ToolCall). The oracle for the prompt half is the model's own chat template, byte for byte
// (testdata/chat_think_goldens/harmony_tools.json, scripts/pin_harmony_tools.py); the oracle for the parse half is what the model
// actually writes (harmony_parse.go).
//
// What the template does, which this file reproduces including the parts that look like accidents:
//
//   - Tools are declared in the DEVELOPER message as a TypeScript-like `namespace functions { … }`, one `type NAME = (_: {…}) => any;` per
//     tool, written by a macro (render_typescript_type) whose Jinja whitespace leaks into the prompt: a property of an object type is
//     preceded by "\n" + 16 spaces, the variants of a oneOf are separated by " | \n". HuggingFace's environment strips only whitespace
//     around block tags (trim_blocks, lstrip_blocks), so the bytes below are exactly what the model is shown. They differ from what
//     OpenAI's own harmony library renders; the checkpoint's template is the contract goinfer follows everywhere else.
//   - A call replays as `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>`, a result as
//     `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>"RESULT"<|end|>` (the result a JSON string).
//   - Text or reasoning beside a call is rendered as an ANALYSIS message — and only while no later assistant turn has answered
//     ("we want to drop the tool call analysis message if the model has output a later <|final|> message").
//   - JSON is Python's: `json.dumps(…, ensure_ascii=False)`, separators ", " and ": ", keys in the order given, no HTML escaping.
//     That is not what goinfer's jsonStr produces (it escapes <, >, & and '), which is right for the families whose templates use
//     Jinja's htmlsafe tojson; HuggingFace's own environment does not, and this checkpoint's golden shows it.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// ---- an order-preserving JSON value ---------------------------------------------------------------------------------------------
//
// A tool's schema is declared property by property in the prompt, in the order the client wrote them, so it cannot be decoded into a
// Go map. jv keeps key order; numbers keep their source text.

type jkind uint8

const (
	jNull jkind = iota
	jBool
	jNum
	jStr
	jArr
	jObj
)

type jv struct {
	kind jkind
	b    bool
	s    string // a string's value, or a number's source text
	arr  []jv
	keys []string
	vals map[string]jv
}

func parseJV(raw []byte) (jv, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := readJV(dec)
	if err != nil {
		return jv{}, err
	}
	if dec.More() {
		return jv{}, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func readJV(dec *json.Decoder) (jv, error) {
	tok, err := dec.Token()
	if err != nil {
		return jv{}, err
	}
	switch t := tok.(type) {
	case json.Delim:
		if t == '{' {
			v := jv{kind: jObj, vals: map[string]jv{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return jv{}, err
				}
				key, _ := kt.(string)
				val, err := readJV(dec)
				if err != nil {
					return jv{}, err
				}
				if _, dup := v.vals[key]; !dup { // Python: a repeated key keeps its first position and its last value
					v.keys = append(v.keys, key)
				}
				v.vals[key] = val
			}
			if _, err := dec.Token(); err != nil { // the closing brace
				return jv{}, err
			}
			return v, nil
		}
		v := jv{kind: jArr}
		for dec.More() {
			el, err := readJV(dec)
			if err != nil {
				return jv{}, err
			}
			v.arr = append(v.arr, el)
		}
		if _, err := dec.Token(); err != nil {
			return jv{}, err
		}
		return v, nil
	case string:
		return jv{kind: jStr, s: t}, nil
	case json.Number:
		return jv{kind: jNum, s: t.String()}, nil
	case bool:
		return jv{kind: jBool, b: t}, nil
	}
	return jv{kind: jNull}, nil
}

func (v jv) get(k string) (jv, bool) {
	if v.kind != jObj {
		return jv{}, false
	}
	x, ok := v.vals[k]
	return x, ok
}

// truthy is Python's truth value, which is what a Jinja `{% if x %}` tests: empty containers, "", 0, false and null are false.
func (v jv) truthy() bool {
	switch v.kind {
	case jBool:
		return v.b
	case jNum:
		return v.s != "0" && v.s != "0.0"
	case jStr:
		return v.s != ""
	case jArr:
		return len(v.arr) > 0
	case jObj:
		return len(v.keys) > 0
	}
	return false
}

// text is the value as Jinja's string concatenation and join would show it.
func (v jv) text() string {
	switch v.kind {
	case jStr, jNum:
		return v.s
	case jBool:
		if v.b {
			return "True"
		}
		return "False"
	case jNull:
		return "None"
	}
	return pyDumps(v)
}

// pyDumps is json.dumps(v, ensure_ascii=False): separators ", " and ": ", keys in order, no HTML escaping.
func pyDumps(v jv) string {
	var b strings.Builder
	writePy(&b, v)
	return b.String()
}

func writePy(b *strings.Builder, v jv) {
	switch v.kind {
	case jNull:
		b.WriteString("null")
	case jBool:
		if v.b {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case jNum:
		b.WriteString(v.s)
	case jStr:
		b.WriteString(pyQuote(v.s))
	case jArr:
		b.WriteByte('[')
		for i, el := range v.arr {
			if i > 0 {
				b.WriteString(", ")
			}
			writePy(b, el)
		}
		b.WriteByte(']')
	case jObj:
		b.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(pyQuote(k))
			b.WriteString(": ")
			writePy(b, v.vals[k])
		}
		b.WriteByte('}')
	}
}

// pyQuote is json.dumps(s, ensure_ascii=False) for a string: the short escapes Python uses, \u00XX for the other control characters,
// and everything else — including <, >, &, ' and U+2028 — as it is.
func pyQuote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// pyArgs renders a call's arguments the way the template's `arguments|tojson` does. Arguments that are not valid JSON (a client
// replaying a malformed call) go in as they are: the prompt is better a faithful record than an error.
func pyArgs(raw json.RawMessage) string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "{}"
	}
	v, err := parseJV(raw)
	if err != nil {
		return string(raw)
	}
	return pyDumps(v)
}

// ---- the declarations ------------------------------------------------------------------------------------------------------------

// tsType is the template's render_typescript_type macro, branch for branch and in the macro's order.
func tsType(p jv) string {
	typ, hasType := p.get("type")
	if hasType && typ.kind == jStr && typ.s == "array" {
		var out string
		if items, ok := p.get("items"); ok && items.truthy() {
			it, _ := items.get("type")
			switch {
			case it.kind == jStr && it.s == "string":
				out = "string[]"
			case it.kind == jStr && (it.s == "number" || it.s == "integer"):
				out = "number[]"
			case it.kind == jStr && it.s == "boolean":
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
		if n, _ := p.get("nullable"); n.truthy() {
			out += " | null"
		}
		return out
	}
	if hasType && typ.kind == jArr && len(typ.arr) > 0 { // "type": ["object", "object"] from a Union
		if len(typ.arr) > 1 {
			parts := make([]string, len(typ.arr))
			for i, t := range typ.arr {
				parts[i] = t.text()
			}
			return strings.Join(parts, " | ")
		}
		return typ.arr[0].text()
	}
	if one, ok := p.get("oneOf"); ok && one.truthy() {
		hasObject := false
		for _, v := range one.arr {
			if t, _ := v.get("type"); t.kind == jStr && t.s == "object" {
				hasObject = true
			}
		}
		if hasObject && len(one.arr) > 1 {
			return "any"
		}
		var b strings.Builder
		for i, v := range one.arr {
			b.WriteString(tsType(v))
			if d, _ := v.get("description"); d.truthy() {
				b.WriteString("// " + d.text())
			}
			if d, ok := v.get("default"); ok {
				// `{{ "// default: " … }}` has no left trim, so the line's 20 spaces of indentation stay.
				b.WriteString(strings.Repeat(" ", 20) + "// default: " + pyDumps(d))
			}
			if i < len(one.arr)-1 {
				b.WriteString(" | \n") // " | " and the newline that lstrip_blocks leaves in front of {% endif %}
			}
		}
		return b.String()
	}
	switch {
	case hasType && typ.kind == jStr && typ.s == "string":
		if e, ok := p.get("enum"); ok && e.truthy() {
			parts := make([]string, len(e.arr))
			for i, v := range e.arr {
				parts[i] = v.text()
			}
			return `"` + strings.Join(parts, `" | "`) + `"`
		}
		if n, _ := p.get("nullable"); n.truthy() {
			return "string | null"
		}
		return "string"
	case hasType && typ.kind == jStr && (typ.s == "number" || typ.s == "integer"):
		return "number"
	case hasType && typ.kind == jStr && typ.s == "boolean":
		return "boolean"
	case hasType && typ.kind == jStr && typ.s == "object":
		props, ok := p.get("properties")
		if !ok || !props.truthy() {
			return "object"
		}
		required := requiredSet(p)
		var b strings.Builder
		b.WriteString("{\n")
		for i, name := range props.keys {
			b.WriteString(name)
			if !required[name] {
				b.WriteString("?")
			}
			// `{{ render_typescript_type(…) }}` is on its own line with no left trim: its newline and indentation are kept.
			b.WriteString(": \n" + strings.Repeat(" ", 16) + tsType(props.vals[name]))
			if i < len(props.keys)-1 {
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
	if r, ok := schema.get("required"); ok && r.kind == jArr {
		for _, v := range r.arr {
			set[v.text()] = true
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
		props, _ := schema.get("properties")
		if err != nil || !props.truthy() {
			b.WriteString("() => any;\n\n")
			continue
		}
		required := requiredSet(schema)
		b.WriteString("(_: {\n")
		for _, name := range props.keys {
			spec := props.vals[name]
			if d, _ := spec.get("description"); d.truthy() {
				b.WriteString("// " + d.text() + "\n")
			}
			b.WriteString(name)
			if !required[name] {
				b.WriteString("?")
			}
			b.WriteString(": " + tsType(spec))
			if def, ok := spec.get("default"); ok {
				enum, _ := spec.get("enum")
				oneOf, _ := spec.get("oneOf")
				switch {
				case enum.truthy():
					b.WriteString(", // default: " + def.text())
				case oneOf.truthy():
					b.WriteString("// default: " + def.text())
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

// ---- the conversation ------------------------------------------------------------------------------------------------------------

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

	// callNames maps a call id to its function, so a result is attributed to the call it answers; lastCall is the template's own
	// fallback (the most recent call's name).
	callNames := map[string]string{}
	lastCall := ""
	for i, t := range turns {
		switch {
		case t.Role == "assistant" && len(t.ToolCalls) > 0:
			// The analysis a call carries is kept only while no later assistant turn has answered.
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
					// The template refuses reasoning AND text on one call turn. goinfer's own parser produces exactly that pair (an analysis
					// message and a commentary preamble before the call), so it is replayed as the model wrote it.
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

// ---- the reply -------------------------------------------------------------------------------------------------------------------

// harmonyToolCalls turns the call messages the parser kept into ToolCalls: a recipient `functions.NAME` is a call to the client's
// function NAME; any other recipient (gpt-oss's built-in browser and python, never declared here) is not something a client can run,
// and a call whose arguments are not a JSON object is not a call either — both are left out, as the other families' parsers leave out
// a call they cannot read.
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

// parseHarmonyTools reads a whole raw gpt-oss reply — channel markers and all, as -reasoning-format none leaves it — into its calls and
// the text the model said to the user.
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
