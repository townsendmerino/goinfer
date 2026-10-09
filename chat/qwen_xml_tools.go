package chat

// Qwen3.5's own tool-call format, as its chat template writes it:
//
//	<tool_call>
//	<function=get_weather>
//	<parameter=city>
//	Paris
//	</parameter>
//	</function>
//	</tool_call>
//
// goinfer's default for the ChatML families is the Qwen2.5/Hermes form (a system block of signatures, "return a json object … within
// <tool_call>", and `<tool_call>{json}</tool_call>`). Measured on Qwen3.5-0.8B and 9B (docs/measurements/qwen35-tool-format-2026-09-30/),
// a model prompted that way follows the JSON instruction and does not write the XML form — so the XML form is not a defect being fixed.
// It is a choice, made per request by `-tool-format`, between two prompts for the same model:
//
//	hermes    goinfer's own prompt (the default)
//	template  the model's own chat template, byte for byte (testdata/chat_think_goldens/qwen35_tools.json)
//
// and the PARSER reads both forms whichever was asked for, so a model that writes the XML form under any prompt is not ignored.
//
// The native renderer is enabled only for a template that declares the form AND has the Qwen3.5 layout (the history rule history.go
// reads from the template text), because that layout is what the golden covers; another template that merely mentions `<function=`
// (Qwen3-Coder) keeps the Hermes renderer until it has an oracle of its own.

import (
	"bytes"
	"encoding/json"
	"strings"
)

// ToolFormat is how a family's tools are put in the prompt and its calls replayed, for a template that declares a native form goinfer can
// reproduce byte for byte (Qwen3.5's XML; Gemma 4's canonical template).
type ToolFormat uint8

const (
	// ToolFormatAuto, the default, is each family's MEASURED default: the model's own template form where a pre-registered A/B adopted it
	// (canonical Gemma 4 — docs/measurements/gemma4-tool-text-order-2026-09-30/), goinfer's own prompt where it did not (Qwen3.5's XML was
	// parked — docs/measurements/qwen35-tool-format-2026-09-30/).
	ToolFormatAuto ToolFormat = iota
	// ToolFormatHermes is goinfer's own prompt: for Qwen the signatures and JSON calls, for Gemma 4 goinfer's own order (text before a call).
	ToolFormatHermes
	// ToolFormatTemplate is the model's own chat template, byte for byte.
	ToolFormatTemplate
)

// ParseToolFormat reads the -tool-format flag's value ("" is auto).
func ParseToolFormat(s string) (ToolFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "auto":
		return ToolFormatAuto, true
	case "hermes":
		return ToolFormatHermes, true
	case "template":
		return ToolFormatTemplate, true
	}
	return ToolFormatAuto, false
}

func (f ToolFormat) String() string {
	switch f {
	case ToolFormatHermes:
		return "hermes"
	case ToolFormatTemplate:
		return "template"
	}
	return "auto"
}

// declaresQwen35XMLTools reports whether a ChatML template writes tool calls in the XML form AND has the Qwen3.5 layout the native
// renderer is pinned to.
func declaresQwen35XMLTools(tmpl string, r *Reasoning) bool {
	return r != nil && r.hist == histQwen35 && strings.Contains(tmpl, "<function=") && strings.Contains(tmpl, "<parameter=")
}

// WithToolFormat returns a copy of t that renders tools in format f. It changes nothing for a template with no native form to switch to.
func (t *Template) WithToolFormat(f ToolFormat) *Template {
	if t == nil || !t.nativeTools || t.toolFormat == f {
		return t
	}
	c := *t
	c.toolFormat = f
	return &c
}

// ToolFormat is the format WithToolFormat selected (auto by default).
func (t *Template) ToolFormat() ToolFormat {
	if t == nil {
		return ToolFormatAuto
	}
	return t.toolFormat
}

// UsesNativeTools reports whether tools are rendered in the model's own template form: a template that declares one, with the format
// forced to `template`, or left on `auto` where that family's measured default is the template's form.
func (t *Template) UsesNativeTools() bool { return t.usesNativeTools() }

// DeclaresNativeTools reports whether t has a native tool form goinfer can render — so `-tool-format template` does something for it.
func (t *Template) DeclaresNativeTools() bool { return t != nil && t.nativeTools }

// usesNativeTools: render the model's own form (UsesNativeTools).
func (t *Template) usesNativeTools() bool {
	if t == nil || !t.nativeTools {
		return false
	}
	// `auto` does not select the native form under -thinking asis: asis is the prompt as goinfer rendered it before thinking was modelled,
	// reasoning ignored, and the native Gemma form replays it. An explicit `template` is asked for by name and is honoured.
	return t.toolFormat == ToolFormatTemplate || (t.toolFormat == ToolFormatAuto && t.nativeByDefault && t.think != ThinkAsIs)
}

// ---- the prompt -----------------------------------------------------------------------------------------------------------------

const qwenXMLToolsPrologue = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

const qwenXMLToolsEpilogue = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

// qwenToolJSON is `tool | tojson` for one declaration, as HuggingFace renders it: {"type": "function", "function": {"name", "description",
// "parameters"}} with Python's separators and the schema's own key order. A missing description or parameters object is left out, as a
// client that omitted it would have left it out.
func qwenToolJSON(t Tool) string {
	var b strings.Builder
	b.WriteString(`{"type": "function", "function": {"name": ` + pyQuote(t.Name))
	if t.Description != "" {
		b.WriteString(`, "description": ` + pyQuote(t.Description))
	}
	if len(bytes.TrimSpace(t.Parameters)) > 0 {
		b.WriteString(`, "parameters": ` + pyArgs(t.Parameters))
	}
	b.WriteString("}}")
	return b.String()
}

// pyStr is Python's str() of a decoded JSON value, which is what the template writes for an argument that is not a mapping or a
// non-string sequence: a string as it is, a number as its digits, a boolean as True / False, null as None. A mapping or a list is JSON.
func pyStr(v jv) string {
	switch v.kind {
	case jStr:
		return v.s
	case jNum:
		return v.s
	case jBool, jNull:
		return v.text()
	}
	return pyDumps(v)
}

// renderQwen35XMLTools renders system + turns + tool declarations the way the Qwen3.5 chat template does, ending with the generation
// prompt's `<|im_start|>assistant\n` (the caller appends the think suffix, as for the Hermes renderer).
func renderQwen35XMLTools(system string, turns []Turn, tools []Tool, hist histKind) string {
	var b strings.Builder
	sys := strings.TrimSpace(system)
	b.WriteString("<|im_start|>system\n" + qwenXMLToolsPrologue)
	for _, t := range tools {
		b.WriteString("\n" + qwenToolJSON(t))
	}
	b.WriteString("\n</tools>" + qwenXMLToolsEpilogue)
	if sys != "" {
		b.WriteString("\n\n" + sys)
	}
	b.WriteString("<|im_end|>\n")

	lq := lastQueryIndex(turns)
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			block, reasoning, content := qwenAssistant(hist, m, i, lq, len(turns))
			content = strings.TrimSpace(content)
			b.WriteString("<|im_start|>assistant\n")
			if block {
				b.WriteString("<think>\n" + reasoning + "\n</think>\n\n")
			}
			b.WriteString(content)
			for ci, c := range m.ToolCalls {
				switch {
				case ci == 0 && content != "":
					b.WriteString("\n\n")
				case ci > 0:
					b.WriteString("\n")
				}
				b.WriteString("<tool_call>\n<function=" + c.Name + ">\n")
				if args, err := parseJV(c.Arguments); err == nil && args.kind == jObj {
					for _, k := range args.keys {
						b.WriteString("<parameter=" + k + ">\n" + pyStr(args.vals[k]) + "\n</parameter>\n")
					}
				}
				b.WriteString("</function>\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			// Consecutive results share one user turn.
			if i == 0 || turns[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n" + strings.TrimSpace(m.Content) + "\n</tool_response>")
			if i == len(turns)-1 || turns[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		default:
			b.WriteString("<|im_start|>user\n" + strings.TrimSpace(m.Content) + "<|im_end|>\n")
		}
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// ---- the reply ------------------------------------------------------------------------------------------------------------------

// parseQwenXMLCall reads one `<function=NAME> <parameter=K>V</parameter>… </function>` body. Values are text: a parameter the tool's schema
// types as a string stays a string, a number/integer/boolean/array/object/null is decoded from its text (the Python forms True, False and
// None included), and a parameter the schema does not name — or no schema at all — is decoded when it is unambiguously JSON (a number, a
// boolean, null, an object or a list) and kept as text otherwise. That is the one place the text and the type can disagree: a value "123"
// for a string parameter must stay "123", which only the schema can say.
func parseQwenXMLCall(body string, tools []Tool) (ToolCall, bool) {
	_, after, ok := strings.Cut(body, "<function=")
	if !ok {
		return ToolCall{}, false
	}
	rest := after
	end := strings.Index(rest, ">")
	if end <= 0 {
		return ToolCall{}, false
	}
	name := strings.TrimSpace(rest[:end])
	rest = rest[end+1:]
	fn := rest
	if j := strings.LastIndex(rest, "</function>"); j >= 0 {
		fn = rest[:j]
	} else {
		return ToolCall{}, false // cut off inside the call: not a call
	}
	types := paramTypes(name, tools)

	var buf bytes.Buffer
	buf.WriteByte('{')
	n := 0
	for {
		p := strings.Index(fn, "<parameter=")
		if p < 0 {
			break
		}
		fn = fn[p+len("<parameter="):]
		e := strings.Index(fn, ">")
		if e <= 0 {
			return ToolCall{}, false
		}
		key := strings.TrimSpace(fn[:e])
		fn = fn[e+1:]
		c := closeParam(fn)
		if c < 0 {
			return ToolCall{}, false
		}
		val := fn[:c]
		fn = fn[c+len("</parameter>"):]
		val = strings.TrimPrefix(val, "\n")
		val = strings.TrimSuffix(val, "\n")
		kb, _ := json.Marshal(key)
		if n > 0 {
			buf.WriteByte(',')
		}
		buf.Write(kb)
		buf.WriteByte(':')
		buf.Write(typedParam(val, types[key]))
		n++
	}
	buf.WriteByte('}')
	return ToolCall{Name: name, Arguments: json.RawMessage(buf.Bytes())}, true
}

// paramTypes is the declared JSON type of each parameter of the tool called name ("" when unknown).
func paramTypes(name string, tools []Tool) map[string]string {
	out := map[string]string{}
	for _, t := range tools {
		if t.Name != name {
			continue
		}
		schema, err := parseJV(t.Parameters)
		if err != nil {
			return out
		}
		props, _ := schema.get("properties")
		for _, k := range props.keys {
			if ty, ok := props.vals[k].get("type"); ok && ty.kind == jStr {
				out[k] = ty.s
			}
		}
	}
	return out
}

// typedParam returns the JSON encoding of a parameter's text value under its declared type ("" = undeclared).
func typedParam(val, typ string) []byte {
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
	// Undeclared: JSON where it is unambiguously JSON, the Python spellings of the constants, text otherwise.
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

// closeParam finds the </parameter> that ends a value: the first one that is followed by another parameter or by the function's end. A
// value may itself contain the text "</parameter>" (a document about this very format); a structural close never is followed by anything
// else, which is what tells them apart.
func closeParam(s string) int {
	const tag = "</parameter>"
	for from := 0; ; {
		i := strings.Index(s[from:], tag)
		if i < 0 {
			return -1
		}
		i += from
		next := strings.TrimLeft(s[i+len(tag):], " \t\r\n")
		if next == "" || strings.HasPrefix(next, "<parameter=") || strings.HasPrefix(next, "</function>") {
			return i
		}
		from = i + len(tag)
	}
}
