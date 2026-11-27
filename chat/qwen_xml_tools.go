package chat

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

type ToolFormat = chatcore.ToolFormat

const (
	ToolFormatAuto     = chatcore.ToolFormatAuto
	ToolFormatHermes   = chatcore.ToolFormatHermes
	ToolFormatTemplate = chatcore.ToolFormatTemplate
)

var ParseToolFormat = chatcore.ParseToolFormat

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
	return t.toolFormat == ToolFormatTemplate || (t.toolFormat == ToolFormatAuto && t.nativeByDefault && t.think != ThinkAsIs)
}

const qwenXMLToolsPrologue = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

const qwenXMLToolsEpilogue = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

func qwenToolJSON(t Tool) string {
	var b strings.Builder
	b.WriteString(`{"type": "function", "function": {"name": ` + chatcore.PyQuote(t.Name))
	if t.Description != "" {
		b.WriteString(`, "description": ` + chatcore.PyQuote(t.Description))
	}
	if len(bytes.TrimSpace(t.Parameters)) > 0 {
		b.WriteString(`, "parameters": ` + chatcore.PyArgs(t.Parameters))
	}
	b.WriteString("}}")
	return b.String()
}

func (t *Template) renderQwen35XMLTools(system string, turns []Turn, tools []Tool, hist histKind) []Segment {
	b := t.newToolBuf(chatMLToolMarks...)
	sys := strings.TrimSpace(system)
	b.Lit("<|im_start|>system\n" + qwenXMLToolsPrologue)
	for _, tl := range tools {
		b.Txt("\n" + qwenToolJSON(tl))
	}
	b.Lit("\n</tools>" + qwenXMLToolsEpilogue)
	if sys != "" {
		b.Lit("\n\n")
		b.Txt(sys)
	}
	b.Lit("<|im_end|>\n")

	lq := lastQueryIndex(turns)
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			block, reasoning, content := qwenAssistant(hist, m, i, lq, len(turns))
			content = strings.TrimSpace(content)
			b.Lit("<|im_start|>assistant\n")
			if block {
				b.Lit("<think>")
				b.Txt("\n" + reasoning + "\n")
				b.Lit("</think>\n\n")
			}
			b.Txt(content)
			for ci, c := range m.ToolCalls {
				switch {
				case ci == 0 && content != "":
					b.Txt("\n\n")
				case ci > 0:
					b.Txt("\n")
				}
				b.Lit("<tool_call>\n<function=")
				b.Txt(c.Name)
				b.Lit(">\n")
				if args, err := chatcore.ParseJV(c.Arguments); err == nil && args.Kind == chatcore.JObj {
					for _, k := range args.Keys {
						b.Lit("<parameter=")
						b.Txt(k)
						b.Lit(">\n")
						b.Txt(chatcore.PyStr(args.Vals[k]))
						b.Lit("\n</parameter>\n")
					}
				}
				b.Lit("</function>\n</tool_call>")
			}
			b.Lit("<|im_end|>\n")
		case "tool":
			if i == 0 || turns[i-1].Role != "tool" {
				b.Lit("<|im_start|>user")
			}
			b.Lit("\n<tool_response>\n")
			b.Txt(strings.TrimSpace(m.Content))
			b.Lit("\n</tool_response>")
			if i == len(turns)-1 || turns[i+1].Role != "tool" {
				b.Lit("<|im_end|>\n")
			}
		default:
			b.Lit("<|im_start|>user\n")
			b.Txt(strings.TrimSpace(m.Content))
			b.Lit("<|im_end|>\n")
		}
	}
	b.Lit("<|im_start|>assistant\n")
	return b.Out()
}

func parseQwenXMLCall(body string, tools []Tool) (ToolCall, bool) {
	_, after, ok := strings.Cut(body, "<function=")
	if !ok {
		return ToolCall{}, false
	}
	namePart, rest, ok := strings.Cut(after, ">")
	if !ok || namePart == "" {
		return ToolCall{}, false
	}
	name := strings.TrimSpace(namePart)
	fn := rest
	if j := strings.LastIndex(rest, "</function>"); j >= 0 {
		fn = rest[:j]
	} else {
		return ToolCall{}, false
	}
	types := chatcore.ParamTypes(name, toCoreTools(tools))

	var buf bytes.Buffer
	buf.WriteByte('{')
	n := 0
	for {
		_, afterParam, ok := strings.Cut(fn, "<parameter=")
		if !ok {
			break
		}
		keyPart, afterKey, ok := strings.Cut(afterParam, ">")
		if !ok || keyPart == "" {
			return ToolCall{}, false
		}
		key := strings.TrimSpace(keyPart)
		fn = afterKey
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
		buf.Write(chatcore.TypedParam(val, types[key]))
		n++
	}
	buf.WriteByte('}')
	return ToolCall{Name: name, Arguments: json.RawMessage(buf.Bytes())}, true
}

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
