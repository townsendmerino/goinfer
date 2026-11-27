package toolwire

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

var chatMLMarks = []string{"<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"}

// RenderChatML renders system + turns + tool declarations in Hermes/ChatML format.
func RenderChatML(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool, hist chatcore.HistKind, group bool) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, chatMLMarks...)
	b.Lit("<|im_start|>system\n")
	if s := strings.TrimSpace(system); s != "" {
		b.Txt(s + "\n\n")
	}
	b.Lit("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
	b.Lit("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for _, tl := range tools {
		b.Txt(chatcore.FuncDefJSON(tl) + "\n")
	}
	b.Lit("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	lq := chatcore.LastQueryIndex(turns)
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			if hist != chatcore.HistNone {
				block, reasoning, content := chatcore.QwenAssistant(hist, m, i, lq, len(turns))
				b.Lit("<|im_start|>assistant\n")
				if block {
					b.Lit("<think>")
					b.Txt("\n" + reasoning + "\n")
					b.Lit("</think>\n\n")
				}
				b.Txt(content)
				for ci, c := range m.ToolCalls {
					if (ci == 0 && content != "") || ci > 0 {
						b.Txt("\n")
					}
					b.Lit("<tool_call>\n")
					b.Txt(chatcore.CallObjectJSON(c, "arguments"))
					b.Lit("\n</tool_call>")
				}
				b.Lit("<|im_end|>\n")
				continue
			}
			b.Lit("<|im_start|>assistant\n")
			b.Txt(m.Content)
			for _, c := range m.ToolCalls {
				b.Lit("\n<tool_call>\n")
				b.Txt(chatcore.CallObjectJSON(c, "arguments"))
				b.Lit("\n</tool_call>")
			}
			b.Lit("<|im_end|>\n")
		case "tool":
			if group {
				if i == 0 || turns[i-1].Role != "tool" {
					b.Lit("<|im_start|>user")
				}
				b.Lit("\n<tool_response>\n")
				b.Txt(m.Content)
				b.Lit("\n</tool_response>")
				if i == len(turns)-1 || turns[i+1].Role != "tool" {
					b.Lit("<|im_end|>\n")
				}
				continue
			}
			b.Lit("<|im_start|>user\n<tool_response>\n")
			b.Txt(m.Content)
			b.Lit("\n</tool_response><|im_end|>\n")
		default:
			b.Lit("<|im_start|>user\n")
			b.Txt(m.Content)
			b.Lit("<|im_end|>\n")
		}
	}
	b.Lit("<|im_start|>assistant\n")
	return b.Out()
}

// ParseChatML parses Hermes JSON or Qwen3.5 XML <tool_call> blocks and leading prose from a ChatML reply.
func ParseChatML(out string, tools []chatcore.Tool) ([]chatcore.ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<tool_call>"); ok {
		lead = before
	}
	var calls []chatcore.ToolCall
	for rest := out; ; {
		_, after, ok := strings.Cut(rest, "<tool_call>")
		if !ok {
			break
		}
		body, next, hasEnd := strings.Cut(after, "</tool_call>")
		rest = next
		if c, ok := chatcore.CallFromJSON(strings.TrimSpace(body), "arguments"); ok {
			calls = append(calls, c)
		} else if strings.Contains(body, "<function=") {
			if c, ok := ParseQwenXMLCall(body, tools); ok {
				calls = append(calls, c)
			}
		}
		if !hasEnd {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// DeclaresQwen35XML reports whether a ChatML template declares Qwen3.5's XML tool-calling layout.
func DeclaresQwen35XML(tmpl string, hist chatcore.HistKind) bool {
	return hist == chatcore.HistQwen35 && strings.Contains(tmpl, "<function=") && strings.Contains(tmpl, "<parameter=")
}

const (
	qwenXMLToolsPrologue = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
	qwenXMLToolsEpilogue = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

func qwenToolJSON(t chatcore.Tool) string {
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

// RenderQwen35XML renders system + turns + tool declarations in Qwen3.5's native XML tool format.
func RenderQwen35XML(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool, hist chatcore.HistKind) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, chatMLMarks...)
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

	lq := chatcore.LastQueryIndex(turns)
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			block, reasoning, content := chatcore.QwenAssistant(hist, m, i, lq, len(turns))
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

// ParseQwenXMLCall parses a single Qwen3.5 <function=NAME><parameter=KEY>...</parameter></function> body.
func ParseQwenXMLCall(body string, tools []chatcore.Tool) (chatcore.ToolCall, bool) {
	_, after, ok := strings.Cut(body, "<function=")
	if !ok {
		return chatcore.ToolCall{}, false
	}
	namePart, rest, ok := strings.Cut(after, ">")
	if !ok || namePart == "" {
		return chatcore.ToolCall{}, false
	}
	name := strings.TrimSpace(namePart)
	fn := rest
	if j := strings.LastIndex(rest, "</function>"); j >= 0 {
		fn = rest[:j]
	} else {
		return chatcore.ToolCall{}, false
	}
	types := chatcore.ParamTypes(name, tools)

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
			return chatcore.ToolCall{}, false
		}
		key := strings.TrimSpace(keyPart)
		fn = afterKey
		c := closeQwenParam(fn)
		if c < 0 {
			return chatcore.ToolCall{}, false
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
	return chatcore.ToolCall{Name: name, Arguments: json.RawMessage(buf.Bytes())}, true
}

func closeQwenParam(s string) int {
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
