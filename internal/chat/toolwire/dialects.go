package toolwire

import (
	"bytes"
	"encoding/json"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

// MistralMarks lists Mistral's control-token strings, tool dialect included.
var MistralMarks = []string{"<s>", "</s>", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "[TOOL_CALLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[INST]", "[/INST]"}

// RenderMistral renders system + turns + tool declarations in Mistral's [AVAILABLE_TOOLS] format.
func RenderMistral(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, MistralMarks...)
	b.Lit("<s>[AVAILABLE_TOOLS] [")
	for i, tl := range tools {
		if i > 0 {
			b.Txt(", ")
		}
		b.Txt(chatcore.FuncDefJSON(tl))
	}
	b.Lit("][/AVAILABLE_TOOLS]")
	lastUser := -1
	for i, m := range turns {
		if m.Role == "user" {
			lastUser = i
		}
	}
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			if len(m.ToolCalls) > 0 {
				b.Lit("[TOOL_CALLS] [")
				for k, c := range m.ToolCalls {
					if k > 0 {
						b.Txt(", ")
					}
					b.Txt(chatcore.CallObjectJSON(c, "arguments"))
				}
				b.Lit("]</s>")
			} else {
				b.Txt(" " + m.Content)
				b.Lit("</s>")
			}
		case "tool":
			b.Lit("[TOOL_RESULTS] {\"content\": ")
			b.Txt(chatcore.JSONStr(m.Content))
			b.Lit("}[/TOOL_RESULTS]")
		default:
			content := m.Content
			if i == lastUser && strings.TrimSpace(system) != "" {
				content = strings.TrimSpace(system) + "\n\n" + content
			}
			b.Lit("[INST] ")
			b.Txt(content)
			b.Lit("[/INST]")
		}
	}
	return b.Out()
}

// ParseMistral parses Mistral [TOOL_CALLS] JSON array tool calls and leading prose.
func ParseMistral(out string) ([]chatcore.ToolCall, string) {
	before, after, ok := strings.Cut(out, "[TOOL_CALLS]")
	if !ok {
		return nil, strings.TrimSpace(out)
	}
	lead := strings.TrimSpace(before)
	rest := after
	if lb := strings.IndexByte(rest, '['); lb >= 0 {
		rest = rest[lb:]
	}
	return chatcore.CallsFromArray(rest, "arguments"), lead
}

var llama3Marks = []string{"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}

// RenderLlama3 renders system + turns + tool declarations in Llama-3's ipython JSON format.
func RenderLlama3(now time.Time, has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, llama3Marks...)
	b.Lit("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
	b.Lit("Environment: ipython\nCutting Knowledge Date: December 2023\nToday Date: " + now.Format("02 Jan 2006") + "\n\n")
	if s := strings.TrimSpace(system); s != "" {
		b.Txt(s)
	}
	b.Lit("<|eot_id|>")
	var instr strings.Builder
	instr.WriteString("Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n")
	for i, tl := range tools {
		instr.WriteString(chatcore.FuncDefJSON(tl))
		if i < len(tools)-1 {
			instr.WriteString("\n")
		}
	}
	first := true
	for _, m := range turns {
		switch m.Role {
		case "assistant":
			b.Lit("<|start_header_id|>assistant<|end_header_id|>\n\n")
			b.Txt(m.Content)
			for _, c := range m.ToolCalls {
				b.Txt(chatcore.CallObjectJSON(c, "parameters"))
			}
			b.Lit("<|eot_id|>")
		case "tool":
			b.Lit("<|start_header_id|>ipython<|end_header_id|>\n\n")
			b.Txt(chatcore.JSONStr(m.Content))
			b.Lit("<|eot_id|>")
		default:
			content := m.Content
			if first {
				content = instr.String() + "\n\n" + content
				first = false
			}
			b.Lit("<|start_header_id|>user<|end_header_id|>\n\n")
			b.Txt(content)
			b.Lit("<|eot_id|>")
		}
	}
	b.Lit("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.Out()
}

// ParseLlama3 parses Llama-3 {name, parameters} JSON tool calls and leading prose.
func ParseLlama3(out string) ([]chatcore.ToolCall, string) {
	s := strings.TrimPrefix(strings.TrimSpace(out), "<|python_tag|>")
	if lb := strings.IndexByte(s, '{'); lb >= 0 {
		if c, ok := chatcore.CallFromJSON(s[lb:], "parameters"); ok {
			return []chatcore.ToolCall{c}, strings.TrimSpace(s[:lb])
		}
	}
	return nil, strings.TrimSpace(out)
}

var flintMarks = []string{
	"<|begin_of_text|>", "<|end_of_text|>",
	"<|system|>", "<|developer|>", "<|user|>", "<|assistant|>", "<|observation|>",
	"<think>", "</think>",
	"<tools>", "</tools>",
	"<tool_call>", "</tool_call>",
	"<arg_key>", "</arg_key>",
	"<arg_value>", "</arg_value>",
	"<tool_response>", "</tool_response>",
}

var flintSanitizer = strings.NewReplacer(
	"\x00", "",
	"<think>", "",
	"</think>", "",
	"<|begin_of_text|>", "&lt;|begin_of_text|&gt;",
	"<|end_of_text|>", "&lt;|end_of_text|&gt;",
	"<|system|>", "&lt;|system|&gt;",
	"<|developer|>", "&lt;|developer|&gt;",
	"<|user|>", "&lt;|user|&gt;",
	"<|assistant|>", "&lt;|assistant|&gt;",
	"<|observation|>", "&lt;|observation|&gt;",
)

func flintText(s string) string { return flintSanitizer.Replace(s) }

func flintToolJSON(t chatcore.Tool) string {
	var b strings.Builder
	b.WriteString(`{"name": ` + chatcore.PyQuote(t.Name))
	if t.Description != "" {
		b.WriteString(`, "description": ` + chatcore.PyQuote(t.Description))
	}
	if len(bytes.TrimSpace(t.Parameters)) > 0 {
		b.WriteString(`, "parameters": ` + chatcore.PyArgs(t.Parameters))
	}
	b.WriteString("}")
	return b.String()
}

// RenderFlint renders system + turns + optional tool declarations in Flint's XML format.
func RenderFlint(has func(string) bool, system string, turns []chatcore.Turn, tools []chatcore.Tool) []chatcore.Segment {
	b := chatcore.NewToolBuf(has, flintMarks...)
	if system != "" {
		b.Lit("<|system|>")
		b.Txt(flintText(system))
	}
	toolsEmitted := len(tools) == 0
	emitTools := func() {
		if toolsEmitted {
			return
		}
		toolsEmitted = true
		b.Lit("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tl := range tools {
			b.Txt("\n" + flintToolJSON(tl))
		}
		b.Lit("\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n")
	}
	for _, m := range turns {
		if !toolsEmitted && m.Role != "system" {
			emitTools()
		}
		switch m.Role {
		case "system":
			b.Lit("<|system|>")
			b.Txt(flintText(m.Content))
		case "developer":
			b.Lit("<|developer|>")
			b.Txt(flintText(m.Content))
		case "assistant":
			b.Lit("<|assistant|>")
			reasoning, content := m.Reasoning, m.Content
			if reasoning == "" && strings.HasPrefix(content, "<think>") {
				if r, rest, ok := strings.Cut(content[len("<think>"):], "</think>"); ok {
					reasoning, content = r, rest
				}
			}
			if reasoning != "" {
				b.Lit("<think>")
				b.Txt(reasoning)
				b.Lit("</think>")
			}
			b.Txt(content)
			for _, c := range m.ToolCalls {
				b.Lit("<tool_call>")
				b.Txt(c.Name)
				if args, err := chatcore.ParseJV(c.Arguments); err == nil && args.Kind == chatcore.JObj {
					for _, k := range args.Keys {
						b.Lit("<arg_key>")
						b.Txt(k)
						b.Lit("</arg_key><arg_value>")
						v := args.Vals[k]
						if v.Kind == chatcore.JStr {
							b.Txt(v.S)
						} else {
							b.Txt(chatcore.PyDumps(v))
						}
						b.Lit("</arg_value>")
					}
				}
				b.Lit("</tool_call>")
			}
		case "tool":
			b.Lit("<|observation|><tool_response>")
			b.Txt(flintText(m.Content))
			b.Lit("</tool_response>")
		default:
			b.Lit("<|user|>")
			b.Txt(flintText(m.Content))
		}
	}
	if !toolsEmitted {
		emitTools()
	}
	b.Lit("<|assistant|>")
	return b.Out()
}

// ParseFlint parses Flint <tool_call>NAME<arg_key>...</arg_value></tool_call> blocks and leading prose.
func ParseFlint(out string, tools []chatcore.Tool) ([]chatcore.ToolCall, string) {
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
		body, next, ok := strings.Cut(after, "</tool_call>")
		if !ok {
			break
		}
		rest = next
		if c, ok := parseFlintCall(body, tools); ok {
			calls = append(calls, c)
		}
	}
	return calls, strings.TrimSpace(lead)
}

func parseFlintCall(body string, tools []chatcore.Tool) (chatcore.ToolCall, bool) {
	var name, rest string
	if ak := strings.Index(body, "<arg_key>"); ak >= 0 {
		name = strings.TrimSpace(body[:ak])
		rest = body[ak:]
	} else {
		name = strings.TrimSpace(body)
	}
	if name == "" || strings.ContainsAny(name, " \t\r\n<>\"'{}") {
		return chatcore.ToolCall{}, false
	}
	types := chatcore.ParamTypes(name, tools)
	var buf bytes.Buffer
	buf.WriteByte('{')
	n := 0
	for rest != "" {
		rest = strings.TrimSpace(rest)
		if rest == "" {
			break
		}
		afterKey, ok := strings.CutPrefix(rest, "<arg_key>")
		if !ok {
			return chatcore.ToolCall{}, false
		}
		key, afterClose, ok := strings.Cut(afterKey, "</arg_key>")
		if !ok || key == "" {
			return chatcore.ToolCall{}, false
		}
		rest, ok = strings.CutPrefix(strings.TrimSpace(afterClose), "<arg_value>")
		if !ok {
			return chatcore.ToolCall{}, false
		}
		ve := closeFlintArgValue(rest)
		if ve < 0 {
			return chatcore.ToolCall{}, false
		}
		val := rest[:ve]
		rest = rest[ve+len("</arg_value>"):]
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

func closeFlintArgValue(s string) int {
	const tag = "</arg_value>"
	for from := 0; ; {
		i := strings.Index(s[from:], tag)
		if i < 0 {
			return -1
		}
		i += from
		next := strings.TrimSpace(s[i+len(tag):])
		if next == "" || strings.HasPrefix(next, "<arg_key>") {
			return i
		}
		from = i + len(tag)
	}
}

// FencedToolCall parses out as a single schema-validated JSON tool call inside a trailing code fence.
func FencedToolCall(out string, tools []chatcore.Tool) (chatcore.ToolCall, string, bool) {
	trimmed := strings.TrimRight(out, " \t\r\n")
	if !strings.HasSuffix(trimmed, "```") || strings.Count(trimmed, "```") != 2 {
		return chatcore.ToolCall{}, "", false
	}
	open := strings.Index(trimmed, "```")
	body := trimmed[open+3 : len(trimmed)-3]
	before, after, ok0 := strings.Cut(body, "\n")
	if !ok0 {
		return chatcore.ToolCall{}, "", false
	}
	if tag := strings.TrimSpace(before); tag != "" && !strings.EqualFold(tag, "json") {
		return chatcore.ToolCall{}, "", false
	}
	content := after
	var raw map[string]json.RawMessage
	dec := json.NewDecoder(strings.NewReader(content))
	if dec.Decode(&raw) != nil || raw == nil || strings.TrimSpace(content[dec.InputOffset():]) != "" {
		return chatcore.ToolCall{}, "", false
	}
	for k := range raw {
		if k != "name" && k != "arguments" && k != "parameters" && k != "id" {
			return chatcore.ToolCall{}, "", false
		}
	}
	var name string
	if json.Unmarshal(raw["name"], &name) != nil {
		return chatcore.ToolCall{}, "", false
	}
	var tool *chatcore.Tool
	for i := range tools {
		if tools[i].Name == name {
			tool = &tools[i]
			break
		}
	}
	if tool == nil {
		return chatcore.ToolCall{}, "", false
	}
	args, ok := raw["arguments"]
	if !ok {
		if args, ok = raw["parameters"]; !ok {
			return chatcore.ToolCall{}, "", false
		}
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(args, &obj) != nil || obj == nil || !chatcore.ArgsMatchSchema(obj, tool.Parameters) {
		return chatcore.ToolCall{}, "", false
	}
	var id string
	if v, ok := raw["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	return chatcore.ToolCall{ID: id, Name: name, Arguments: args}, strings.TrimSpace(trimmed[:open]), true
}
