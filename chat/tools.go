package chat

import (
	"encoding/json"
	"strings"
)

// Tool calling. Each family declares tools in the prompt, emits a call, and feeds
// a result back in its own syntax — so (per Ollama's lesson) we render and PARSE
// against the model's template, not by blindly scanning for JSON. RenderTools
// builds the tool-aware prompt; ParseToolCalls turns the model's output back into
// structured calls. Supported: ChatML/Qwen (Hermes <tool_call>), Mistral
// ([TOOL_CALLS]), Llama-3 (bare {name,parameters}), and Gemma 4 (its bespoke
// <|tool_call> micro-language).

// Tool is a function the model may call (OpenAI "function" shape).
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage // JSON Schema for the arguments object
}

// ToolCall is one call the model emitted (or that we render into history).
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage // a JSON object
}

// RenderTools renders system + turns + tool declarations into the prompt, ending
// with the generation prompt. Falls back to plain Render when the family has no
// tool support wired or no tools are given.
func (t *Template) RenderTools(system string, turns []Turn, tools []Tool) string {
	if len(tools) == 0 {
		return t.Render(system, turns)
	}
	switch t.name {
	case "chatml", "mellum2":
		return renderChatMLTools(system, turns, tools)
	case "mistral":
		return renderMistralTools(system, turns, tools)
	case "llama3":
		return renderLlama3Tools(system, turns, tools)
	case "gemma4":
		return renderGemma4Tools(system, turns, tools)
	}
	return t.Render(system, turns) // gemma3 etc.: no native tool template
}

// SupportsTools reports whether this family has a tool-calling template.
func (t *Template) SupportsTools() bool {
	switch t.name {
	case "chatml", "mellum2", "mistral", "llama3", "gemma4":
		return true
	}
	return false
}

// ToolCallWrapper returns how a single tool call is framed for this family: the
// literal prefix/suffix around the JSON, the arguments key ("arguments" or
// "parameters"), and whether the call object is wrapped in a one-element array
// (Mistral). ok is false when the family has no JSON-constrainable call form
// (Gemma 4's bespoke micro-language → parse-only, no logit constraint).
func (t *Template) ToolCallWrapper() (prefix, suffix, argsKey string, array, ok bool) {
	switch t.name {
	case "chatml", "mellum2":
		return "<tool_call>\n", "\n</tool_call>", "arguments", false, true
	case "llama3":
		return "", "", "parameters", false, true
	case "mistral":
		return "[TOOL_CALLS] ", "", "arguments", true, true
	}
	return "", "", "", false, false
}

// ParseToolCalls extracts the tool calls from the model's output for this family,
// returning them plus any leading natural-language text (the content). Empty
// calls means the model answered normally.
func (t *Template) ParseToolCalls(out string) ([]ToolCall, string) {
	switch t.name {
	case "chatml", "mellum2":
		return parseChatMLTools(out)
	case "mistral":
		return parseMistralTools(out)
	case "llama3":
		return parseLlama3Tools(out)
	case "gemma4":
		return parseGemma4Tools(out)
	}
	return nil, out
}

// funcDefJSON renders a tool as the OpenAI function-definition JSON object that
// the JSON families embed in the prompt.
func funcDefJSON(t Tool) string {
	m := map[string]any{"type": "function", "function": map[string]any{
		"name": t.Name, "description": t.Description, "parameters": json.RawMessage(t.Parameters),
	}}
	b, _ := json.Marshal(m)
	return string(b)
}

func jsonStr(s string) string { b, _ := json.Marshal(s); return string(b) }

// callObjectJSON renders {"name":..,"arguments":{...}} (argsKey lets Llama-3 use
// "parameters"); used to render assistant tool-call history.
func callObjectJSON(c ToolCall, argsKey string) string {
	args := string(c.Arguments)
	if args == "" {
		args = "{}"
	}
	return `{"name": ` + jsonStr(c.Name) + `, "` + argsKey + `": ` + args + `}`
}

// --- ChatML / Qwen (Hermes) ---

func renderChatMLTools(system string, turns []Turn, tools []Tool) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	if s := strings.TrimSpace(system); s != "" {
		b.WriteString(s + "\n\n")
	}
	b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
	b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for _, tl := range tools {
		b.WriteString(funcDefJSON(tl) + "\n")
	}
	b.WriteString("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	for _, m := range turns {
		switch m.Role {
		case "assistant":
			b.WriteString("<|im_start|>assistant\n")
			b.WriteString(m.Content)
			for _, c := range m.ToolCalls {
				b.WriteString("\n<tool_call>\n" + callObjectJSON(c, "arguments") + "\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			b.WriteString("<|im_start|>user\n<tool_response>\n" + m.Content + "\n</tool_response><|im_end|>\n")
		default:
			b.WriteString("<|im_start|>user\n" + m.Content + "<|im_end|>\n")
		}
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func parseChatMLTools(out string) ([]ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<tool_call>"); ok {
		lead = before
	}
	var calls []ToolCall
	for rest := out; ; {
		i := strings.Index(rest, "<tool_call>")
		if i < 0 {
			break
		}
		rest = rest[i+len("<tool_call>"):]
		j := strings.Index(rest, "</tool_call>")
		body := rest
		if j >= 0 {
			body = rest[:j]
			rest = rest[j+len("</tool_call>"):]
		}
		if c, ok := callFromJSON(strings.TrimSpace(body), "arguments"); ok {
			calls = append(calls, c)
		}
		if j < 0 {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// --- Mistral ---

func renderMistralTools(system string, turns []Turn, tools []Tool) string {
	var b strings.Builder
	b.WriteString("<s>[AVAILABLE_TOOLS] [")
	for i, tl := range tools {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(funcDefJSON(tl))
	}
	b.WriteString("][/AVAILABLE_TOOLS]")
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
				b.WriteString("[TOOL_CALLS] [")
				for k, c := range m.ToolCalls {
					if k > 0 {
						b.WriteString(", ")
					}
					b.WriteString(callObjectJSON(c, "arguments"))
				}
				b.WriteString("]</s>")
			} else {
				b.WriteString(" " + m.Content + "</s>")
			}
		case "tool":
			b.WriteString("[TOOL_RESULTS] {\"content\": " + jsonStr(m.Content) + "}[/TOOL_RESULTS]")
		default:
			content := m.Content
			if i == lastUser && strings.TrimSpace(system) != "" {
				content = strings.TrimSpace(system) + "\n\n" + content
			}
			b.WriteString("[INST] " + content + "[/INST]")
		}
	}
	return b.String()
}

func parseMistralTools(out string) ([]ToolCall, string) {
	before, after, ok := strings.Cut(out, "[TOOL_CALLS]")
	if !ok {
		return nil, strings.TrimSpace(out)
	}
	lead := strings.TrimSpace(before)
	rest := after
	if lb := strings.IndexByte(rest, '['); lb >= 0 {
		rest = rest[lb:]
	}
	return callsFromArray(rest, "arguments"), lead
}

// --- Llama-3 ---

func renderLlama3Tools(system string, turns []Turn, tools []Tool) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
	b.WriteString("Environment: ipython\nCutting Knowledge Date: December 2023\nToday Date: " + timeNow().Format("02 Jan 2006") + "\n\n")
	if s := strings.TrimSpace(system); s != "" {
		b.WriteString(s)
	}
	b.WriteString("<|eot_id|>")
	var instr strings.Builder
	instr.WriteString("Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n")
	for i, tl := range tools {
		instr.WriteString(funcDefJSON(tl))
		if i < len(tools)-1 {
			instr.WriteString("\n")
		}
	}
	first := true
	for _, m := range turns {
		switch m.Role {
		case "assistant":
			b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n" + m.Content)
			for _, c := range m.ToolCalls {
				b.WriteString(callObjectJSON(c, "parameters"))
			}
			b.WriteString("<|eot_id|>")
		case "tool":
			b.WriteString("<|start_header_id|>ipython<|end_header_id|>\n\n" + jsonStr(m.Content) + "<|eot_id|>")
		default:
			content := m.Content
			if first { // tool instructions ride the first user turn
				content = instr.String() + "\n\n" + content
				first = false
			}
			b.WriteString("<|start_header_id|>user<|end_header_id|>\n\n" + content + "<|eot_id|>")
		}
	}
	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

func parseLlama3Tools(out string) ([]ToolCall, string) {
	s := strings.TrimPrefix(strings.TrimSpace(out), "<|python_tag|>")
	if lb := strings.IndexByte(s, '{'); lb >= 0 {
		if c, ok := callFromJSON(s[lb:], "parameters"); ok {
			return []ToolCall{c}, strings.TrimSpace(s[:lb])
		}
	}
	return nil, strings.TrimSpace(out)
}

// --- shared JSON-call parsing ---

// callFromJSON parses {"name":..,"<argsKey>":{...}} into a ToolCall. Decodes the
// first JSON object (ignoring trailing content like an EOS marker), and is
// tolerant of "arguments"/"parameters" and a stringified arguments value.
func callFromJSON(s, argsKey string) (ToolCall, bool) {
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
	return ToolCall{ID: id, Name: name, Arguments: normalizeArgs(args)}, true
}

// callsFromArray parses [{...},{...}] of call objects (first JSON array, trailing
// content ignored).
func callsFromArray(s, argsKey string) []ToolCall {
	var arr []json.RawMessage
	if json.NewDecoder(strings.NewReader(s)).Decode(&arr) != nil {
		return nil
	}
	var calls []ToolCall
	for _, e := range arr {
		if c, ok := callFromJSON(string(e), argsKey); ok {
			calls = append(calls, c)
		}
	}
	return calls
}

// normalizeArgs ensures the arguments are a JSON object ({} if absent; unwraps a
// stringified object, which some models emit).
func normalizeArgs(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	var s string
	if json.Unmarshal(raw, &s) == nil { // "{\"k\":..}" → {"k":..}
		var obj map[string]any
		if json.Unmarshal([]byte(s), &obj) == nil {
			return json.RawMessage(s)
		}
	}
	return raw
}
