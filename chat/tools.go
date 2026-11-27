package chat

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
	"github.com/townsendmerino/goinfer/internal/chat/harmony"
)

// Tool calling. Each family declares tools in the prompt, emits a call, and feeds a result back in its own syntax, so goinfer renders
// and PARSES against the model's template rather than scanning for JSON. RenderTools builds the tool-aware prompt; ParseToolCalls
// turns the model's output back into structured calls. Supported: ChatML/Qwen (Hermes <tool_call>, or Qwen3.5's XML), Mistral
// ([TOOL_CALLS]), Llama-3 (bare {name,parameters}), Gemma 4 (its <|tool_call> micro-language) and Harmony.

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
// tool support wired or no tools are given. It is the concatenation of RenderToolsSegments.
func (t *Template) RenderTools(system string, turns []Turn, tools []Tool) string {
	var sb strings.Builder
	for _, s := range t.RenderToolsSegments(system, turns, tools) {
		sb.WriteString(s.Text)
	}
	return sb.String()
}

// RenderToolsSegments is RenderTools for the EncodeSegments path: the family's control tokens are Special spans and everything a user,
// a tool, the system prompt or the model wrote is a non-special span, so a marker typed into a message stays literal text instead of
// forging a turn. With no tools it delegates to RenderSegments.
func (t *Template) RenderToolsSegments(system string, turns []Turn, tools []Tool) []Segment {
	if len(tools) == 0 {
		return t.RenderSegments(system, turns)
	}
	system = t.systemOr(system)
	switch t.name {
	case "chatml", "mellum2":
		var segs []Segment
		if t.bos != "" {
			segs = append(segs, Segment{Text: t.bos, Special: true})
		}
		if t.usesNativeTools() {
			segs = append(segs, t.renderQwen35XMLTools(system, turns, tools, t.historyKind())...)
		} else {
			segs = append(segs, t.renderChatMLTools(system, turns, tools, t.historyKind(), t.groupsToolResults)...)
		}
		return append(segs, t.thinkSuffixSegs()...)
	case "mistral":
		return t.renderMistralTools(system, turns, tools)
	case "llama3":
		return t.renderLlama3Tools(system, turns, tools)
	case "gemma4":
		if t.usesNativeTools() {
			return t.renderGemma4NativeTools(system, turns, tools, t.gemma4Think())
		}
		return t.renderGemma4Tools(system, turns, tools, t.gemma4Think(), t.historyKind() == histGemma4 || t.reason != nil && t.reason.oldGemma4, t.reason != nil && t.reason.oldGemma4, t.reason.offScaffold())
	case "harmony": // already segment-built: the markers are special, everything a user or tool wrote is hardened content
		return harmony.Segments(timeNow(), t.harmonyEffort(), system, toCoreTurns(turns), toCoreTools(tools))
	case "flint":
		return t.renderFlintTools(system, turns, tools)
	case "inkling2":
		return t.inkling2Segments(system, turns, tools)
	}
	return t.RenderSegments(system, turns) // gemma3 etc.: no native tool template
}

// SupportsTools reports whether this family has a tool-calling template.
func (t *Template) SupportsTools() bool {
	switch t.name {
	case "chatml", "mellum2", "mistral", "llama3", "gemma4", "harmony", "flint", "inkling2":
		return true
	}
	return false
}

// harmonyEffort is a Harmony template's reasoning effort, "medium" (the template's own default) when none was set.
func (t *Template) harmonyEffort() string {
	if t.effort == "" {
		return "medium"
	}
	return t.effort
}

// ToolCallWrapper returns how a single tool call is framed for this family: the
// literal prefix/suffix around the JSON, the arguments key ("arguments" or
// "parameters"), and whether the call object is wrapped in a one-element array
// (Mistral). ok is false when the family has no JSON-constrainable call form
// (Gemma 4's bespoke micro-language → parse-only, no logit constraint).
func (t *Template) ToolCallWrapper() (prefix, suffix, argsKey string, array, ok bool) {
	switch t.name {
	case "chatml", "mellum2":
		if t.usesNativeTools() {
			return "", "", "", false, false
		}
		return "<tool_call>\n", "\n</tool_call>", "arguments", false, true
	case "llama3":
		return "", "", "parameters", false, true
	case "mistral":
		return "[TOOL_CALLS] ", "", "arguments", true, true
	}
	return "", "", "", false, false
}

// ParseToolCalls extracts the tool calls from the model's output for this family,
// returning them plus any leading natural-language text (the content).
func (t *Template) ParseToolCalls(out string) ([]ToolCall, string) {
	switch t.name {
	case "chatml", "mellum2":
		return parseChatMLTools(out, nil)
	case "mistral":
		return parseMistralTools(out)
	case "llama3":
		return parseLlama3Tools(out)
	case "gemma4":
		return parseGemma4Tools(out)
	case "harmony":
		return parseHarmonyTools(out)
	case "flint":
		return parseFlintTools(out, nil)
	case "inkling2":
		return parseInklingTools(out)
	}
	return nil, out
}

// ParseToolCallsFor is ParseToolCalls with the request's tool list in hand, which lets it recover a bare or lenient fenced call.
func (t *Template) ParseToolCallsFor(out string, tools []Tool) ([]ToolCall, string) {
	var calls []ToolCall
	var lead string
	switch {
	case t != nil && (t.name == "chatml" || t.name == "mellum2"):
		calls, lead = parseChatMLTools(out, tools)
	case t != nil && t.name == "flint":
		calls, lead = parseFlintTools(out, tools)
	default:
		calls, lead = t.ParseToolCalls(out)
	}
	if len(calls) > 0 || !t.AcceptsBareToolCall() {
		return calls, lead
	}
	if c, ok := chatcore.BareToolCall(out, toCoreTools(tools)); ok {
		return []ToolCall{fromCoreCall(c)}, ""
	}
	if t.lenientFenced {
		if c, fenceLead, ok := fencedToolCall(out, tools); ok {
			return []ToolCall{c}, fenceLead
		}
	}
	return calls, lead
}

// AcceptsBareToolCall reports whether ParseToolCallsFor also accepts an unwrapped
// call for this family.
func (t *Template) AcceptsBareToolCall() bool {
	switch t.name {
	case "chatml", "mellum2":
		return true
	}
	return false
}

func funcDefJSON(t Tool) string {
	return chatcore.FuncDefJSON(chatcore.Tool{Name: t.Name, Description: t.Description, Parameters: t.Parameters})
}

func callObjectJSON(c ToolCall, argsKey string) string {
	return chatcore.CallObjectJSON(chatcore.ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments}, argsKey)
}

const (
	jStr = chatcore.JStr
	jObj = chatcore.JObj
)

var (
	jsonStr          = chatcore.JSONStr
	apostropheEscape = chatcore.ApostropheEscape
	parseJV          = chatcore.ParseJV
	pyDumps          = chatcore.PyDumps
	pyQuote          = chatcore.PyQuote
	pyArgs           = chatcore.PyArgs
)

func parseHarmonyTools(out string) ([]ToolCall, string) {
	calls, lead := harmony.ParseTools(out)
	return fromCoreCalls(calls), lead
}

// --- ChatML / Qwen (Hermes) ---

var chatMLToolMarks = []string{"<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"}

func (t *Template) renderChatMLTools(system string, turns []Turn, tools []Tool, hist histKind, group bool) []Segment {
	b := t.newToolBuf(chatMLToolMarks...)
	b.Lit("<|im_start|>system\n")
	if s := strings.TrimSpace(system); s != "" {
		b.Txt(s + "\n\n")
	}
	b.Lit("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
	b.Lit("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for _, tl := range tools {
		b.Txt(funcDefJSON(tl) + "\n")
	}
	b.Lit("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	lq := lastQueryIndex(turns)
	for i, m := range turns {
		switch m.Role {
		case "assistant":
			if hist != histNone {
				block, reasoning, content := qwenAssistant(hist, m, i, lq, len(turns))
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
					b.Txt(callObjectJSON(c, "arguments"))
					b.Lit("\n</tool_call>")
				}
				b.Lit("<|im_end|>\n")
				continue
			}
			b.Lit("<|im_start|>assistant\n")
			b.Txt(m.Content)
			for _, c := range m.ToolCalls {
				b.Lit("\n<tool_call>\n")
				b.Txt(callObjectJSON(c, "arguments"))
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

func parseChatMLTools(out string, tools []Tool) ([]ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<tool_call>"); ok {
		lead = before
	}
	var calls []ToolCall
	for rest := out; ; {
		_, after, ok := strings.Cut(rest, "<tool_call>")
		if !ok {
			break
		}
		body, next, hasEnd := strings.Cut(after, "</tool_call>")
		rest = next
		if c, ok := chatcore.CallFromJSON(strings.TrimSpace(body), "arguments"); ok {
			calls = append(calls, fromCoreCall(c))
		} else if strings.Contains(body, "<function=") {
			if c, ok := parseQwenXMLCall(body, tools); ok {
				calls = append(calls, c)
			}
		}
		if !hasEnd {
			break
		}
	}
	return calls, strings.TrimSpace(lead)
}

// --- Mistral ---

func (t *Template) renderMistralTools(system string, turns []Turn, tools []Tool) []Segment {
	b := t.newToolBuf(mistralMarks...)
	b.Lit("<s>[AVAILABLE_TOOLS] [")
	for i, tl := range tools {
		if i > 0 {
			b.Txt(", ")
		}
		b.Txt(funcDefJSON(tl))
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
					b.Txt(callObjectJSON(c, "arguments"))
				}
				b.Lit("]</s>")
			} else {
				b.Txt(" " + m.Content)
				b.Lit("</s>")
			}
		case "tool":
			b.Lit("[TOOL_RESULTS] {\"content\": ")
			b.Txt(jsonStr(m.Content))
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
	return fromCoreCalls(chatcore.CallsFromArray(rest, "arguments")), lead
}

// --- Llama-3 ---

var llama3ToolMarks = []string{"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}

func (t *Template) renderLlama3Tools(system string, turns []Turn, tools []Tool) []Segment {
	b := t.newToolBuf(llama3ToolMarks...)
	b.Lit("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
	b.Lit("Environment: ipython\nCutting Knowledge Date: December 2023\nToday Date: " + timeNow().Format("02 Jan 2006") + "\n\n")
	if s := strings.TrimSpace(system); s != "" {
		b.Txt(s)
	}
	b.Lit("<|eot_id|>")
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
			b.Lit("<|start_header_id|>assistant<|end_header_id|>\n\n")
			b.Txt(m.Content)
			for _, c := range m.ToolCalls {
				b.Txt(callObjectJSON(c, "parameters"))
			}
			b.Lit("<|eot_id|>")
		case "tool":
			b.Lit("<|start_header_id|>ipython<|end_header_id|>\n\n")
			b.Txt(jsonStr(m.Content))
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

func parseLlama3Tools(out string) ([]ToolCall, string) {
	s := strings.TrimPrefix(strings.TrimSpace(out), "<|python_tag|>")
	if lb := strings.IndexByte(s, '{'); lb >= 0 {
		if c, ok := chatcore.CallFromJSON(s[lb:], "parameters"); ok {
			return []ToolCall{fromCoreCall(c)}, strings.TrimSpace(s[:lb])
		}
	}
	return nil, strings.TrimSpace(out)
}

// --- Flint ---

var flintToolMarks = []string{
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

func flintText(s string) string {
	return flintSanitizer.Replace(s)
}

func flintToolJSON(t Tool) string {
	var b strings.Builder
	b.WriteString(`{"name": ` + pyQuote(t.Name))
	if t.Description != "" {
		b.WriteString(`, "description": ` + pyQuote(t.Description))
	}
	if len(bytes.TrimSpace(t.Parameters)) > 0 {
		b.WriteString(`, "parameters": ` + pyArgs(t.Parameters))
	}
	b.WriteString("}")
	return b.String()
}

func (t *Template) renderFlintTools(system string, turns []Turn, tools []Tool) []Segment {
	b := t.newToolBuf(flintToolMarks...)
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
				if args, err := parseJV(c.Arguments); err == nil && args.Kind == jObj {
					for _, k := range args.Keys {
						b.Lit("<arg_key>")
						b.Txt(k)
						b.Lit("</arg_key><arg_value>")
						v := args.Vals[k]
						if v.Kind == jStr {
							b.Txt(v.S)
						} else {
							b.Txt(pyDumps(v))
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

func parseFlintTools(out string, tools []Tool) ([]ToolCall, string) {
	lead := out
	if before, _, ok := strings.Cut(out, "<tool_call>"); ok {
		lead = before
	}
	var calls []ToolCall
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

func parseFlintCall(body string, tools []Tool) (ToolCall, bool) {
	var name, rest string
	if ak := strings.Index(body, "<arg_key>"); ak >= 0 {
		name = strings.TrimSpace(body[:ak])
		rest = body[ak:]
	} else {
		name = strings.TrimSpace(body)
	}
	if name == "" || strings.ContainsAny(name, " \t\r\n<>\"'{}") {
		return ToolCall{}, false
	}
	types := chatcore.ParamTypes(name, toCoreTools(tools))
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
			return ToolCall{}, false
		}
		key, afterClose, ok := strings.Cut(afterKey, "</arg_key>")
		if !ok || key == "" {
			return ToolCall{}, false
		}
		rest, ok = strings.CutPrefix(strings.TrimSpace(afterClose), "<arg_value>")
		if !ok {
			return ToolCall{}, false
		}
		ve := closeFlintArgValue(rest)
		if ve < 0 {
			return ToolCall{}, false
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
	return ToolCall{Name: name, Arguments: json.RawMessage(buf.Bytes())}, true
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

// ToolCallOpener reports the literal that bounds the prose lead for this family, and whether prose may be streamed INCREMENTALLY
// before the generation ends.
func (t *Template) ToolCallOpener() (string, bool) {
	switch t.name {
	case "chatml", "mellum2", "flint":
		return "<tool_call>", true
	case "gemma4":
		return "<|tool_call>", true
	}
	return "", false
}

// StreamableLen returns how many bytes of pending may be emitted now without
// risking that a later tool-call opener makes them part of a call rather than
// prose.
func StreamableLen(pending, opener string) int {
	if opener == "" {
		return len(pending)
	}
	if i := strings.Index(pending, opener); i >= 0 {
		return i
	}
	maxHold := min(len(opener)-1, len(pending))
	for k := maxHold; k > 0; k-- {
		if strings.HasSuffix(pending, opener[:k]) {
			return len(pending) - k
		}
	}
	return len(pending)
}

// ProseStreamer releases the prose part of a tool-capable generation as it arrives, holding back exactly what could still turn out not
// to be prose.
type ProseStreamer struct {
	opener     string
	bareAware  bool
	fenceAware bool
	pending    strings.Builder
	started    bool
	done       bool
	held       bool
}

// NewProseStreamer returns a streamer for a family's opener (Template.ToolCallOpener).
func NewProseStreamer(opener string) *ProseStreamer { return &ProseStreamer{opener: opener} }

// NewBareAwareProseStreamer is NewProseStreamer for a family that also accepts a
// bare, unwrapped call (Template.AcceptsBareToolCall).
func NewBareAwareProseStreamer(opener string) *ProseStreamer {
	return &ProseStreamer{opener: opener, bareAware: true}
}

// FenceAware makes the streamer hold everything from the first code fence on, for a template with
// WithLenientToolCalls.
func (p *ProseStreamer) FenceAware() *ProseStreamer {
	p.fenceAware = true
	return p
}

// Done reports whether the opener has been seen, after which nothing more is prose.
func (p *ProseStreamer) Done() bool { return p.done }

// Push feeds the next generated chunk and returns the text that is safe to emit
// now — possibly empty.
func (p *ProseStreamer) Push(chunk string) string {
	if p.done || p.held {
		return ""
	}
	p.pending.WriteString(chunk)
	buf := p.pending.String()
	if p.bareAware && !p.started {
		if t := strings.TrimLeft(buf, " \t\r\n"); t != "" && t[0] == '{' {
			p.held = true
			return ""
		}
	}

	n := StreamableLen(buf, p.opener)
	fenced := false
	if p.fenceAware && !strings.Contains(buf, p.opener) {
		if m := StreamableLen(buf, "```"); m < n || strings.Contains(buf, "```") {
			n = min(n, m)
			fenced = strings.Contains(buf, "```")
		}
	}
	safe, rest := buf[:n], buf[n:]
	if strings.Contains(buf, p.opener) {
		p.done = true
		rest = ""
	} else if fenced {
		p.held = true
		rest = ""
	}

	if !p.started {
		safe = strings.TrimLeft(safe, " \t\r\n")
		if safe != "" {
			p.started = true
		}
	}
	if p.done || p.held {
		safe = strings.TrimRight(safe, " \t\r\n")
	} else if tail := len(safe) - len(strings.TrimRight(safe, " \t\r\n")); tail > 0 {
		rest = safe[len(safe)-tail:] + rest
		safe = safe[:len(safe)-tail]
	}

	p.pending.Reset()
	p.pending.WriteString(rest)
	return safe
}
