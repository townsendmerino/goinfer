package chat

import (
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
	"github.com/townsendmerino/goinfer/internal/chat/harmony"
	"github.com/townsendmerino/goinfer/internal/chat/toolwire"
)

// Tool is a function the model may call (OpenAI "function" shape).
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ToolCall is one call the model emitted or that is replayed into history.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

type ToolFormat = chatcore.ToolFormat

const (
	ToolFormatAuto     = chatcore.ToolFormatAuto
	ToolFormatHermes   = chatcore.ToolFormatHermes
	ToolFormatTemplate = chatcore.ToolFormatTemplate
)

var ParseToolFormat = chatcore.ParseToolFormat

// ProseStreamer incrementally releases prose ahead of a tool-call opener while holding back partial opener prefixes.
type ProseStreamer = toolwire.ProseStreamer

var (
	StreamableLen             = toolwire.StreamableLen
	NewProseStreamer          = toolwire.NewProseStreamer
	NewBareAwareProseStreamer = toolwire.NewBareAwareProseStreamer
	apostropheEscape          = chatcore.ApostropheEscape
)

func declaresQwen35XMLTools(tmpl string, r *Reasoning) bool {
	if r == nil {
		return false
	}
	return toolwire.DeclaresQwen35XML(tmpl, r.hist)
}

// WithToolFormat returns a copy of t that renders tools in format f.
func (t *Template) WithToolFormat(f ToolFormat) *Template {
	if t == nil || !t.nativeTools || t.toolFormat == f {
		return t
	}
	c := *t
	c.toolFormat = f
	return &c
}

// ToolFormat returns the tool format selected on t (ToolFormatAuto by default).
func (t *Template) ToolFormat() ToolFormat {
	if t == nil {
		return ToolFormatAuto
	}
	return t.toolFormat
}

// UsesNativeTools reports whether tools are rendered in the model's native template format.
func (t *Template) UsesNativeTools() bool { return t.usesNativeTools() }

// DeclaresNativeTools reports whether t declares a native tool format goinfer can render.
func (t *Template) DeclaresNativeTools() bool { return t != nil && t.nativeTools }

func (t *Template) usesNativeTools() bool {
	if t == nil || !t.nativeTools {
		return false
	}
	return t.toolFormat == ToolFormatTemplate || (t.toolFormat == ToolFormatAuto && t.nativeByDefault && t.think != ThinkAsIs)
}

// RenderTools renders system + turns + tool declarations into a prompt ending with the generation prompt.
func (t *Template) RenderTools(system string, turns []Turn, tools []Tool) string {
	var sb strings.Builder
	for _, s := range t.RenderToolsSegments(system, turns, tools) {
		sb.WriteString(s.Text)
	}
	return sb.String()
}

// RenderToolsSegments renders system + turns + tool declarations into Special and literal-text segments.
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
			segs = append(segs, toolwire.RenderQwen35XML(t.has, system, toCoreTurns(turns), toCoreTools(tools), t.historyKind())...)
		} else {
			segs = append(segs, toolwire.RenderChatML(t.has, system, toCoreTurns(turns), toCoreTools(tools), t.historyKind(), t.groupsToolResults)...)
		}
		return append(segs, t.thinkSuffixSegs()...)
	case "mistral":
		return toolwire.RenderMistral(t.has, system, toCoreTurns(turns), toCoreTools(tools))
	case "llama3":
		return toolwire.RenderLlama3(timeNow(), t.has, system, toCoreTurns(turns), toCoreTools(tools))
	case "gemma4":
		if t.usesNativeTools() {
			return t.renderGemma4NativeTools(system, turns, tools, t.gemma4Think())
		}
		return t.renderGemma4Tools(system, turns, tools)
	case "harmony":
		return harmony.Segments(timeNow(), t.harmonyEffort(), system, toCoreTurns(turns), toCoreTools(tools))
	case "flint":
		return t.renderFlintTools(system, turns, tools)
	case "inkling2":
		return t.inkling2Segments(system, turns, tools)
	}
	return t.RenderSegments(system, turns)
}

func (t *Template) renderFlintTools(system string, turns []Turn, tools []Tool) []Segment {
	return toolwire.RenderFlint(t.has, system, toCoreTurns(turns), toCoreTools(tools))
}

// SupportsTools reports whether this family has a tool-calling template.
func (t *Template) SupportsTools() bool {
	switch t.name {
	case "chatml", "mellum2", "mistral", "llama3", "gemma4", "harmony", "flint", "inkling2":
		return true
	}
	return false
}

func (t *Template) harmonyEffort() string {
	if t.effort == "" {
		return "medium"
	}
	return t.effort
}

// ToolCallWrapper returns how a single JSON tool call is framed for constrained decoding.
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

// ParseToolCalls extracts structured tool calls and leading prose from the model's reply.
func (t *Template) ParseToolCalls(out string) ([]ToolCall, string) {
	switch t.name {
	case "chatml", "mellum2":
		calls, lead := toolwire.ParseChatML(out, nil)
		return fromCoreCalls(calls), lead
	case "mistral":
		calls, lead := toolwire.ParseMistral(out)
		return fromCoreCalls(calls), lead
	case "llama3":
		calls, lead := toolwire.ParseLlama3(out)
		return fromCoreCalls(calls), lead
	case "gemma4":
		return parseGemma4Tools(out)
	case "harmony":
		return parseHarmonyTools(out)
	case "flint":
		calls, lead := toolwire.ParseFlint(out, nil)
		return fromCoreCalls(calls), lead
	case "inkling2":
		return parseInklingTools(out)
	}
	return nil, out
}

// ParseToolCallsFor extracts tool calls using the declared tool schemas for parameter typing and bare/fenced recovery.
func (t *Template) ParseToolCallsFor(out string, tools []Tool) ([]ToolCall, string) {
	coreTools := toCoreTools(tools)
	var calls []ToolCall
	var lead string
	switch {
	case t != nil && (t.name == "chatml" || t.name == "mellum2"):
		cs, l := toolwire.ParseChatML(out, coreTools)
		calls, lead = fromCoreCalls(cs), l
	case t != nil && t.name == "flint":
		cs, l := toolwire.ParseFlint(out, coreTools)
		calls, lead = fromCoreCalls(cs), l
	default:
		calls, lead = t.ParseToolCalls(out)
	}
	if len(calls) > 0 || !t.AcceptsBareToolCall() {
		return calls, lead
	}
	if c, ok := chatcore.BareToolCall(out, coreTools); ok {
		return []ToolCall{fromCoreCall(c)}, ""
	}
	if t.lenientFenced {
		if c, fenceLead, ok := toolwire.FencedToolCall(out, coreTools); ok {
			return []ToolCall{fromCoreCall(c)}, fenceLead
		}
	}
	return calls, lead
}

// AcceptsBareToolCall reports whether ParseToolCallsFor also accepts an unwrapped JSON call.
func (t *Template) AcceptsBareToolCall() bool {
	switch t.name {
	case "chatml", "mellum2":
		return true
	}
	return false
}

// ToolCallOpener reports the literal bounding the prose lead and whether prose may be streamed incrementally.
func (t *Template) ToolCallOpener() (string, bool) {
	switch t.name {
	case "chatml", "mellum2", "flint":
		return "<tool_call>", true
	case "gemma4":
		return "<|tool_call>", true
	}
	return "", false
}

func parseHarmonyTools(out string) ([]ToolCall, string) {
	calls, lead := harmony.ParseTools(out)
	return fromCoreCalls(calls), lead
}

func funcDefJSON(t Tool) string {
	return chatcore.FuncDefJSON(chatcore.Tool{Name: t.Name, Description: t.Description, Parameters: t.Parameters})
}
