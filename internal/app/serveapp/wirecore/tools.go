package wirecore

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

// ToolsConstrainedFromStart reports whether tool constraints will govern output from token 1 (requiring thinking-off prompt rendering).
func ToolsConstrainedFromStart(forced *chat.Tool, namedForce bool, union string, tools []chat.Tool, unionEnabled bool) bool {
	if forced != nil {
		return true
	}
	return !namedForce && len(tools) >= 1 && union == "required" && unionEnabled
}

// ForcedCallEnforcement reports how a must-call tool_choice is enforced on tm ("grammar", "post-check", "off", or "").
func ForcedCallEnforcement(tm *chat.Template, unionEnabled bool) string {
	switch {
	case tm == nil || !tm.SupportsTools():
		return ""
	case !unionEnabled:
		return "off"
	}
	if _, _, _, _, ok := tm.ToolCallWrapper(); ok {
		return "grammar"
	}
	return "post-check"
}

var toolUnionOffSeen sync.Map

// WarnToolUnionOff logs once per distinct error why a request's multi-tool union could not be compiled.
func WarnToolUnionOff(err error) {
	if _, seen := toolUnionOffSeen.LoadOrStore(err.Error(), true); !seen {
		fmt.Fprintf(os.Stderr, "goinfer: tool calls are not constrained for these tools (each distinct reason is reported once): %v\n", err)
	}
}

// BuildToolMasker constructs the token-1 Masker, lazy auto-union LazyMasker, or post-check flag for a tool-enabled request.
func BuildToolMasker(tm *chat.Template, forced *chat.Tool, namedForce bool, union string, tools []chat.Tool, unionEnabled, unionAuto bool, tokenBytes func() [][]byte, eos []int) (masker *constrain.Masker, lazy *constrain.LazyMasker, postCheck bool, err error) {
	if forced == nil && !namedForce && len(tools) >= 1 {
		if !unionEnabled || (union != "auto" && union != "required") || (union == "auto" && !unionAuto) {
			return nil, nil, false, nil
		}
		prefix, suffix, argsKey, array, ok := tm.ToolCallWrapper()
		if !ok {
			return nil, nil, union == "required" && ForcedCallEnforcement(tm, unionEnabled) == "post-check", nil
		}
		if union == "required" {
			specs := make([]constrain.ToolSpec, len(tools))
			for i, t := range tools {
				specs[i] = constrain.ToolSpec{Name: t.Name, Parameters: t.Parameters}
			}
			g, gerr := constrain.ToolCallsGrammar(prefix, suffix, argsKey, array, specs)
			if gerr != nil {
				WarnToolUnionOff(gerr)
				return nil, nil, false, nil
			}
			return constrain.NewMasker(g, tokenBytes(), eos).StopWhenComplete(), nil, false, nil
		}
		lm, lerr := tm.ToolGrammar(tools, tokenBytes(), eos)
		if lerr != nil {
			WarnToolUnionOff(lerr)
			return nil, nil, false, nil
		}
		return nil, lm, false, nil
	}
	if forced == nil {
		if namedForce {
			names := make([]string, 0, len(tools))
			for _, t := range tools {
				names = append(names, t.Name)
			}
			avail := "none were supplied"
			if len(names) > 0 {
				avail = "available: " + strings.Join(names, ", ")
			}
			return nil, nil, false, fmt.Errorf("tool_choice names a function that is not in tools (%s)", avail)
		}
		return nil, nil, false, nil
	}
	prefix, suffix, argsKey, array, ok := tm.ToolCallWrapper()
	if !ok {
		if namedForce {
			return nil, nil, false, fmt.Errorf("tool_choice forces function %q but this model's chat template has no constrainable tool-call form", forced.Name)
		}
		return nil, nil, union == "required" && ForcedCallEnforcement(tm, unionEnabled) == "post-check", nil
	}
	g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced.Parameters)
	if gerr != nil {
		if namedForce {
			return nil, nil, false, fmt.Errorf("tool_choice forces function %q but its schema cannot be constrained: %w", forced.Name, gerr)
		}
		return nil, nil, false, nil
	}
	return constrain.NewMasker(g, tokenBytes(), eos).StopWhenComplete(), nil, false, nil
}

// ToAPICalls formats parsed tool calls in OpenAI's response shape, minting call_ IDs when empty.
func ToAPICalls(calls []chat.ToolCall) []map[string]any {
	out := make([]map[string]any, len(calls))
	for i, c := range calls {
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		id := c.ID
		if id == "" {
			id = "call_" + stores.MintID()
		}
		out[i] = map[string]any{
			"id": id, "type": "function",
			"function": map[string]any{"name": c.Name, "arguments": args},
		}
	}
	return out
}

// StreamToolCalls formats parsed tool calls for an OpenAI SSE delta chunk (adding per-call index).
func StreamToolCalls(calls []chat.ToolCall) []map[string]any {
	out := ToAPICalls(calls)
	for i := range out {
		out[i]["index"] = i
	}
	return out
}

// ToolChoiceMode parses OpenAI's tool_choice field into "auto", "none", "required", or "function".
func ToolChoiceMode(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "auto"
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return "function"
}

// OpenAIUnionMode maps an OpenAI tool_choice to BuildToolMasker's union mode ("auto", "required", or "").
func OpenAIUnionMode(toolChoice json.RawMessage) string {
	switch ToolChoiceMode(toolChoice) {
	case "auto":
		return "auto"
	case "required":
		return "required"
	}
	return ""
}

// ForcedTool returns the single tool forced by toolChoice (or the lone declared tool prior to a tool result).
func ForcedTool(toolChoice json.RawMessage, tools []chat.Tool, afterToolResult bool) *chat.Tool {
	switch ToolChoiceMode(toolChoice) {
	case "none":
		return nil
	case "function":
		var c struct {
			Function struct{ Name string } `json:"function"`
			Name     string                `json:"name"`
		}
		if json.Unmarshal(toolChoice, &c) == nil {
			if c.Function.Name == "" {
				c.Function.Name = c.Name
			}
			for i := range tools {
				if tools[i].Name == c.Function.Name {
					return &tools[i]
				}
			}
		}
		return nil
	}
	if len(tools) == 1 && !(afterToolResult && ToolChoiceMode(toolChoice) == "auto") {
		return &tools[0]
	}
	return nil
}

// EndsWithToolResult reports whether the last turn in turns is a tool result or mid-loop continuation.
func EndsWithToolResult(turns []chat.Turn) bool {
	if len(turns) == 0 {
		return false
	}
	last := turns[len(turns)-1]
	return last.Role == "tool" || last.ToolLoop
}
