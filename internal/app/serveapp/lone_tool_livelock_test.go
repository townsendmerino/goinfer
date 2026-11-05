package serveapp

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// PRE-REGISTERED (docs/queue-correctness.md G39). With exactly ONE tool and tool_choice auto, forcedTool must
// not constrain the reply to that tool's call on EVERY turn: on the turn after the tool's result a client with
// a single tool could never get an answer (`serve check`'s `tools, OpenAI` row failed with "turn two asked for
// the tool again instead of answering — the agent-livelock shape").
//
// The rule: the lone-tool convenience is for the model's FIRST call. It is lifted when the conversation
// already ends in a tool result and tool_choice is auto (or absent). An explicit "required", a named function,
// and "none" mean what they always meant.

func TestForcedTool_loneToolConvenienceEndsAtTheToolResult(t *testing.T) {
	lone := []chat.Tool{{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)}}
	two := append(append([]chat.Tool(nil), lone...), chat.Tool{Name: "get_time", Parameters: json.RawMessage(`{"type":"object"}`)})
	choice := func(s string) json.RawMessage {
		if s == "" {
			return nil
		}
		return json.RawMessage(s)
	}
	cases := []struct {
		name        string
		toolChoice  string
		tools       []chat.Tool
		afterResult bool
		want        string // "" = not forced
	}{
		{"auto, one tool, first turn: the convenience stays", `"auto"`, lone, false, "get_weather"},
		{"absent tool_choice, one tool, first turn: the convenience stays", ``, lone, false, "get_weather"},
		{"auto, one tool, AFTER a tool result: the model is free to answer", `"auto"`, lone, true, ""},
		{"absent tool_choice, one tool, AFTER a tool result: free to answer", ``, lone, true, ""},
		{"required, one tool, after a tool result: the client asked for a call", `"required"`, lone, true, "get_weather"},
		{"a named function, after a tool result: the client asked for it", `{"type":"function","function":{"name":"get_weather"}}`, lone, true, "get_weather"},
		{"none is never forced", `"none"`, lone, false, ""},
		{"auto, two tools: never forced either way", `"auto"`, two, false, ""},
		{"auto, two tools, after a result: never forced", `"auto"`, two, true, ""},
	}
	for _, c := range cases {
		got := forcedTool(choice(c.toolChoice), c.tools, c.afterResult)
		name := ""
		if got != nil {
			name = got.Name
		}
		if name != c.want {
			t.Errorf("%s: forced %q, want %q", c.name, name, c.want)
		}
	}
}

func TestEndsWithToolResult(t *testing.T) {
	for _, c := range []struct {
		name  string
		turns []chat.Turn
		want  bool
	}{
		{"empty", nil, false},
		{"a user question", []chat.Turn{{Role: "user", Content: "hi"}}, false},
		{"call then result", []chat.Turn{{Role: "user"}, {Role: "assistant", ToolCalls: []chat.ToolCall{{Name: "f"}}}, {Role: "tool", Content: "r"}}, true},
		{"result then a new user question", []chat.Turn{{Role: "tool", Content: "r"}, {Role: "user", Content: "and now?"}}, false},
		{"an Anthropic user turn that carries tool results", []chat.Turn{{Role: "user", ToolLoop: true}}, true},
	} {
		if got := endsWithToolResult(c.turns); got != c.want {
			t.Errorf("%s: endsWithToolResult = %v, want %v", c.name, got, c.want)
		}
	}
}
