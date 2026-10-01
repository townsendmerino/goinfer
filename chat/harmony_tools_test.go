package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type harmonyToolGolden struct {
	Cases []struct {
		Name     string `json:"name"`
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			Thinking  string `json:"thinking"`
			Name      string `json:"name"`
			CallID    string `json:"tool_call_id"`
			ToolCalls []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
		Tools []struct {
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		Prompt string `json:"prompt"`
	} `json:"cases"`
}

func loadHarmonyToolGolden(t *testing.T) harmonyToolGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "harmony_tools.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g harmonyToolGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestHarmonyTools_matchHF pins tool rendering to gpt-oss's own chat template, byte for byte: the `namespace functions` block (over
// schemas that reach each branch of the template's TypeScript-type macro, including the whitespace Jinja leaks into it), the system
// block's extra line, how a call and its result replay, and when the analysis beside a call is kept or dropped. Hand-written
// expectations would only test that the code does what was typed; these are what HuggingFace rendered from the template in the GGUF.
func TestHarmonyTools_matchHF(t *testing.T) {
	defer func() { timeNow = time.Now }()
	timeNow = func() time.Time { return time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC) }
	g := loadHarmonyToolGolden(t)
	if len(g.Cases) < 16 {
		t.Fatalf("only %d golden cases", len(g.Cases))
	}
	for _, c := range g.Cases {
		system, turns := "", []Turn(nil)
		for _, m := range c.Messages {
			switch {
			case m.Role == "system":
				system = m.Content
			case m.Role == "assistant":
				tu := Turn{Role: "assistant", Content: m.Content, Reasoning: m.Thinking}
				for _, tc := range m.ToolCalls {
					tu.ToolCalls = append(tu.ToolCalls, ToolCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments})
				}
				turns = append(turns, tu)
			case m.Role == "tool":
				turns = append(turns, Turn{Role: "tool", Content: m.Content, ToolName: m.Name, ToolCallID: m.CallID})
			default:
				turns = append(turns, Turn{Role: "user", Content: m.Content})
			}
		}
		var tools []Tool
		for _, tl := range c.Tools {
			tools = append(tools, Tool{Name: tl.Function.Name, Description: tl.Function.Description, Parameters: tl.Function.Parameters})
		}
		got := strings.Replace(Harmony().RenderTools(system, turns, tools), "Current date: 2026-08-16", "Current date: {DATE}", 1)
		if got != c.Prompt {
			i := 0
			for i < len(got) && i < len(c.Prompt) && got[i] == c.Prompt[i] {
				i++
			}
			lo := max(0, i-60)
			t.Errorf("%s: first difference at byte %d\n got  %q\n want %q", c.Name, i, got[lo:min(len(got), i+120)], c.Prompt[lo:min(len(c.Prompt), i+120)])
		}
	}
}
