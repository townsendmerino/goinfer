package chat

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type histCase struct {
	Name     string            `json:"name"`
	Tools    bool              `json:"tools"`
	Messages []json.RawMessage `json:"messages"`
	Prompts  map[string]string `json:"prompts"`
}
type histGolden struct {
	Checkpoint   string     `json:"checkpoint"`
	ChatTemplate string     `json:"chat_template"`
	Cases        []histCase `json:"cases"`
}

func loadHistGoldens(t *testing.T) []histGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []histGolden
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	if len(gs) != 4 {
		t.Fatalf("expected the four pinned checkpoints, got %d", len(gs))
	}
	return gs
}

// goldenTurns converts an HF-shaped message list into Turns the way serve does: reasoning from reasoning_content or reasoning,
// tool calls as ToolCalls, a tool message carrying its function name.
func goldenTurns(t *testing.T, msgs []json.RawMessage) []Turn {
	t.Helper()
	var turns []Turn
	for _, raw := range msgs {
		var m struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
			Name             string `json:"name"`
			ToolCalls        []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		tu := Turn{Role: m.Role, Content: m.Content, ToolName: m.Name}
		tu.Reasoning = m.ReasoningContent
		if tu.Reasoning == "" {
			tu.Reasoning = m.Reasoning
		}
		for _, c := range m.ToolCalls {
			tu.ToolCalls = append(tu.ToolCalls, ToolCall{Name: c.Function.Name, Arguments: c.Function.Arguments})
		}
		turns = append(turns, tu)
	}
	return turns
}

// afterSystemTurn drops the system turn (the tools preamble: its JSON key order and spacing are goinfer's own, checked
// structurally elsewhere — TestRenderTools_declarations) and leaves the conversation.
func afterSystemTurn(s string) string {
	if i := strings.Index(s, "<|im_end|>\n"); i >= 0 {
		return s[i+len("<|im_end|>\n"):]
	}
	return s
}

// compactToolCalls rewrites the JSON inside every <tool_call>\n…\n</tool_call> compactly, so a Hermes-JSON call compares
// structurally (goinfer and Jinja's tojson differ in spacing) while everything around it compares byte for byte.
func compactToolCalls(s string) string {
	const open, closing = "<tool_call>\n", "\n</tool_call>"
	var out strings.Builder
	for {
		i := strings.Index(s, open)
		if i < 0 {
			out.WriteString(s)
			return out.String()
		}
		j := strings.Index(s[i:], closing)
		if j < 0 {
			out.WriteString(s)
			return out.String()
		}
		out.WriteString(s[:i+len(open)])
		body := s[i+len(open) : i+j]
		var buf bytes.Buffer
		if json.Compact(&buf, []byte(body)) == nil {
			out.WriteString(buf.String())
		} else {
			out.WriteString(body)
		}
		s = s[i+j:]
	}
}

// assistantThroughFirstCall returns the first assistant turn up to its first <tool_call>, trailing newlines trimmed: the part
// of a Qwen3.5 tool-loop prompt that is comparable with goinfer's (its call syntax is XML there, Hermes JSON here, and the
// newlines before the call differ with it — a separate gap).
func assistantThroughFirstCall(s string) string {
	i := strings.Index(s, "<|im_start|>assistant\n")
	if i < 0 {
		return ""
	}
	j := strings.Index(s[i:], "<tool_call>")
	if j < 0 {
		return s[i:]
	}
	return strings.TrimRight(s[i:i+j], "\n")
}

var histMode = map[string]ThinkMode{"unset": ThinkTemplate, "false": ThinkOff, "true": ThinkOn}

// TestThinkHistory_matchHF pins the HISTORY rule per checkpoint: every conversation the generator made, in every thinking mode
// (the rule does not depend on it), renders byte for byte what HuggingFace renders from the real template — assistant turns
// before the last user query stripped of reasoning, those after it carrying it, tags in content extracted the way the template
// does. Qwen3.5 tool loops are compared through the first call only; Gemma 4's consecutive-assistant-turn and text-with-call
// case is not modelled by goinfer's Gemma tool renderer (pre-existing, unrelated to reasoning) and is skipped, named.
func TestThinkHistory_matchHF(t *testing.T) {
	ran, skipped := 0, 0
	for _, g := range loadHistGoldens(t) {
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Fatal(err)
		}
		if tmpl.Reasoning() == nil || tmpl.Reasoning().hist == histNone {
			t.Fatalf("%s: the history rule was not recognised from its real template", g.Checkpoint)
		}
		for _, c := range g.Cases {
			if strings.HasPrefix(g.Checkpoint, "gemma-4") && c.Name == "tool_loop_text" {
				// tool_loop_text: Gemma's template writes an assistant message's TEXT after its tool responses (and then closes the turn),
				// goinfer's tool renderer writes it before the calls and leaves the turn open. Pre-existing and unrelated to reasoning;
				// whether to follow the template here is its own measured decision (docs/tasks/task-qwen35-think-prompt-2026-09.md).
				// (two_after*, consecutive assistant messages sharing one model turn, is compared like every other case.)
				skipped++
				continue
			}
			turns := goldenTurns(t, c.Messages)
			for mode, want := range c.Prompts {
				tm := tmpl.WithThinking(histMode[mode])
				var got string
				if c.Tools {
					got = tm.RenderTools("", turns, goldenTools)
				} else {
					got = tm.Render("", turns)
				}
				switch {
				case c.Tools && strings.HasPrefix(g.Checkpoint, "qwen3.5"):
					got, want = assistantThroughFirstCall(got), assistantThroughFirstCall(want)
				case c.Tools && strings.HasPrefix(g.Checkpoint, "qwen3"):
					got, want = compactToolCalls(afterSystemTurn(got)), compactToolCalls(afterSystemTurn(want))
				}
				ran++
				if got != want {
					t.Errorf("%s / %s / enable_thinking=%s:\n got %q\nwant %q", g.Checkpoint, c.Name, mode, got, want)
				}
			}
		}
	}
	t.Logf("%d prompts compared against HF (%d cases skipped: Gemma tool_loop_text)", ran, skipped)
	if ran < 100 {
		t.Fatalf("only %d prompts compared; the gate is not covering the goldens", ran)
	}
}

// ThinkAsIs is the pre-thinking bytes: replayed reasoning must change nothing under it, for every case, text and tool paths.
func TestThinkHistory_asIsIgnoresReasoning(t *testing.T) {
	for _, g := range loadHistGoldens(t) {
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for _, c := range g.Cases {
			turns := goldenTurns(t, c.Messages)
			bare := make([]Turn, len(turns))
			copy(bare, turns)
			for i := range bare {
				bare[i].Reasoning = ""
			}
			asis := tmpl.WithThinking(ThinkAsIs)
			var with, without string
			if c.Tools {
				with, without = asis.RenderTools("", turns, goldenTools), asis.RenderTools("", bare, goldenTools)
			} else {
				with, without = asis.Render("", turns), asis.Render("", bare)
			}
			if with != without {
				t.Errorf("%s / %s: -thinking asis changed with replayed reasoning:\n with    %q\n without %q", g.Checkpoint, c.Name, with, without)
			}
		}
	}
}

// The ToolLoop marker: a user turn that only accompanies tool results (an Anthropic client's reminder text in the same message
// as a tool_result) is not a new query, so the turns before it keep their reasoning; without the marker they would lose it on
// the very turn after the first tool result — exactly the turns the rule exists to keep.
func TestThinkHistory_toolLoopUserTurnIsNotAQuery(t *testing.T) {
	var tmpl *Template
	for _, g := range loadHistGoldens(t) {
		if g.Checkpoint == "qwen3-4b" {
			tmpl, _ = Detect(Meta{ChatTemplate: g.ChatTemplate})
		}
	}
	turns := []Turn{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", Content: "", Reasoning: "need the weather", ToolCalls: []ToolCall{{Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)}}},
		{Role: "tool", Content: "18C", ToolName: "get_weather"},
		{Role: "user", Content: "<system-reminder>stay brief</system-reminder>"},
	}
	plain := tmpl.WithThinking(ThinkTemplate).RenderTools("", turns, goldenTools)
	if strings.Contains(plain, "need the weather") {
		t.Fatal("setup: without the marker the trailing user turn is a query, so the earlier reasoning must be stripped")
	}
	turns[3].ToolLoop = true
	marked := tmpl.WithThinking(ThinkTemplate).RenderTools("", turns, goldenTools)
	if !strings.Contains(marked, "<think>\nneed the weather\n</think>") {
		t.Fatalf("with the marker the reasoning must be kept, got %q", marked)
	}
	if !strings.Contains(marked, "<|im_start|>user\n<system-reminder>stay brief</system-reminder>") {
		t.Fatalf("the reminder text itself must still be rendered as a user turn, got %q", marked)
	}
}

func TestSplitTagged(t *testing.T) {
	cases := []struct {
		in, wantR, wantC string
		ok               bool
	}{
		{"<think>\nplan\n</think>\n\nHello", "plan", "Hello", true},
		{"no tags here", "", "no tags here", false},
		{"plan only\n</think>\n\nHello", "plan only", "Hello", true},            // template opened the block in the prompt
		{"a<think>b</think>\nc</think>\n\nd", "b", "d", true},                   // first </think> ends the reasoning, the LAST one starts the content
		{"<think>\n\nspaced\n\n</think>\n\n\nAnswer", "spaced", "Answer", true}, // newlines around the reasoning are stripped
		{"<think>x</think>", "x", "", true},
	}
	for _, tc := range cases {
		r, c, ok := splitTagged(tc.in)
		if r != tc.wantR || c != tc.wantC || ok != tc.ok {
			t.Errorf("splitTagged(%q) = (%q, %q, %v), want (%q, %q, %v)", tc.in, r, c, ok, tc.wantR, tc.wantC, tc.ok)
		}
	}
}

func TestStripChannels(t *testing.T) {
	cases := map[string]string{
		"plain":                                  "plain",
		"<|channel>thought\nX\n<channel|>Final":  "Final",
		"  a <|channel>thought\nX<channel|> b  ": "a  b",
		"<|channel>thought\nX<channel|>One<|channel>thought\nY<channel|>Two": "OneTwo",
	}
	for in, want := range cases {
		if got := stripChannels(in); got != want {
			t.Errorf("stripChannels(%q) = %q, want %q", in, got, want)
		}
	}
}
