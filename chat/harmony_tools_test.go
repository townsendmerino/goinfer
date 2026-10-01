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

// A gpt-oss call is a message addressed `to=functions.NAME`; parseHarmonyTools reads a RAW reply (what -reasoning-format none leaves),
// ReplySplitter.ToolCalls reads it from the splitter every other route holds. They must agree.
func TestHarmonyToolCalls(t *testing.T) {
	const call = `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris",  "units":"c"}`
	reply := "<|channel|>analysis<|message|>need the weather<|end|><|start|>assistant" + call

	calls, lead := parseHarmonyTools(reply)
	if len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Paris","units":"c"}` {
		t.Fatalf("parseHarmonyTools: %+v", calls)
	}
	if lead != "" {
		t.Errorf("a reply that only reasons and calls has no prose, got %q", lead)
	}

	// The same reply through the ReplySplitter (the route every decode loop uses): reasoning and prose separated, call kept.
	rs := Harmony().NewReplySplitter(nil)
	var reasoning, content strings.Builder
	for i := 0; i < len(reply); i += 5 {
		r, c := rs.Push(reply[i:min(i+5, len(reply))])
		reasoning.WriteString(r)
		content.WriteString(c)
	}
	r, c, _ := rs.Finish()
	reasoning.WriteString(r)
	content.WriteString(c)
	got := rs.ToolCalls()
	if len(got) != 1 || got[0].Name != "get_weather" || string(got[0].Arguments) != string(calls[0].Arguments) {
		t.Errorf("ReplySplitter.ToolCalls = %+v, want what parseHarmonyTools read", got)
	}
	if reasoning.String() != "need the weather" || content.String() != "" {
		t.Errorf("a call must appear in neither stream: reasoning=%q content=%q", reasoning.String(), content.String())
	}

	// A preamble before the call is prose: the lead of the call.
	_, lead = parseHarmonyTools("<|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant" + call)
	if lead != "Let me check." {
		t.Errorf("preamble lead = %q", lead)
	}

	// No call: the reply is returned untouched, as every family's parser returns a prose answer.
	plain := "<|channel|>analysis<|message|>t<|end|><|start|>assistant<|channel|>final<|message|>Hello"
	if cs, out := parseHarmonyTools(plain); len(cs) != 0 || out != plain {
		t.Errorf("a prose reply must come back as it is: %v %q", cs, out)
	}
	if got := (*ReplySplitter)(nil).ToolCalls(); got != nil {
		t.Error("nil ReplySplitter has no calls")
	}
	if got := ChatML().NewReplySplitter(nil); got != nil {
		t.Error("setup: ChatML has no splitter")
	}
}

// What is not a call a client can run is left out rather than guessed at.
func TestHarmonyToolCalls_leavesOutWhatIsNotACall(t *testing.T) {
	msg := func(header, body string) string {
		return "<|channel|>" + header + "<|message|>" + body + "<|end|><|start|>assistant"
	}
	cases := map[string]string{
		"a built-in tool the client never declared": msg("commentary to=browser.search <|constrain|>json", `{"query":"x"}`),
		"python":                           msg("analysis to=python code", `print(1)`),
		"arguments that are not an object": msg("commentary to=functions.f <|constrain|>json", `[1,2]`),
		"arguments that are not JSON":      msg("commentary to=functions.f <|constrain|>json", `{"city": `),
		"no function name":                 msg("commentary to=functions. <|constrain|>json", `{}`),
	}
	for name, in := range cases {
		if calls, _ := parseHarmonyTools(in + "<|channel|>final<|message|>x"); len(calls) != 0 {
			t.Errorf("%s: got calls %+v", name, calls)
		}
	}
	// Empty arguments are an empty object (a function with no parameters).
	calls, _ := parseHarmonyTools("<|channel|>commentary to=functions.ping <|constrain|>json<|message|>")
	if len(calls) != 1 || calls[0].Name != "ping" || string(calls[0].Arguments) != "{}" {
		t.Errorf("a no-argument call: %+v", calls)
	}
	// Two calls in one reply (the model normally stops at the first <|call|>, but a raw reply may carry more).
	two := msg("commentary to=functions.a <|constrain|>json", `{"n":1}`) + "<|channel|>commentary to=functions.b <|constrain|>json<|message|>{\"n\":2}"
	if calls, _ := parseHarmonyTools(two); len(calls) != 2 || calls[0].Name != "a" || calls[1].Name != "b" {
		t.Errorf("two calls: %+v", calls)
	}
}

// The template's own parser side: gpt-oss supports tools, has no JSON-constrainable call form, and is buffered (no safe opener to
// stream prose against) — the same shape Gemma 4's bespoke form has, minus the opener.
func TestHarmony_toolCapabilities(t *testing.T) {
	h := Harmony()
	if !h.SupportsTools() {
		t.Error("harmony must support tools now")
	}
	if _, _, _, _, ok := h.ToolCallWrapper(); ok {
		t.Error("harmony has no JSON call wrapper to constrain a decode to")
	}
	if _, ok := h.ToolCallOpener(); ok {
		t.Error("harmony's call is a message header, not a prefix a prose streamer can hold against")
	}
	if h.AcceptsBareToolCall() {
		t.Error("harmony never accepts a bare JSON call")
	}
}
