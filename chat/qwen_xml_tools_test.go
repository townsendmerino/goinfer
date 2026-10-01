package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type qwenToolGolden struct {
	ChatTemplate string `json:"chat_template"`
	Cases        []struct {
		Name     string `json:"name"`
		Mode     string `json:"mode"`
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			Reasoning string `json:"reasoning_content"`
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

func loadQwenToolGolden(t *testing.T) qwenToolGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "qwen35_tools.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g qwenToolGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestQwen35NativeTools_matchHF pins the NATIVE tool rendering to the Qwen3.5 chat template, byte for byte, over conversations that reach
// each branch of its tool handling: the declarations and the format instructions, the system text after them, a call's parameters (a
// string raw, a number as Python's str, True / False / None, JSON for a list or a mapping), the newline rule around a call and between
// calls, results grouped into one user turn, and the think block after the last query — in every thinking mode. Expectations are what
// HuggingFace rendered from the template, not what the code was typed to do.
func TestQwen35NativeTools_matchHF(t *testing.T) {
	g := loadQwenToolGolden(t)
	tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
	if err != nil {
		t.Fatal(err)
	}
	if !tmpl.DeclaresNativeTools() {
		t.Fatal("the Qwen3.5 template must be recognised as declaring the native tool form")
	}
	if len(g.Cases) < 20 {
		t.Fatalf("only %d golden cases", len(g.Cases))
	}
	mode := map[string]ThinkMode{"unset": ThinkTemplate, "false": ThinkOff, "true": ThinkOn}
	for _, c := range g.Cases {
		system, turns := "", []Turn(nil)
		for _, m := range c.Messages {
			switch m.Role {
			case "system":
				system = m.Content
			case "assistant":
				tu := Turn{Role: "assistant", Content: m.Content, Reasoning: m.Reasoning}
				for _, tc := range m.ToolCalls {
					tu.ToolCalls = append(tu.ToolCalls, ToolCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments})
				}
				turns = append(turns, tu)
			case "tool":
				turns = append(turns, Turn{Role: "tool", Content: m.Content})
			default:
				turns = append(turns, Turn{Role: "user", Content: m.Content})
			}
		}
		var tools []Tool
		for _, tl := range c.Tools {
			tools = append(tools, Tool{Name: tl.Function.Name, Description: tl.Function.Description, Parameters: tl.Function.Parameters})
		}
		tm := tmpl.WithToolFormat(ToolFormatTemplate).WithThinking(mode[c.Mode])
		got := tm.RenderTools(system, turns, tools)
		if got != c.Prompt {
			i := 0
			for i < len(got) && i < len(c.Prompt) && got[i] == c.Prompt[i] {
				i++
			}
			lo := max(0, i-70)
			t.Errorf("%s (enable_thinking=%s): first difference at byte %d\n got  %q\n want %q", c.Name, c.Mode, i, got[lo:min(len(got), i+110)], c.Prompt[lo:min(len(c.Prompt), i+110)])
		}
	}
}

// -tool-format hermes (the default) must stay exactly what it was: the same template, switched off, renders goinfer's own prompt; and a
// template with no native form is untouched by the switch.
func TestToolFormat_hermesIsUnchangedAndSwitchable(t *testing.T) {
	g := loadQwenToolGolden(t)
	tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
	tools := []Tool{{Name: "get_weather", Description: "Weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}
	turns := []Turn{{Role: "user", Content: "Weather?"}}

	hermes := tmpl.RenderTools("", turns, tools) // the default format
	if !strings.Contains(hermes, "return a json object with function name and arguments within <tool_call></tool_call> XML tags") {
		t.Fatalf("the default must stay goinfer's Hermes prompt:\n%s", hermes)
	}
	if got := tmpl.WithToolFormat(ToolFormatHermes).RenderTools("", turns, tools); got != hermes {
		t.Error("an explicit hermes format must render exactly the default")
	}
	native := tmpl.WithToolFormat(ToolFormatTemplate).RenderTools("", turns, tools)
	if native == hermes || !strings.Contains(native, "<function=example_function_name>") {
		t.Errorf("the template format must render the model's own prompt:\n%s", native)
	}
	// Constraining a decode needs a JSON wrapper; the native form has none.
	if _, _, _, _, ok := tmpl.ToolCallWrapper(); !ok {
		t.Error("hermes keeps its JSON call wrapper (so forced tool_choice stays constrained)")
	}
	if _, _, _, _, ok := tmpl.WithToolFormat(ToolFormatTemplate).ToolCallWrapper(); ok {
		t.Error("the native form is XML: it must not offer a JSON wrapper")
	}
	// A family with no native form is untouched by the switch.
	for _, other := range []*Template{ChatML(), Llama3(), Harmony(), Gemma4(), nil} {
		if other.WithToolFormat(ToolFormatTemplate) != other {
			t.Errorf("%v: WithToolFormat must be a no-op without a native form", nameOf(other))
		}
		if other.DeclaresNativeTools() {
			t.Errorf("%v declares no native tool form", nameOf(other))
		}
	}
	if f, ok := ParseToolFormat("TEMPLATE"); !ok || f != ToolFormatTemplate {
		t.Error("ParseToolFormat(TEMPLATE)")
	}
	if f, ok := ParseToolFormat(""); !ok || f != ToolFormatHermes {
		t.Error("the empty value is the default, hermes")
	}
	if _, ok := ParseToolFormat("xml"); ok {
		t.Error("an unknown format must be refused")
	}
	// The Qwen3 (Hermes-native) template has no XML form: nothing to switch to.
	q3 := loadQwen3Template(t)
	if q3.DeclaresNativeTools() {
		t.Error("Qwen3's own template is the Hermes JSON form: it declares no native XML tools")
	}
}

func loadQwen3Template(t *testing.T) *Template {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "think_modes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []struct {
		Checkpoint   string `json:"checkpoint"`
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	for _, g := range gs {
		if g.Checkpoint == "qwen3-4b" {
			tm, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
			if err != nil {
				t.Fatal(err)
			}
			return tm
		}
	}
	t.Fatal("no qwen3-4b template in the think_modes golden")
	return nil
}

func jsonEqual(t *testing.T, a, b string) bool {
	t.Helper()
	var x, y any
	if json.Unmarshal([]byte(a), &x) != nil || json.Unmarshal([]byte(b), &y) != nil {
		return false
	}
	ax, _ := json.Marshal(x)
	by, _ := json.Marshal(y)
	return string(ax) == string(by)
}

var searchTool = Tool{Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object","properties":{
	"query":{"type":"string"},"max_results":{"type":"integer"},"ratio":{"type":"number"},"safe":{"type":"boolean"},
	"tags":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"},"numstr":{"type":"string"},"nothing":{"type":["string","null"]}}}`)}

// An XML call is read into the same ToolCall a JSON call would be, with each parameter typed by the schema: the text is all the
// model wrote, so only the schema can say that "123" is a string here and a number there.
func TestParseXMLCalls_typedBySchema(t *testing.T) {
	out := "I'll look that up.\n\n<tool_call>\n<function=search>\n<parameter=query>\ngo 1.24 release\n</parameter>\n<parameter=max_results>\n3\n</parameter>\n" +
		"<parameter=ratio>\n1.5\n</parameter>\n<parameter=safe>\nTrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b c\"]\n</parameter>\n" +
		"<parameter=opts>\n{\"k\": [1, 2]}\n</parameter>\n<parameter=numstr>\n123\n</parameter>\n<parameter=nothing>\nNone\n</parameter>\n</function>\n</tool_call>"
	tm := &Template{name: "chatml"}
	calls, lead := tm.ParseToolCallsFor(out, []Tool{searchTool})
	if len(calls) != 1 || calls[0].Name != "search" || lead != "I'll look that up." {
		t.Fatalf("calls=%+v lead=%q", calls, lead)
	}
	want := `{"query":"go 1.24 release","max_results":3,"ratio":1.5,"safe":true,"tags":["a","b c"],"opts":{"k":[1,2]},"numstr":"123","nothing":null}`
	if !jsonEqual(t, string(calls[0].Arguments), want) {
		t.Errorf("arguments:\n got  %s\n want %s", calls[0].Arguments, want)
	}

	// Without the tool list nothing says "123" is a string, so unambiguous JSON is decoded and everything else stays text.
	calls, _ = tm.ParseToolCalls(out)
	if len(calls) != 1 {
		t.Fatal("no call without a tool list")
	}
	var args map[string]json.RawMessage
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("arguments are not a JSON object: %s", calls[0].Arguments)
	}
	if string(args["query"]) != `"go 1.24 release"` || string(args["max_results"]) != "3" || string(args["safe"]) != "true" || string(args["nothing"]) != "null" {
		t.Errorf("untyped decode: %s", calls[0].Arguments)
	}

	// A value that does not fit its declared type is kept as text rather than lost or invented.
	bad := "<tool_call>\n<function=search>\n<parameter=max_results>\nthree\n</parameter>\n<parameter=safe>\nmaybe\n</parameter>\n</function>\n</tool_call>"
	calls, _ = tm.ParseToolCallsFor(bad, []Tool{searchTool})
	if len(calls) != 1 || !jsonEqual(t, string(calls[0].Arguments), `{"max_results":"three","safe":"maybe"}`) {
		t.Errorf("misfit values: %+v", calls)
	}
}

// The Hermes JSON form still parses as it did; the two forms can share one reply; and what is not a complete call is not a call.
func TestParseXMLCalls_formsAndFailures(t *testing.T) {
	tm := &Template{name: "chatml"}
	tools := []Tool{searchTool}
	jsonCall := "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	xmlCall := "<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"

	calls, _ := tm.ParseToolCallsFor(jsonCall, tools)
	if len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city": "Paris"}` {
		t.Errorf("the JSON form regressed: %+v", calls)
	}
	calls, _ = tm.ParseToolCallsFor(jsonCall+"\n"+xmlCall, tools)
	if len(calls) != 2 || calls[0].Name != "get_weather" || calls[1].Name != "search" {
		t.Errorf("both forms in one reply: %+v", calls)
	}
	calls, _ = tm.ParseToolCallsFor(xmlCall+"\n"+xmlCall, tools)
	if len(calls) != 2 {
		t.Errorf("two XML calls: %+v", calls)
	}
	// No parameters is an empty object.
	if calls, _ = tm.ParseToolCallsFor("<tool_call>\n<function=ping>\n</function>\n</tool_call>", nil); len(calls) != 1 || string(calls[0].Arguments) != "{}" {
		t.Errorf("a call with no parameters: %+v", calls)
	}

	for name, in := range map[string]string{
		"cut off inside the call":      "<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n",
		"cut off inside a parameter":   "<tool_call>\n<function=search>\n<parameter=query>\nx",
		"no function name":             "<tool_call>\n<function=>\n</function>\n</tool_call>",
		"a function with no closing >": "<tool_call>\n<function=search",
		"prose that mentions the tag":  "Use <function=search> to search.",
		"an empty block":               "<tool_call>\n</tool_call>",
	} {
		if calls, _ := tm.ParseToolCallsFor(in, tools); len(calls) != 0 {
			t.Errorf("%s: parsed %+v, want no call", name, calls)
		}
	}
}

// The renderer's output must parse back to the arguments it was given: a call replayed into a prompt and the same call read from a reply
// are one format, and a type that does not survive the trip (a bool as True, null as None, a string that looks like a number) is a defect.
func TestQwen35XML_roundTrip(t *testing.T) {
	g := loadQwenToolGolden(t)
	tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
	native := tmpl.WithToolFormat(ToolFormatTemplate)
	args := `{"query": "go 1.24", "max_results": 3, "ratio": 1.5, "safe": true, "tags": ["a", "b c"], "opts": {"k": [1, 2], "z": "x"}, "numstr": "123", "nothing": null}`
	turns := []Turn{
		{Role: "user", Content: "go"},
		{Role: "assistant", Content: "", ToolCalls: []ToolCall{{Name: "search", Arguments: json.RawMessage(args)}}},
	}
	prompt := native.RenderTools("", turns, []Tool{searchTool})
	i := strings.Index(prompt, "<tool_call>\n<function=search>")
	j := strings.Index(prompt[i:], "</tool_call>")
	if i < 0 || j < 0 {
		t.Fatalf("no rendered call in:\n%s", prompt)
	}
	calls, _ := native.ParseToolCallsFor(prompt[i:i+j+len("</tool_call>")], []Tool{searchTool})
	if len(calls) != 1 || !jsonEqual(t, string(calls[0].Arguments), args) {
		t.Errorf("round trip:\n sent %s\n back %+v", args, calls)
	}
	// And a multi-line, quoted, markup-bearing string survives (values are raw text between the tags).
	nasty := `{"query": "line1\nline2 \"quoted\" <b>it's</b> </parameter> & done"}`
	turns[1].ToolCalls = []ToolCall{{Name: "search", Arguments: json.RawMessage(nasty)}}
	prompt = native.RenderTools("", turns, []Tool{searchTool})
	i = strings.Index(prompt, "<tool_call>\n<function=search>")
	calls, _ = native.ParseToolCallsFor(prompt[i:], []Tool{searchTool})
	if len(calls) != 1 {
		t.Fatalf("nasty value: %+v", calls)
	}
	if !jsonEqual(t, string(calls[0].Arguments), nasty) {
		t.Errorf("a value that contains </parameter> must survive whole:\n got  %s\n want %s", calls[0].Arguments, nasty)
	}
}

// Only a template goinfer recognises as managed opts in. Gemma 4's canonical template does; an older Gemma 4 template, whose loop goinfer
// has no oracle for, does not — so `-tool-format template` leaves it exactly as it was.
func TestDeclaresNativeTools_gemma4(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []struct {
		Checkpoint   string `json:"checkpoint"`
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	for _, g := range gs {
		if g.Checkpoint != "gemma-4-26b-a4b-it" {
			continue
		}
		canonical, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if !canonical.DeclaresNativeTools() {
			t.Error("Gemma 4's canonical template must declare a native tool form")
		}
		older, _ := Detect(Meta{ChatTemplate: strings.ReplaceAll(g.ChatTemplate, "enable_thinking | default(false)", "enable_thinking is defined and enable_thinking")})
		if older == nil || older.DeclaresNativeTools() {
			t.Error("a Gemma 4 template goinfer cannot recognise as managed must not opt in")
		}
		turns := []Turn{{Role: "user", Content: "Weather?"}, {Role: "assistant", Content: "Checking.", ToolCalls: []ToolCall{{Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)}}}, {Role: "tool", ToolName: "get_weather", Content: "18C"}}
		tools := []Tool{{Name: "get_weather", Description: "W", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}
		if a, b := older.RenderTools("", turns, tools), older.WithToolFormat(ToolFormatTemplate).RenderTools("", turns, tools); a != b {
			t.Error("-tool-format template must be a no-op on a template with no native form")
		}
		return
	}
	t.Fatal("no gemma-4 template in the history golden")
}
