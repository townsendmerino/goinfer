package chat

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

var weatherTool = Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	Parameters: json.RawMessage(`{"type":"object","properties":{` +
		`"location":{"type":"string","description":"City name"},` +
		`"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}`),
}

// TestParseToolCalls feeds each family the call exactly as its model emits it and
// checks we recover the name + arguments (parse against the template, not a naive
// JSON scan).
func TestParseToolCalls(t *testing.T) {
	cases := map[string]struct {
		fam, out string
	}{
		"chatml":  {"chatml", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"},
		"mellum2": {"mellum2", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"},
		"mistral": {"mistral", "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]"},
		"llama3":  {"llama3", "{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\"}}"},
		"gemma4":  {"gemma4", "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			tmpl := byName(t, c.fam)
			calls, _ := tmpl.ParseToolCalls(c.out)
			if len(calls) != 1 {
				t.Fatalf("got %d calls, want 1 (%q)", len(calls), c.out)
			}
			if calls[0].Name != "get_weather" {
				t.Errorf("name = %q, want get_weather", calls[0].Name)
			}
			var args map[string]any
			if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
				t.Fatalf("arguments not JSON object: %q (%v)", calls[0].Arguments, err)
			}
			if args["location"] != "Paris" {
				t.Errorf("location = %v, want Paris", args["location"])
			}
		})
	}
}

// TestParseToolCalls_trailing checks the parsers tolerate trailing content (real
// model output ends with an EOS / turn marker) and leading natural-language text.
func TestParseToolCalls_trailing(t *testing.T) {
	cases := map[string]struct{ fam, out string }{
		"chatml":  {"chatml", "Sure!<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call><|im_end|>"},
		"mistral": {"mistral", "[TOOL_CALLS] [{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}]</s>"},
		"llama3":  {"llama3", "{\"name\":\"get_weather\",\"parameters\":{\"location\":\"Paris\"}}<|eot_id|>"},
		"gemma4":  {"gemma4", "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|><turn|>"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			calls, _ := byName(t, c.fam).ParseToolCalls(c.out)
			if len(calls) != 1 || calls[0].Name != "get_weather" {
				t.Fatalf("got %d calls (want 1) from %q", len(calls), c.out)
			}
			var args map[string]any
			_ = json.Unmarshal(calls[0].Arguments, &args)
			if args["location"] != "Paris" {
				t.Errorf("location = %v", args["location"])
			}
		})
	}
}

// TestRenderTools_declarations checks each family's declaration prompt carries
// the tool so the model knows it exists (structural — the embedded tool JSON's
// exact spacing isn't byte-matched, unlike the chat templates).
func TestRenderTools_declarations(t *testing.T) {
	markers := map[string][]string{
		"chatml":  {"<tools>", "get_weather", "<tool_call>"},
		"mellum2": {"<tools>", "get_weather", "<tool_call>"},
		"mistral": {"[AVAILABLE_TOOLS]", "get_weather"},
		"llama3":  {"function call", "get_weather"},
		"gemma4":  {"<|tool>declaration:get_weather", `<|"|>OBJECT<|"|>`},
	}
	for fam, want := range markers {
		tmpl := byName(t, fam)
		got := tmpl.RenderTools("", []Turn{{Role: "user", Content: "hi"}}, []Tool{weatherTool})
		for _, m := range want {
			if !strings.Contains(got, m) {
				t.Errorf("%s declaration missing %q in:\n%s", fam, m, got)
			}
		}
	}
}

// TestGemma4_declaration_byteExact pins the Gemma 4 declaration micro-language
// against the HF golden (its schema rendering is deterministic).
func TestGemma4_declaration_byteExact(t *testing.T) {
	timeNow = func() time.Time { return time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC) }
	defer func() { timeNow = time.Now }()
	g := loadToolGolden(t, "gemma4")
	if g == nil {
		return
	}
	want := goldenCase(g, "declare")
	got := Gemma4().RenderTools("", []Turn{{Role: "user", Content: "Weather in Paris?"}}, []Tool{weatherTool})
	if got != want {
		t.Errorf("gemma4 declaration not byte-exact:\n got: %q\nwant: %q", got, want)
	}
}

func byName(t *testing.T, fam string) *Template {
	t.Helper()
	for _, c := range ctors {
		if tmpl := c(); tmpl.Name() == fam {
			return tmpl
		}
	}
	t.Fatalf("no template %q", fam)
	return nil
}

type toolGolden struct {
	Family string `json:"family"`
	Cases  []struct {
		Name     string `json:"name"`
		Rendered string `json:"rendered"`
	} `json:"cases"`
}

func loadToolGolden(t *testing.T, fam string) *toolGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/chat_goldens/tools_" + fam + ".json")
	if err != nil {
		t.Skipf("no tool golden for %s", fam)
		return nil
	}
	var g toolGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return &g
}

func goldenCase(g *toolGolden, name string) string {
	for _, c := range g.Cases {
		if c.Name == name {
			return c.Rendered
		}
	}
	return ""
}
