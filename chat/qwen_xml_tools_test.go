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
