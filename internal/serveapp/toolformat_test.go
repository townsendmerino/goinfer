package serveapp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

func TestToolFormatDefault(t *testing.T) {
	for in, want := range map[string]chat.ToolFormat{"": chat.ToolFormatHermes, "hermes": chat.ToolFormatHermes, "template": chat.ToolFormatTemplate, " Template ": chat.ToolFormatTemplate} {
		if got := (config{toolFormat: in}).toolFormatDefault(); got != want {
			t.Errorf("-tool-format %q resolved to %v, want %v", in, got, want)
		}
	}
	if _, ok := chat.ParseToolFormat("xml"); ok {
		t.Error("an unknown -tool-format must be refused at startup")
	}
}

var qwenToolsForTest = []chat.Tool{{Name: "get_weather", Description: "Weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}}

// The server default is applied to the model's template once, at load; every per-request layer after it (the request's thinking mode, a
// reasoning effort) must keep it. Hermes stays the default and renders goinfer's own prompt; template renders the model's own.
func TestToolFormat_survivesRequestLayers(t *testing.T) {
	base := templateFromGolden(t, "qwen3.5-9b")
	turns := []chat.Turn{{Role: "user", Content: "Weather in Paris?"}}
	isNative := func(tm *chat.Template) bool {
		return strings.Contains(tm.RenderTools("", turns, qwenToolsForTest), "<function=example_function_name>")
	}

	hermes := &loadedModel{tmpl: base.WithThinking(chat.ThinkTemplate).WithToolFormat(chat.ToolFormatHermes)}
	native := &loadedModel{tmpl: base.WithThinking(chat.ThinkTemplate).WithToolFormat(chat.ToolFormatTemplate)}
	for name, ts := range map[string]thinkSettings{
		"no request settings": {},
		"thinking off":        {explicit: true, mode: chat.ThinkOff},
		"thinking on":         {explicit: true, mode: chat.ThinkOn},
		"an effort":           {effort: "high"},
		"both":                {explicit: true, mode: chat.ThinkOff, effort: "low"},
	} {
		if isNative(hermes.templateFor(ts)) {
			t.Errorf("hermes + %s: rendered the native form", name)
		}
		if !isNative(native.templateFor(ts)) {
			t.Errorf("template + %s: lost the native form", name)
		}
	}
	// A model with no native form is untouched by the default.
	other := &loadedModel{tmpl: chat.ChatML().WithToolFormat(chat.ToolFormatTemplate)}
	if isNative(other.templateFor(thinkSettings{})) {
		t.Error("a ChatML template with no native form must keep the Hermes prompt")
	}
}

// Under the native form there is no JSON wrapper to constrain a decode to, so a tool_choice that NAMES a function cannot be honoured and is
// a 400 (as for Gemma 4 and gpt-oss); a union / lone-tool choice runs unconstrained rather than failing. Hermes keeps its constraint.
func TestToolFormat_forcedToolUnderNative(t *testing.T) {
	base := templateFromGolden(t, "qwen3.5-9b").WithThinking(chat.ThinkTemplate)
	forced := &qwenToolsForTest[0]

	native := &loadedModel{tmpl: base.WithToolFormat(chat.ToolFormatTemplate)}
	gr := genRequest{}
	err := constrainForcedTool(native, &gr, forced, true, "", qwenToolsForTest)
	if err == nil || !strings.Contains(err.Error(), "no constrainable tool-call form") {
		t.Errorf("a named tool_choice under the native form must be refused, got %v", err)
	}
	if gr.sp.LogitProcessor != nil {
		t.Error("nothing may be constrained under the native form")
	}
	gr = genRequest{}
	if err := constrainForcedTool(native, &gr, forced, false, "", qwenToolsForTest); err != nil || gr.sp.LogitProcessor != nil {
		t.Errorf("a lone-tool convenience must run unconstrained without an error: %v", err)
	}
	if err := constrainForcedTool(native, &gr, nil, false, "required", append([]chat.Tool{}, qwenToolsForTest...)); err != nil || gr.sp.LogitProcessor != nil {
		t.Errorf("a union choice must run unconstrained without an error: %v", err)
	}
}

func TestToolFormatNote(t *testing.T) {
	base := templateFromGolden(t, "qwen3.5-9b")
	if got := toolFormatNote(base); !strings.Contains(got, "goinfer's Hermes form") || !strings.Contains(got, "-tool-format template") {
		t.Errorf("default note: %q", got)
	}
	if got := toolFormatNote(base.WithToolFormat(chat.ToolFormatTemplate)); !strings.Contains(got, "the model's own template form") {
		t.Errorf("native note: %q", got)
	}
	for _, tm := range []*chat.Template{chat.ChatML(), chat.Llama3(), chat.Harmony(), nil} {
		if got := toolFormatNote(tm); got != "" {
			t.Errorf("a model with no native form gets no note, got %q", got)
		}
	}
}
