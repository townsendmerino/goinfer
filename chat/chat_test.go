package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type goldenFile struct {
	Family       string `json:"family"`
	ChatTemplate string `json:"chat_template"`
	Cases        []struct {
		Name     string `json:"name"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Rendered string `json:"rendered"`
		// UpstreamOnly marks a case that records what the MODEL'S template produces where this
		// renderer deliberately differs, rather than what this renderer must produce. Only
		// chatml/no_system uses it today (N-37: Qwen 2.5 inserts a default system prompt; the
		// generic ChatML renderer, shared with non-Qwen families, does not). The divergence is
		// asserted in full by TestChatML_noSystem_documentedDivergence — this flag keeps the
		// equality sweep below from failing on it, and is opt-in per case so it cannot quietly
		// excuse a real regression.
		UpstreamOnly bool `json:"upstream_only"`
	} `json:"cases"`
}

var ctors = map[string]func() *Template{
	"gemma3": Gemma3, "gemma4": Gemma4, "chatml": ChatML, "llama3": Llama3, "mistral": Mistral, "mellum2": Mellum2,
	"ministral": Ministral, "phi3": Phi3, "phi3_orig": Phi3Orig,
}

func loadGoldens(t *testing.T) []goldenFile {
	t.Helper()
	paths, _ := filepath.Glob("../testdata/chat_goldens/*.json")
	if len(paths) == 0 {
		t.Skip("no chat goldens; run scripts/gen_chat_goldens.py")
	}
	var gs []goldenFile
	for _, p := range paths {
		if strings.HasPrefix(filepath.Base(p), "tools_") {
			continue // tool-render goldens are exercised by tools_test.go
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		var g goldenFile
		if err := json.Unmarshal(raw, &g); err != nil {
			t.Fatalf("parse %s: %v", p, err)
		}
		gs = append(gs, g)
	}
	return gs
}

// TestRender_goldens is the acceptance gate: each family's renderer must match
// HuggingFace apply_chat_template byte-for-byte.
func TestRender_goldens(t *testing.T) {
	// Pin the clock to the date baked into the fixtures (Llama-3's preamble).
	timeNow = func() time.Time { return time.Date(2025, time.January, 1, 0, 0, 0, 0, time.UTC) }
	defer func() { timeNow = time.Now }()

	for _, g := range loadGoldens(t) {
		ctor, ok := ctors[g.Family]
		if !ok {
			t.Errorf("no renderer for family %q", g.Family)
			continue
		}
		for _, c := range g.Cases {
			var system string
			var turns []Turn
			for _, m := range c.Messages {
				if m.Role == "system" {
					system = m.Content
				} else {
					turns = append(turns, Turn{Role: m.Role, Content: m.Content})
				}
			}
			if c.UpstreamOnly {
				continue // see UpstreamOnly's doc comment; asserted by its own test
			}
			got := ctor().Render(system, turns)
			if got != c.Rendered {
				t.Errorf("%s/%s render mismatch:\n got: %q\nwant: %q", g.Family, c.Name, got, c.Rendered)
			}
		}
	}
}

// TestDetect_fromTemplate confirms each family's real chat_template string
// fingerprints to the right renderer.
func TestDetect_fromTemplate(t *testing.T) {
	for _, g := range loadGoldens(t) {
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Errorf("%s: Detect from template: %v", g.Family, err)
			continue
		}
		if tmpl.Name() != g.Family {
			t.Errorf("%s: Detect → %q, want %q", g.Family, tmpl.Name(), g.Family)
		}
	}
}

// TestDetect_declinesSmolLM3AndOlmo3 is M-36's decline gate (docs/audit-2026-09-10.md): both
// families use plain <|im_start|>/<|im_end|> markers — the same substring ChatML's own Detect
// test matches — but each diverges from generic ChatML enough that silently rendering it that
// way would be wrong, not just imprecise: SmolLM3 (HuggingFaceTB/SmolLM3-3B) always emits its own
// "## Metadata" system preamble the caller never asked for; Olmo 3 (allenai/Olmo-3-7B-Instruct)
// uses <functions>/<function_calls> XML for tool declarations/calls, not ChatML/Qwen's Hermes
// <tool_call> JSON dialect. Per the user's own design decision (this session, 2026-09-16):
// decline rather than guess at an unverified template, since only Ministral 3 was independently
// confirmed enough to be worth a real renderer. Both fingerprints below are the exact real
// substrings fetched live from each checkpoint's own chat_template.jinja on 2026-09-16, not
// synthesized guesses — see the docs/audit-2026-09-10.md closure note for the full excerpts.
func TestDetect_declinesSmolLM3AndOlmo3(t *testing.T) {
	cases := map[string]string{
		"smollm3": `{{- "<|im_start|>system\n" -}}{%- if "/system_override" in system_message -%}{{- custom_instructions -}}{%- else -%}{{- "## Metadata\n\n" -}}{%- endif -%}`,
		"olmo3":   `{%- elif message['role'] == 'assistant' -%}{{- '<|im_start|>assistant\n' -}}{%- if message.get('function_calls', none) is not none -%}{{- '<function_calls>' + message['function_calls'] + '</function_calls>' -}}{%- endif -%}`,
	}
	for name, tmpl := range cases {
		if _, err := Detect(Meta{ChatTemplate: tmpl}); err != ErrUnknownTemplate {
			t.Errorf("%s: Detect from template = %v, want ErrUnknownTemplate — a template using "+
				"generic <|im_start|> markers but a distinctive per-family shape must decline, not "+
				"silently render as plain ChatML (M-36)", name, err)
		}
	}
	// Control: a genuinely plain ChatML template (no "## Metadata", no "<function_calls>") must
	// still match — these two fingerprints must not be so broad they catch ordinary ChatML too.
	if tmpl, err := Detect(Meta{ChatTemplate: "{{ '<|im_start|>' + role + '\\n' + content + '<|im_end|>' }}"}); err != nil || tmpl.Name() != "chatml" {
		t.Errorf("plain ChatML template: Detect = (%v, %v), want (chatml, nil) — the new "+
			"fingerprints are too broad and now shadow ordinary ChatML checkpoints", tmpl, err)
	}
}

// TestDetect_fallback covers the bare-checkpoint heuristic and the unknown case.
func TestDetect_fallback(t *testing.T) {
	cases := map[string]string{
		"<|im_start|>":        "chatml",
		"<start_of_turn>":     "gemma3",
		"<|turn>":             "gemma4",
		"<|start_header_id|>": "llama3",
	}
	for tok, want := range cases {
		marker := tok
		tmpl, err := Detect(Meta{HasToken: func(s string) bool { return s == marker }})
		if err != nil {
			t.Errorf("fallback %q: %v", marker, err)
			continue
		}
		if tmpl.Name() != want {
			t.Errorf("fallback %q → %q, want %q", marker, tmpl.Name(), want)
		}
	}
	// No template, no markers → explicit error for the raw-completion fallback.
	if _, err := Detect(Meta{HasToken: func(string) bool { return false }}); err != ErrUnknownTemplate {
		t.Errorf("bare/unknown: err = %v, want ErrUnknownTemplate", err)
	}
	if _, err := Detect(Meta{ChatTemplate: "{{ some unknown jinja }}"}); err != ErrUnknownTemplate {
		t.Errorf("unknown template: err = %v, want ErrUnknownTemplate", err)
	}
}
