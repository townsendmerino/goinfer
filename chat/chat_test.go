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
