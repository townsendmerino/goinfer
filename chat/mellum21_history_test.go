package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Mellum2.1's chat template (JetBrains/Mellum2.1-12B-A2.5B-Thinking @ 92ddae9f, 2026-10-09) adds Qwen3's thinking control and
// history rule to 2.0's ChatML body. The golden is Hugging Face's own rendering of that template over the same conversations as
// think_history.json, in every enable_thinking mode (scripts/pin_chat_think_history.py, PIN_CKPTS=mellum2.1; 20 cases x 3 modes).
// Before the Detect and histMellum21 edits goinfer matched 10 of those 60 prompts; every one must now be byte-identical.
func TestMellum21_templateMatchesHuggingFace(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "mellum21_think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []histGolden
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	if len(gs) != 1 {
		t.Fatalf("expected one pinned checkpoint, got %d", len(gs))
	}
	g := gs[0]
	tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if tmpl.Reasoning() == nil {
		t.Fatal("the 2.1 template declares enable_thinking, but Detect attached no reasoning behaviour")
	}
	if got := tmpl.Reasoning().hist; got != histMellum21 {
		t.Fatalf("history rule = %v, want histMellum21", got)
	}
	n := 0
	for _, c := range g.Cases {
		turns := goldenTurns(t, c.Messages)
		for mode, want := range c.Prompts {
			tm := tmpl.WithToolFormat(ToolFormatAuto).WithThinking(histMode[mode])
			var got string
			if c.Tools {
				got = tm.RenderTools("", turns, goldenTools)
				got, want = compactToolCalls(afterSystemTurn(got)), compactToolCalls(afterSystemTurn(want))
			} else {
				got = tm.Render("", turns)
			}
			if got != want {
				t.Errorf("%s / enable_thinking=%s (tools=%v):\n got %q\nwant %q", c.Name, mode, c.Tools, got, want)
			}
			n++
		}
	}
	if n != 60 {
		t.Fatalf("compared %d prompts, want 60 (20 cases x 3 modes)", n)
	}
}
