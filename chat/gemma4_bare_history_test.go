package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Gemma 4 E4B's current template (google/gemma-4-E4B-it @ ee0ef602) and Nemotron 3.5 Bolt E4B's (nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
// @ 49843eed, the template its Q4_K_M GGUF embeds) are the canonical Gemma 4 template without the closed scaffold after a thinking-off
// generation prompt; Bolt's also writes an assistant message's text before its tool calls. The golden is Hugging Face's own rendering of
// both templates (scripts/pin_chat_think_history.py, PIN_CKPTS=gemma-4-E4B-it,nemotron-3.5-bolt-e4b; 24 cases x 3 modes each, the
// two-turn tool conversation included). Every prompt must be byte-identical, tool declarations included.
func TestGemma4BareOff_templateMatchesHuggingFace(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "gemma4_bare_think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []histGolden
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	wantContentFirst := map[string]bool{"gemma-4-E4B-it": false, "nemotron-3.5-bolt-e4b": true}
	if len(gs) != len(wantContentFirst) {
		t.Fatalf("expected %d pinned checkpoints, got %d", len(wantContentFirst), len(gs))
	}
	for _, g := range gs {
		contentFirst, ok := wantContentFirst[g.Checkpoint]
		if !ok {
			t.Fatalf("unexpected checkpoint %q", g.Checkpoint)
		}
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Fatalf("%s: Detect: %v", g.Checkpoint, err)
		}
		r := tmpl.Reasoning()
		if r == nil || r.hist != histGemma4 || !r.bareOff || r.offScaffold() {
			t.Fatalf("%s: want the canonical Gemma 4 reasoning with a bare thinking-off prompt, got %+v", g.Checkpoint, r)
		}
		if !tmpl.WithToolFormat(ToolFormatAuto).WithThinking(ThinkTemplate).usesNativeTools() {
			t.Fatalf("%s: auto does not select the native tool renderer", g.Checkpoint)
		}
		if tmpl.contentBeforeCalls != contentFirst {
			t.Fatalf("%s: contentBeforeCalls = %v, want %v", g.Checkpoint, tmpl.contentBeforeCalls, contentFirst)
		}
		n := 0
		for _, c := range g.Cases {
			turns := goldenTurns(t, c.Messages)
			for mode, want := range c.Prompts {
				tm := tmpl.WithToolFormat(ToolFormatAuto).WithThinking(histMode[mode])
				var got string
				if c.Tools {
					got = tm.RenderTools("", turns, goldenTools)
				} else {
					got = tm.Render("", turns)
				}
				if got != want {
					t.Errorf("%s / %s / enable_thinking=%s (tools=%v):\n got %q\nwant %q", g.Checkpoint, c.Name, mode, c.Tools, got, want)
				}
				n++
			}
		}
		if n != 72 {
			t.Fatalf("%s: compared %d prompts, want 72 (24 cases x 3 modes)", g.Checkpoint, n)
		}
	}
}
