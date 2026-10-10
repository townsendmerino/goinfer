package multimodal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S14e4a (ii) of docs/tasks/task-multimodal-support-2026-10.md: VoxtralPrompt with no language, on the real tekken.json, is transformers 5.15.0's own
// apply_transcription_request(language=None) request id for id, for a 3 s clip, the registered 5.9 s clip and the 36 s two-window clip (scripts/pin_voxtral_nolang.py; tokenizer and processor only).
func TestVoxtralPrompt_noLanguageMatchesProcessor(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "voxtral_nolang_ids.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Cases map[string]struct {
			Samples int
			IDs     []int
		}
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		root = filepath.Join(home, "models")
	}
	tekken, err := os.ReadFile(filepath.Join(root, "voxtral-mini-3b-2507", "tekken.json"))
	if err != nil {
		t.Skipf("no tekken.json: %v", err)
	}
	tk, err := tokenizer.LoadTekken(tekken)
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.Cases) != 3 {
		t.Fatalf("the reference has %d cases, want 3", len(ref.Cases))
	}
	for name, c := range ref.Cases {
		ids, _, err := VoxtralPrompt(tk, VoxtralAudioTokens(c.Samples), "")
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, c.IDs) {
			t.Errorf("%s: the Go prompt (%d ids) differs from the processor's (%d ids)", name, len(ids), len(c.IDs))
		}
	}
}
