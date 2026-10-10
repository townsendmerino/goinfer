package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// G-S14e4a (i) of docs/tasks/task-multimodal-support-2026-10.md: Load(dir) on a directory that holds tekken.json and no tokenizer.json (Voxtral Mini) is the Tekken tokenizer: it gives
// mistral_common's ids on every golden string, exactly as LoadTekken of the same bytes does. A directory with neither file still fails, naming tokenizer.json.
func TestLoad_tekkenDirectory(t *testing.T) {
	g, raw := loadTekkenGolden(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tekken.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(dir with tekken.json): %v", err)
	}
	if n, first := stringMismatches(t, tk, g, plainIDs); n != 0 {
		t.Fatalf("Load(dir) differs from mistral_common on %d of %d golden strings (first %d)", n, len(g.Strings), first)
	}
	if id, ok := tk.TokenID("[AUDIO]"); !ok || id != 24 {
		t.Fatalf("[AUDIO] = %d (found %v), want 24", id, ok)
	}
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("Load of an empty directory succeeded")
	}
}
