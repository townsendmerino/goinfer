package tokenizer

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestGlmOcrTokenizer_matchesHF: goinfer's tokenizer yields ids IDENTICAL to HF's AutoTokenizer on
// the real zai-org/GLM-OCR tokenizer.json, over digit runs of 1-7 digits (the Split regex caps
// digits at 3 per piece), newlines and runs of them, leading/trailing/repeated spaces, CJK, and
// EVERY added token 59246-59281 — alone, flanked, doubled and embedded in the real chat-template
// shape — including the ones tokenizer.json marks special=false (<think>, </think>, the tool tags,
// <|begin_of_video|>, the box tags, <|image|>, <|video|>). The golden (scripts/pin_glm_ocr_tokenizer.py)
// is HF's output with add_special_tokens=False, so this pins text->ids, not a template.
//
// The 6.8 MB tokenizer.json is not committed: the test skips cleanly without
// $GOINFER_MODELS_DIR/glm-ocr (default ~/models/glm-ocr). Read for --- PASS, not ok.
func TestGlmOcrTokenizer_matchesHF(t *testing.T) {
	raw, err := os.ReadFile("testdata/glm_ocr_tokenizer_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_glm_ocr_tokenizer.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Added []struct {
			Content string `json:"content"`
			ID      int    `json:"id"`
		} `json:"added"`
		Cases []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	dir := modelPath("glm-ocr")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tokenizer.json under %s (zai-org/GLM-OCR; set GOINFER_MODELS_DIR)", dir)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(g.Added) != 36 {
		t.Fatalf("golden lists %d added tokens, want 36 (59246..59281)", len(g.Added))
	}
	// Each added token's id resolves to its content in goinfer's vocabulary.
	for _, a := range g.Added {
		if id, ok := tk.TokenID(a.Content); !ok || id != a.ID {
			t.Errorf("TokenID(%q) = %d, %v; want %d", a.Content, id, ok, a.ID)
		}
	}
	bad := 0
	for _, c := range g.Cases {
		got, err := tk.Encode(c.Text, false)
		if err != nil {
			t.Errorf("Encode(%q): %v", c.Text, err)
			bad++
			continue
		}
		if !slices.Equal(got, c.IDs) {
			bad++
			if bad <= 12 {
				t.Errorf("MISMATCH %q\n  goinfer %v\n  HF      %v", c.Text, got, c.IDs)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d cases differ from HF", bad, len(g.Cases))
	}
	t.Logf("%d cases identical to HF's AutoTokenizer, covering %d added tokens (%d with special=false)", len(g.Cases)-bad, len(g.Added), 15)
}
