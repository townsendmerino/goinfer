package multimodal

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQwenASRPromptOpenLanguage_realTokenizer is G-S14c4d's prompt control (docs/tasks/task-multimodal-support-2026-10.md): on the real Qwen3-ASR tokenizer the opened prompt is the
// ordinary prompt plus exactly one token, that token is the one the model generates first (the id of "language"), and encoding the text "...assistant\nlanguage" gives the same ids, so
// the prefix is not a tokenisation different from the model's own.
func TestQwenASRPromptOpenLanguage_realTokenizer(t *testing.T) {
	dir := os.Getenv("GOINFER_QWEN3ASR_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3-asr-0.6b")
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Skipf("no Qwen3-ASR tokenizer (%v)", err)
	}
	const n = 12
	base, pos, err := QwenASRPrompt(tk, "", n, "")
	if err != nil {
		t.Fatal(err)
	}
	open, pos2, err := QwenASRPromptOpenLanguage(tk, "", n)
	if err != nil {
		t.Fatal(err)
	}
	if pos != pos2 || len(open) != len(base)+1 || !slices.Equal(open[:len(base)], base) {
		t.Fatalf("the opened prompt is not the base prompt plus one token (len %d vs %d, audioPos %d vs %d)", len(open), len(base), pos2, pos)
	}
	w, ok := tk.TokenID(QwenASRLanguageWord)
	if !ok || open[len(open)-1] != w {
		t.Fatalf("last id %d, want the id of %q (%d, found %v)", open[len(open)-1], QwenASRLanguageWord, w, ok)
	}
	// the generated reply "language English<asr_text>..." must START with this one token, or the forced word is not what the model writes
	ids, err := tk.Encode("language English<asr_text>Hello", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 || ids[0] != w {
		t.Fatalf("the reply text encodes to %v: its first id is not the id of %q (%d)", ids, QwenASRLanguageWord, w)
	}
	// and the text form of the prefix tokenises to the same ids as base + the word
	txt, err := tk.Encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(txt) < 2 || txt[len(txt)-1] != w || txt[len(txt)-2] == w {
		t.Fatalf("the text form ends %v: not the word after the assistant newline", txt[max(0, len(txt)-3):])
	}
}
