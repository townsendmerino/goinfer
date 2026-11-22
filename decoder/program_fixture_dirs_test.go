package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestProgramFixtureDirs_tinyIsANameToken: programFixtureDirs takes the tiny fixtures and not the real checkpoints whose names
// merely contain "tiny" (TinyLlama 1.1B in AWQ and GPTQ, TinyMistral 248M), which would load at f32 as full models.
func TestProgramFixtureDirs_tinyIsANameToken(t *testing.T) {
	root := t.TempDir()
	for _, d := range []string{"llama-tiny", "tiny-qwen2-moe", "gemma4-moe-tiny-k3", "qwen3_5-tiny", "tinyllama-awq", "tinyllama-gptq",
		"tinymistral-248m", "whisper-tiny-ts", "notiny-x"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, d, "config.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(root, "mixtral-tiny-noconfig"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"gptoss_tiny.gguf", "gemma1-tiny.gguf", "tinyllama-q4.gguf", "llama_tiny_text_golden.json"} {
		if err := os.WriteFile(filepath.Join(root, f), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var got []string
	for _, d := range programFixtureDirs([]string{root}) {
		got = append(got, filepath.Base(d))
	}
	want := []string{"gemma1-tiny.gguf", "gemma4-moe-tiny-k3", "gptoss_tiny.gguf", "llama-tiny", "qwen3_5-tiny", "tiny-qwen2-moe", "whisper-tiny-ts"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("programFixtureDirs picked\n  %v\nwant\n  %v", got, want)
	}
}
