package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestProgramFixtureDirs_tinyIsANameToken: programFixtureDirs takes the tiny fixtures and not the real checkpoints whose names
// merely contain "tiny" (TinyLlama 1.1B in AWQ and GPTQ, TinyMistral 248M), which would load at f32 as full models.
func TestProgramFixtureDirs_tinyIsANameToken(t *testing.T) {
	root := t.TempDir()
	for _, d := range []string{"llama-tiny", "tiny-qwen2-moe", "gemma4-moe-tiny-k3", "gemma4-moe-kv-tiny32", "qwen3_5-tiny", "tinyllama-awq", "tinyllama-gptq",
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
	dirs, refused := programFixtureDirs([]string{root})
	if len(refused) > 0 {
		t.Errorf("refused %v: every fixture here is empty", refused)
	}
	var got []string
	for _, d := range dirs {
		got = append(got, filepath.Base(d))
	}
	want := []string{"gemma1-tiny.gguf", "gemma4-moe-kv-tiny32", "gemma4-moe-tiny-k3", "gptoss_tiny.gguf", "llama-tiny", "qwen3_5-tiny", "tiny-qwen2-moe", "whisper-tiny-ts"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("programFixtureDirs picked\n  %v\nwant\n  %v", got, want)
	}
}

// TestProgramFixtureDirs_refusesOverTheWeightCap is the size cap's plant: a directory and a .gguf file the name filter admits,
// each with weights one byte over programFixtureWeightCap, are refused by name and not listed; a directory at the cap, and
// a non-weight file over it, are listed. The files are sparse, so the test writes no data.
func TestProgramFixtureDirs_refusesOverTheWeightCap(t *testing.T) {
	root := t.TempDir()
	sized := func(path string, n int64) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(path, n); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range []string{"big-tiny", "sharded-tiny", "at-cap-tiny", "tokenizer-tiny"} {
		sized(filepath.Join(root, d, "config.json"), 2)
	}
	sized(filepath.Join(root, "big-tiny", "model.safetensors"), programFixtureWeightCap+1)
	sized(filepath.Join(root, "sharded-tiny", "model-00001-of-00002.safetensors"), programFixtureWeightCap/2)
	sized(filepath.Join(root, "sharded-tiny", "model-00002-of-00002.safetensors"), programFixtureWeightCap/2+1)
	sized(filepath.Join(root, "at-cap-tiny", "model.safetensors"), programFixtureWeightCap)
	sized(filepath.Join(root, "tokenizer-tiny", "tokenizer.json"), programFixtureWeightCap+1)
	sized(filepath.Join(root, "big_tiny.gguf"), programFixtureWeightCap+1)

	dirs, refused := programFixtureDirs([]string{root})
	var got []string
	for _, d := range dirs {
		got = append(got, filepath.Base(d))
	}
	if want := []string{"at-cap-tiny", "tokenizer-tiny"}; !reflect.DeepEqual(got, want) {
		t.Errorf("listed %v, want %v", got, want)
	}
	var names []string
	for _, r := range refused {
		name, _, _ := strings.Cut(r, " ")
		names = append(names, name)
	}
	if want := []string{"big-tiny", "big_tiny.gguf", "sharded-tiny"}; !reflect.DeepEqual(names, want) {
		t.Errorf("refused %q, want by name %v", refused, want)
	}
}
