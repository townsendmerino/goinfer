package serveapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// tinyChatTemplate is a plain ChatML template: chat.Detect reads it as ChatML (the <|im_start|> markers), so a chat
// request through the real handlers renders a prompt.
const tinyChatTemplate = `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`

// tinyChatTokenizer is a byte-level BPE tokenizer.json small enough for the tiny fixtures' 256-id vocabularies: printable
// ASCII, the byte-level space and newline, ChatML's three special tokens and one merge. It encodes about one character per
// token, which is all a handler test needs.
func tinyChatTokenizer(t *testing.T) []byte {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := '!'; c <= '~'; c++ {
		vocab[string(c)] = len(vocab)
	}
	var added []map[string]any
	for _, s := range []string{"<|endoftext|>", "<|im_start|>", "<|im_end|>"} {
		id := len(vocab)
		vocab[s] = id
		added = append(added, map[string]any{"id": id, "content": s, "special": true})
	}
	// One merge, so the tokenizer loads as an encoder (a BPE with no merge ranks is decode-only); "hi" is otherwise unused.
	vocab["hi"] = len(vocab)
	b, err := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"h i"}},
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false},
		"decoder":       map[string]any{"type": "ByteLevel"},
		"added_tokens":  added,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// tinyChatCheckpoint is a loadable chat model built from COMMITTED files: testdata/llama-tiny's config and weights, plus
// tinyChatTokenizer and tinyChatTemplate, in a temp dir. Handler tests that need a real model with a tokenizer and a chat
// template run on it in CI. The tiny GGUFs carry no tokenizer, and no committed checkpoint carries one.
func tinyChatCheckpoint(t *testing.T) string {
	t.Helper()
	src := filepath.Join("..", "..", "testdata", "llama-tiny")
	dir := t.TempDir()
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatalf("committed fixture %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tinyChatTokenizer(t), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tinyChatTemplate), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
