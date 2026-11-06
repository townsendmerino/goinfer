package chatapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// tinyChatModel is a model directory chat can load and talk to: the 256-token tiny llama fixture's config and weights, with a
// byte-level BPE tokenizer over the same 256 ids (ids 2, 3 and 4 are <|endoftext|>, <|im_start|> and <|im_end|> instead of bytes
// 2 to 4, and id 5 is the merge "he", since a BPE vocab with no merges loads decode-only) and a ChatML template. Its replies are
// noise, the same noise for the same greedy request.
func tinyChatModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		src := filepath.Join("..", "..", "..", "testdata", "llama-tiny", name)
		if err := os.Link(src, filepath.Join(dir, name)); err != nil {
			b, err := os.ReadFile(src)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	specials := map[int]string{2: "<|endoftext|>", 3: "<|im_start|>", 4: "<|im_end|>"}
	vocab := map[string]int{}
	for b, r := range byteLevelRunes() {
		if _, special := specials[b]; !special && b != 5 {
			vocab[string(r)] = b
		}
	}
	vocab["he"] = 5
	var added []map[string]any
	for id, s := range specials {
		vocab[s] = id
		added = append(added, map[string]any{"id": id, "content": s, "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false})
	}
	tok := map[string]any{
		"version": "1.0", "added_tokens": added, "normalizer": nil, "post_processor": nil,
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
		"decoder":       map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"h e"}},
	}
	cfg := map[string]any{
		"chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
		"eos_token":     "<|im_end|>",
	}
	for name, v := range map[string]any{"tokenizer.json": tok, "tokenizer_config.json": cfg} {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// byteLevelRunes is GPT-2's byte-to-unicode table: the printable rune each byte is spelled as in a byte-level vocab.
func byteLevelRunes() [256]rune {
	var out [256]rune
	var direct [256]bool
	for _, r := range [][2]int{{'!', '~'}, {'¡', '¬'}, {'®', 'ÿ'}} {
		for b := r[0]; b <= r[1]; b++ {
			out[b], direct[b] = rune(b), true
		}
	}
	n := 0
	for b := range 256 {
		if !direct[b] {
			out[b] = rune(256 + n)
			n++
		}
	}
	return out
}
