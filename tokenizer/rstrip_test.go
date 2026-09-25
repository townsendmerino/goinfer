package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A Llama-2-style SentencePiece tokenizer.json (▁ dummy prefix, byte fallback, "<s>"/"</s>" and no
// pad, as Phi-3's) whose two added turn markers carry rstrip, as Phi-3's do. rstripFlag is
// substituted per case.
func rstripTokenizer(t *testing.T, rstripFlag string) *Tokenizer {
	t.Helper()
	tk, err := LoadJSONBytes([]byte(strings.ReplaceAll(llamaStyleJSON(t), "RS", rstripFlag)))
	if err != nil {
		t.Fatalf("LoadJSONBytes: %v", err)
	}
	return tk
}

func llamaStyleJSON(t *testing.T) string {
	t.Helper()
	blob := `{"added_tokens":[` +
		`{"id":9,"content":"<|user|>","special":true,"rstrip":RS},` +
		`{"id":10,"content":"<|end|>","special":true,"rstrip":RS}],` +
		`"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},` +
		`"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},` +
		`"model":{"type":"BPE","byte_fallback":true,"unk_token":"<unk>",` +
		`"vocab":{"<unk>":0,"<s>":1,"</s>":267,"▁":2,"H":3,"i":4,"▁H":5,"▁Hi":6,"Hi":7,"<|user|>":9,"<|end|>":10,BYTES},` +
		`"merges":["▁ H","▁H i","H i"]}}`
	var bytes []string // byte fallback needs every <0xNN> piece; "\n" is <0x0A> = id 21
	for b := range 256 {
		bytes = append(bytes, fmt.Sprintf(`"<0x%02X>":%d`, b, 11+b))
	}
	return strings.ReplaceAll(blob, "BYTES", strings.Join(bytes, ","))
}

// TestAddedToken_rstrip: an rstrip added token swallows the whitespace after it, so
// "<|user|>\nHi" encodes as <|user|> ▁Hi — HF and llama.cpp both produce that for Phi-3, where
// goinfer used to emit <|user|> ▁ \n Hi (measured 2026-09-25 on the real Phi-3-mini GGUF: three
// extra tokens per turn marker). Both halves are checked: the whole-string encoder, and
// EncodeSegments, where the marker and the newline sit in DIFFERENT segments and the strip has
// to cross the boundary. The rstrip=false case proves the premise in the same test.
func TestAddedToken_rstrip(t *testing.T) {
	const text = "<|user|>\nHi<|end|>\n"
	segs := []Segment{{Text: "<|user|>", Special: true}, {Text: "\nHi"}, {Text: "<|end|>", Special: true}, {Text: "\n"}}
	for _, c := range []struct {
		flag string
		want []int
	}{
		{"false", []int{9, 2, 21, 7, 10, 2, 21}},
		{"true", []int{9, 6, 10}},
	} {
		tk := rstripTokenizer(t, c.flag)
		got, err := tk.Encode(text, false)
		if err != nil {
			t.Fatal(err)
		}
		if !equalInts(got, c.want) {
			t.Errorf("rstrip=%s Encode = %v, want %v", c.flag, got, c.want)
		}
		got, err = tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		if !equalInts(got, c.want) {
			t.Errorf("rstrip=%s EncodeSegments = %v, want %v", c.flag, got, c.want)
		}
	}
}

// TestLoadJSON_llamaStyleSpecials: a SentencePiece tokenizer.json spelling its specials "<s>"/"</s>"
// with no pad (Llama-2, Mistral, Phi-3) must load. The loader used to require Gemma's
// "<bos>"/"<eos>"/"<pad>", so Phi-3 from safetensors could not tokenize at all (2026-09-25). A
// vocab with neither BOS spelling still fails, as Gemma's contract requires.
func TestLoadJSON_llamaStyleSpecials(t *testing.T) {
	tk := rstripTokenizer(t, "true")
	if tk.special.BOS != 1 || tk.special.EOS != 267 || tk.special.Pad != -1 {
		t.Errorf("BOS/EOS/Pad = %d/%d/%d, want 1/267/-1", tk.special.BOS, tk.special.EOS, tk.special.Pad)
	}
	noBOS := strings.ReplaceAll(strings.ReplaceAll(llamaStyleJSON(t), "RS", "true"), `"<s>":1,`, "")
	if _, err := LoadJSONBytes([]byte(noBOS)); err == nil {
		t.Error("a SentencePiece vocab with neither <bos> nor <s> loaded; want an error")
	}
}

// TestLoad_spmReadsSiblingChatTemplate: a SentencePiece checkpoint's chat template lives in
// tokenizer_config.json beside tokenizer.json. Only the byte-level path read it, so Phi-3 and
// Mistral safetensors reached chat.Detect with none and ran as raw completions. A directory load
// reads it; a blob load (no siblings, M-14) does not.
func TestLoad_spmReadsSiblingChatTemplate(t *testing.T) {
	dir := t.TempDir()
	blob := strings.ReplaceAll(llamaStyleJSON(t), "RS", "true")
	const tmpl = "{% for message in messages %}<|user|>{{ message['content'] }}<|end|>{% endfor %}"
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(blob), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, _ := json.Marshal(map[string]string{"chat_template": tmpl})
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if tk.ChatTemplate() != tmpl {
		t.Errorf("directory load ChatTemplate = %q, want the sibling config's", tk.ChatTemplate())
	}
	if tk, err = LoadJSONBytes([]byte(blob)); err != nil || tk.ChatTemplate() != "" {
		t.Errorf("blob load ChatTemplate = %q (err %v), want empty: no sibling reads (M-14)", tk.ChatTemplate(), err)
	}
}
