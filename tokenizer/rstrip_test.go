package tokenizer

import (
	"fmt"
	"strings"
	"testing"
)

// A Llama-2-style SentencePiece tokenizer.json (▁ dummy prefix, byte fallback) whose two added
// turn markers carry rstrip, as Phi-3's do. rstripFlag is substituted per case.
func rstripTokenizer(t *testing.T, rstripFlag string) *Tokenizer {
	t.Helper()
	blob := `{"added_tokens":[` +
		`{"id":9,"content":"<|user|>","special":true,"rstrip":RS},` +
		`{"id":10,"content":"<|end|>","special":true,"rstrip":RS}],` +
		`"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},` +
		`"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},` +
		`"model":{"type":"BPE","byte_fallback":true,"unk_token":"<unk>",` +
		`"vocab":{"<unk>":0,"<s>":1,"<bos>":267,"<eos>":268,"<pad>":269,"▁":2,"H":3,"i":4,"▁H":5,"▁Hi":6,"Hi":7,"<|user|>":9,"<|end|>":10,BYTES},` +
		`"merges":["▁ H","▁H i","H i"]}}`
	var bytes []string // byte fallback needs every <0xNN> piece; "\n" is <0x0A> = id 21
	for b := range 256 {
		bytes = append(bytes, fmt.Sprintf(`"<0x%02X>":%d`, b, 11+b))
	}
	blob = strings.ReplaceAll(blob, "BYTES", strings.Join(bytes, ","))
	tk, err := LoadJSONBytes([]byte(strings.ReplaceAll(blob, "RS", rstripFlag)))
	if err != nil {
		t.Fatalf("LoadJSONBytes: %v", err)
	}
	return tk
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
