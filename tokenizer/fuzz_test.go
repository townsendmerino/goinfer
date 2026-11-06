package tokenizer

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

// tokenizer.json is untrusted input (it ships in a
// model directory). parseTokenizerJSON is the parse surface; it must turn a
// malformed/hostile file into a typed error, never a panic or OOM.

var tokSeeds = []string{
	`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}`,
	`{"model":{"type":"BPE","vocab":{"<s>":0,"a":1},"merges":[]},"added_tokens":[{"id":1,"content":"a"}]}`,
	`{"model":{"vocab":{"x":0},"merges":[]}}`, // empty type (GPT-2)
	// hostile ids — exercise the idToPiece sizing/indexing.
	`{"model":{"type":"BPE","vocab":{"x":2147483647}}}`, // maxID+1 overflows int32
	`{"model":{"type":"BPE","vocab":{"x":-5}}}`,         // negative id
	`{"model":{"type":"BPE","vocab":{"x":1073741824}}}`, // huge positive id → big make
}

// FuzzParseTokenizerJSON feeds arbitrary bytes to the tokenizer.json parser: a
// typed error or a built Tokenizer, never a panic. jsonPath names a directory
// with no sibling files, so a byte-level pipeline cleanly errors rather than
// reading the host filesystem.
func FuzzParseTokenizerJSON(f *testing.F) {
	for _, s := range tokSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		tk, err := parseTokenizerJSON(raw, "fuzz-nonexistent-dir/tokenizer.json", "fuzz-nonexistent-dir")
		if err != nil {
			return
		}
		// A built tokenizer must be self-consistent: every vocab id is a valid
		// index into idToPiece (else Decode/TokenText would panic at runtime).
		for _, id := range tk.vocab {
			if int(id) < 0 || int(id) >= len(tk.idToPiece) {
				t.Fatalf("vocab id %d out of idToPiece range [0,%d)", id, len(tk.idToPiece))
			}
		}
	})
}

// A model file is untrusted, and ggufMeta is a hand-written binary reader in front of the whole tokenizer. Bar: a typed error, never a panic, and no
// allocation a short file can inflate.
func FuzzReadGGUFMeta(f *testing.F) {
	for _, p := range []string{"testdata/chatml-tiny.gguf"} {
		raw, err := os.ReadFile(p)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(raw)
		for _, n := range []int{4, 8, 24, 64, len(raw) / 2, len(raw) - 1} {
			f.Add(raw[:n])
		}
		hostile := append([]byte(nil), raw...)
		binary.LittleEndian.PutUint64(hostile[8:], 1<<62) // tensor count
		f.Add(hostile)
		hostile = append([]byte(nil), raw...)
		binary.LittleEndian.PutUint64(hostile[16:], 1<<62) // key count
		f.Add(hostile)
	}
	f.Add([]byte{})
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, raw []byte) {
		_, _ = readGGUFMeta(bytes.NewReader(raw), int64(len(raw)))
		_, _ = LoadGGUFBytes(raw)
	})
}

// A tekken.json ships in a model directory, as tokenizer.json does.
func FuzzLoadTekken(f *testing.F) {
	f.Add([]byte(`{"config":{"pattern":"\\s+|\\S+","default_vocab_size":4,"default_num_special_tokens":1},"vocab":[{"rank":0,"token_bytes":"YQ==","token_str":"a"}],"special_tokens":[{"rank":0,"token_str":"<s>","is_control":true}]}`))
	f.Add([]byte(`{"config":{"pattern":"("},"vocab":[{"rank":-1,"token_bytes":"!!"}],"special_tokens":[{}]}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`[`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		tk, err := LoadTekken(raw)
		if err != nil {
			return
		}
		ids, err := tk.Encode("hello world", false)
		if err != nil {
			return
		}
		_, _ = tk.Decode(ids)
	})
}

// A prompt's content is client text, malformed UTF-8 included, and is tokenized without the added-token parse; template text takes
// the parse. Every id either gives must be in the vocabulary. On plain ASCII, which no normalizer changes, decoding gives the
// text back, unless the added-token parse took a marker: a marker may swallow the whitespace beside it.
func FuzzEncode(f *testing.F) {
	for _, s := range []string{
		"", " ", "hello world", "don't 2026-11-04 $89.99\n\n\t  x", tkCorpus, "<|im_start|>user\nhi<|im_end|>", "<|im_start", "<tool_call>", "日本語 🦊 é", "\xff\xfe abc", "a\xc3",
		"\xed\xa0\x80 x", strings.Repeat("a", 4096), strings.Repeat(" ", 4096), strings.Repeat("🦊", 512),
	} {
		f.Add(s)
	}
	tk, err := LoadGGUF("testdata/qwen2-0.5b-coder-meta.gguf")
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, s string) {
		for _, parseSpecial := range []bool{false, true} {
			ids, err := tk.encode(s, false, parseSpecial)
			if err != nil {
				t.Fatalf("encode(%q, parseSpecial %v): %v", s, parseSpecial, err)
			}
			for _, id := range ids {
				if id < 0 || id >= len(tk.idToPiece) {
					t.Fatalf("encode(%q, parseSpecial %v) gave id %d of %d", s, parseSpecial, id, len(tk.idToPiece))
				}
			}
			got, err := tk.Decode(ids)
			if err != nil {
				t.Fatalf("Decode of encode(%q, parseSpecial %v): %v", s, parseSpecial, err)
			}
			if !parseSpecial && strings.IndexFunc(s, func(r rune) bool { return r > 0x7f }) < 0 && got != s {
				t.Fatalf("encode then Decode of %q gave %q", s, got)
			}
		}
	})
}
