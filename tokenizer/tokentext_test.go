package tokenizer

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestTokenText_reconstructs: concatenating TokenText over an encoding must
// reproduce Decode byte-for-byte, for tokenizers without the SPM leading-space
// strip (Gemma byte-fallback and the byte-level families). This is the contract
// constrained decoding relies on — that a token's masked surface bytes are
// exactly what it contributes to the output. Skips when a tokenizer is absent.
func TestTokenText_reconstructs(t *testing.T) {
	cases := []struct {
		dir  string
		mode tokMode
	}{
		{"../testdata/gemma-3-270m", modeGemma},
		{"../testdata/llama3-tokenizer", modeByteLevel},
		{"../testdata/qwen3-1.7b", modeByteLevel},
	}
	prompts := []string{
		"Hello, world!", `{"name": "Ada", "age": 36, "tags": ["x", "y"]}`,
		"café 🦄 \t newline\n", "  spaced  ", "12345 -0.5e+10",
	}
	for _, c := range cases {
		if _, err := os.Stat(c.dir); errors.Is(err, fs.ErrNotExist) {
			t.Logf("skip %s (absent)", c.dir)
			continue
		}
		tk, err := Load(c.dir)
		if err != nil {
			t.Fatalf("Load(%s): %v", c.dir, err)
		}
		if tk.mode != c.mode {
			t.Fatalf("%s: mode %d, want %d", c.dir, tk.mode, c.mode)
		}
		if tk.stripLeadingSpace {
			t.Fatalf("%s strips leading space — TokenText reconstruction test assumes it does not", c.dir)
		}
		for _, p := range prompts {
			ids, err := tk.Encode(p, false)
			if err != nil {
				t.Fatalf("%s Encode(%q): %v", c.dir, p, err)
			}
			var got bytes.Buffer
			for _, id := range ids {
				got.Write(tk.TokenText(id))
			}
			want, err := tk.Decode(ids)
			if err != nil {
				t.Fatalf("%s Decode: %v", c.dir, err)
			}
			if got.String() != want {
				t.Errorf("%s: TokenText concat %q != Decode %q (prompt %q)", c.dir, got.String(), want, p)
			}
		}
	}
}
