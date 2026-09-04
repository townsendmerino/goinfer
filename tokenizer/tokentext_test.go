package tokenizer

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"testing"
	"unicode/utf8"
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
	// G-10: EXERCISED cases are counted, and zero of them is a SKIP rather than a PASS. Every
	// case is fixture-gated, so on a checkout without them this loop did nothing at all and the
	// test still printed PASS — a green that proves the reconstruction property holds nowhere.
	exercised := 0
	for _, c := range cases {
		if _, err := os.Stat(c.dir); errors.Is(err, fs.ErrNotExist) {
			t.Logf("skip %s (absent)", c.dir)
			continue
		}
		exercised++
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
	if exercised == 0 {
		t.Skip("no tokenizer fixture present — TokenText reconstruction was checked on nothing; " +
			"this is a SKIP, not a pass (G-10)")
	}
	t.Logf("TokenText reconstruction verified on %d tokenizer(s)", exercised)
}

// TestTokenText_addedTokenIsVerbatimInByteLevelMode pins V-13 (docs/review-2026-09-04.md):
// TokenText's byte-level branch pushed EVERY id through byteDecoder, including added/special
// tokens, whose surface is stored VERBATIM rather than byte-level-encoded — the exact category
// error N-24 fixed in decodeByteLevel (tokenizer/bytelevel.go), left unfixed here. A rune in
// U+0080–U+0143 in an added token's text (é, ü, ñ — any chat template spelling a role in a
// non-ASCII language) is itself one of the byte-level table's "printable" targets, so pushing it
// through byteDecoder maps it back to a SINGLE raw byte instead of its real multi-byte UTF-8
// encoding: invalid UTF-8, and a wrong surface for the constrained-decoding mask table this
// function feeds. No tokenizer fixture needed — the Tokenizer is built directly, byteDecoder from
// the same buildByteLevelTables the real byte-level tokenizers use.
func TestTokenText_addedTokenIsVerbatimInByteLevelMode(t *testing.T) {
	_, dec := buildByteLevelTables()
	const added = "café" // added token surface, stored verbatim — 'é' is U+00E9, in the printable byte-table range
	tk := &Tokenizer{
		mode:        modeByteLevel,
		idToPiece:   []string{added, "hi"}, // id 1: an ordinary byte-level piece (ASCII, byte-encoded == itself)
		isAdded:     []bool{true, false},
		byteDecoder: dec,
	}
	got := tk.TokenText(0)
	if string(got) != added {
		t.Errorf("TokenText(added token %q) = %q (% x), want the verbatim surface unchanged (V-13)",
			added, got, got)
	}
	if !utf8.Valid(got) {
		t.Errorf("TokenText(added token) produced invalid UTF-8: % x", got)
	}
	// Regression guard: an ordinary (non-added) byte-level piece must still decode through the
	// byte table, or this fix would have just special-cased the wrong condition.
	if got := tk.TokenText(1); string(got) != "hi" {
		t.Errorf("TokenText(ordinary byte-level token) = %q, want %q", got, "hi")
	}
}
