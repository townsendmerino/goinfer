package tokenizer

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// M2 tokenizer parity. Assets live under testdata/ and are per-machine (the
// 270M checkpoint ships tokenizer.json); the test SKIPS cleanly when they're
// absent, so a fresh checkout stays green — the same convention encoder/ and
// the M1 loader use.
//
// Get the checkpoint + regenerate the golden:
//
//	huggingface-cli download google/gemma-3-270m --local-dir testdata/gemma-3-270m
//	.venv/bin/python scripts/pin_gemma_tokenizer.py
const (
	gemmaModelDir      = "../testdata/gemma-3-270m"
	gemmaTokGoldenPath = "../testdata/gemma_tokenizer_golden.json"
)

type tokGolden struct {
	SpecialTokens struct {
		BOS         int `json:"bos"`
		EOS         int `json:"eos"`
		Pad         int `json:"pad"`
		Unk         int `json:"unk"`
		StartOfTurn int `json:"start_of_turn"`
		EndOfTurn   int `json:"end_of_turn"`
	} `json:"special_tokens"`
	Cases []struct {
		Text      string   `json:"text"`
		IDsBOS    []int    `json:"ids_bos"`
		IDs       []int    `json:"ids"`
		Tokens    []string `json:"tokens"`
		DecodeBOS string   `json:"decode_bos"`
		Decode    string   `json:"decode"`
	} `json:"cases"`
}

func loadTokGolden(t *testing.T) tokGolden {
	t.Helper()
	raw, err := os.ReadFile(gemmaTokGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with scripts/pin_gemma_tokenizer.py", gemmaTokGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g tokGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}

// TestEncodeDecode_goldenParity loads the real tokenizer and asserts that
// every golden prompt encodes id-for-id (with and without BOS) and decodes
// back to HF's rendering. A single drift here silently degrades generation,
// so the bar is exact equality, not a tolerance.
func TestEncodeDecode_goldenParity(t *testing.T) {
	g := loadTokGolden(t) // skips if golden absent
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — huggingface-cli download google/gemma-3-270m --local-dir %s",
			gemmaModelDir, gemmaModelDir)
	}

	tk, err := Load(gemmaModelDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	sp := tk.Special()
	if sp.BOS != g.SpecialTokens.BOS || sp.EOS != g.SpecialTokens.EOS ||
		sp.Pad != g.SpecialTokens.Pad || sp.StartOfTurn != g.SpecialTokens.StartOfTurn ||
		sp.EndOfTurn != g.SpecialTokens.EndOfTurn {
		t.Errorf("Special() = %+v, want bos=%d eos=%d pad=%d sot=%d eot=%d", sp,
			g.SpecialTokens.BOS, g.SpecialTokens.EOS, g.SpecialTokens.Pad,
			g.SpecialTokens.StartOfTurn, g.SpecialTokens.EndOfTurn)
	}

	for _, c := range g.Cases {
		t.Run(caseName(c.Text), func(t *testing.T) {
			got, err := tk.Encode(c.Text, false)
			if err != nil {
				t.Fatalf("Encode(addBOS=false): %v", err)
			}
			if !equalInts(got, c.IDs) {
				t.Errorf("Encode(%q, false)\n  got  %v\n  want %v\n  toks %v", c.Text, got, c.IDs, c.Tokens)
			}

			gotBOS, err := tk.Encode(c.Text, true)
			if err != nil {
				t.Fatalf("Encode(addBOS=true): %v", err)
			}
			if !equalInts(gotBOS, c.IDsBOS) {
				t.Errorf("Encode(%q, true)\n  got  %v\n  want %v", c.Text, gotBOS, c.IDsBOS)
			}

			dec, err := tk.Decode(c.IDs)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if dec != c.Decode {
				t.Errorf("Decode(%v) = %q, want %q", c.IDs, dec, c.Decode)
			}

			decBOS, err := tk.Decode(c.IDsBOS)
			if err != nil {
				t.Fatalf("Decode(bos): %v", err)
			}
			if decBOS != c.DecodeBOS {
				t.Errorf("Decode(%v) = %q, want %q", c.IDsBOS, decBOS, c.DecodeBOS)
			}
		})
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// caseName makes a readable, unique-enough subtest name from a prompt.
func caseName(s string) string {
	if s == "" {
		return "empty"
	}
	const max = 24
	out := make([]rune, 0, max)
	for _, r := range s {
		if len(out) >= max {
			break
		}
		if r == ' ' {
			r = '_'
		}
		if r < 0x20 {
			r = '.'
		}
		out = append(out, r)
	}
	return string(out)
}
