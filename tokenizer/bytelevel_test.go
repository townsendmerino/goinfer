package tokenizer

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// G3 byte-level BPE parity (Qwen + Llama-3). Same per-machine-asset convention
// as the M2 Gemma test: SKIPS cleanly when the tokenizer or golden is absent so
// a fresh checkout stays green.
//
// Get the tokenizers + regenerate the goldens:
//
//	hf download Qwen/Qwen3-1.7B --local-dir testdata/qwen3-1.7b
//	.venv/bin/python scripts/pin_qwen3_tokenizer.py
//	hf download NousResearch/Meta-Llama-3-8B tokenizer.json tokenizer_config.json --local-dir testdata/llama3-tokenizer
//	.venv/bin/python scripts/pin_llama3_tokenizer.py

// TestByteLevel_qwen3GoldenParity: Qwen normalizes NFC, takes one digit per
// token, and adds no BOS (ids_bos == ids).
func TestByteLevel_qwen3GoldenParity(t *testing.T) {
	runByteLevelParity(t, "../testdata/qwen3-1.7b", "../testdata/qwen3_tokenizer_golden.json")
}

// TestByteLevel_llama3GoldenParity: Llama-3 has no normalizer, groups digits in
// runs of ≤3, and prepends <|begin_of_text|> (ids_bos has the extra BOS) — the
// same byte-level core, knobs read from tokenizer.json.
func TestByteLevel_llama3GoldenParity(t *testing.T) {
	runByteLevelParity(t, "../testdata/llama3-tokenizer", "../testdata/llama3_tokenizer_golden.json")
}

// TestByteLevel_mellum2GoldenParity: Mellum2's pre_tokenizer is
// Sequence[Digits{individual_digits}, ByteLevel] with no normalizer, so each
// digit is isolated before the GPT-2 split — a leading space never attaches to a
// digit (" 1" → "Ġ" + "1"). HF adds no special tokens at encode (post_processor
// is plain ByteLevel; bos_token is defined but not auto-prepended), so the golden
// has ids_bos == ids and we assert the HF-faithful no-special `ids` column plus
// the decode round-trip. (The with-BOS column is omitted: whether to prepend a
// defined-but-not-auto bos_token is a caller choice orthogonal to this fix.)
func TestByteLevel_mellum2GoldenParity(t *testing.T) {
	g := loadGoldenAt(t, "../testdata/mellum2_tokenizer_golden.json")
	const modelDir = "../testdata/mellum2-tokenizer"
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tokenizer at %s", modelDir)
	}
	tk, err := Load(modelDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tk.mode != modeByteLevel {
		t.Fatalf("resolved mode %d, want modeByteLevel", tk.mode)
	}
	if !tk.splitDigits {
		t.Fatalf("splitDigits not detected for Mellum2 (Digits{individual_digits} pretokenizer)")
	}
	for _, c := range g.Cases {
		t.Run(caseName(c.Text), func(t *testing.T) {
			got, err := tk.Encode(c.Text, false)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			if !equalInts(got, c.IDs) {
				t.Errorf("Encode(%q)\n  got  %v\n  want %v\n  toks %v", c.Text, got, c.IDs, c.Tokens)
			}
			dec, err := tk.Decode(c.IDs)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if dec != c.Decode {
				t.Errorf("Decode(%v) = %q, want %q", c.IDs, dec, c.Decode)
			}
		})
	}
}

// runByteLevelParity asserts every golden prompt encodes id-for-id without
// special tokens (ids) and with (ids_bos), and decodes back to HF's rendering.
// A single drift silently degrades generation, so the bar is exact equality.
func runByteLevelParity(t *testing.T, modelDir, goldenPath string) {
	t.Helper()
	raw, err := os.ReadFile(goldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with the matching scripts/pin_*_tokenizer.py", goldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g tokGolden // same shape as the Gemma golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tokenizer at %s", modelDir)
	}

	tk, err := Load(modelDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tk.mode != modeByteLevel {
		t.Fatalf("resolved mode %d, want modeByteLevel", tk.mode)
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
		})
	}
}
