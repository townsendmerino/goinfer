package tokenizer

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// GGUF tokenizer parity (G7 follow-up). The bare-GGUF loader must reproduce the
// HF ids id-for-id, and agree with the tokenizer.json loader on the same model.
// Both assets are TinyLlama (Llama-2 SentencePiece, tokenizer.ggml.model ==
// "llama"); skips cleanly when they're absent so a fresh checkout stays green.
//
// Regenerate the golden:
//
//	hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 tokenizer.json tokenizer_config.json --local-dir testdata/tinyllama-1.1b
//	.venv/bin/python scripts/pin_tinyllama_tokenizer.py

const (
	tinyllamaGGUF   = "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	tinyllamaGolden = "../testdata/tinyllama_tokenizer_golden.json"
	// Byte-level (gpt2-family) GGUF + the same model's committed tokenizer.json.
	llama32GGUF = "../testdata/llama32-gguf/llama-3.2-1b-instruct-Q4_K_M.gguf"
	llama32JSON = "../testdata/llama3.2-1b"
	mellumGGUF  = "../testdata/mellum-gguf/Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf"
)

// TestLoadGGUF_mellum2DigitParity pins the GGUF tokenizer path for Mellum2:
// tokenizer.ggml.pre == "mellum2" must select the same byte-level knobs
// (splitDigits on) that Load reads from the tokenizer.json, so a bare GGUF
// reproduces HF's digit-isolating tokenization id-for-id. Validated against the
// committed mellum2 golden (HF oracle). Loads only GGUF metadata (mmap), not the
// ~8 GB weights; skips when the GGUF is absent.
func TestLoadGGUF_mellum2DigitParity(t *testing.T) {
	g := loadGoldenAt(t, "../testdata/mellum2_tokenizer_golden.json")
	if _, err := os.Stat(mellumGGUF); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mellum2 GGUF at %s", mellumGGUF)
	}
	gg, err := LoadGGUF(mellumGGUF)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	if gg.mode != modeByteLevel || !gg.splitDigits {
		t.Fatalf("GGUF knobs: mode=%d splitDigits=%v, want byteLevel + splitDigits", gg.mode, gg.splitDigits)
	}
	for _, c := range g.Cases {
		t.Run(caseName(c.Text), func(t *testing.T) {
			got, err := gg.Encode(c.Text, false)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			if !equalInts(got, c.IDs) {
				t.Errorf("GGUF Encode(%q)\n  got  %v\n  want %v\n  toks %v", c.Text, got, c.IDs, c.Tokens)
			}
		})
	}
}

// TestLoadGGUF_tinyllamaParity: the tokenizer built from GGUF metadata alone
// (vocab + merges + special ids) must match the HF oracle id-for-id, with the
// ▁ dummy prefix on encode and the single-leading-space strip on decode.
func TestLoadGGUF_tinyllamaParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: reads a ~1 GB GGUF")
	}
	g := loadGoldenAt(t, tinyllamaGolden)
	if _, err := os.Stat(tinyllamaGGUF); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF at %s", tinyllamaGGUF)
	}

	tk, err := LoadGGUF(tinyllamaGGUF)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	if tk.mode != modeGemma {
		t.Fatalf("resolved mode %d, want modeGemma (byte-fallback)", tk.mode)
	}
	if sp := tk.Special(); sp.BOS != g.SpecialTokens.BOS || sp.EOS != g.SpecialTokens.EOS {
		t.Errorf("special tokens: got BOS=%d EOS=%d, want BOS=%d EOS=%d",
			sp.BOS, sp.EOS, g.SpecialTokens.BOS, g.SpecialTokens.EOS)
	}

	checkGoldenCases(t, tk, g)
}

// TestLoadGGUF_byteLevelMatchesJSON: the byte-level (gpt2-family) GGUF tokenizer
// must reproduce the same ids as Load on the same model's tokenizer.json — the
// GGUF metadata (tokens + merges + tokenizer.ggml.pre) and the tokenizer.json
// carry the same byte-level vocab/merges/knobs, so the two extraction paths
// agree id-for-id. The json path is itself HF-golden-validated for this family
// (llama3_tokenizer_golden.json), so agreement pins the GGUF path to HF. Uses a
// real Llama-3.2-1B-Instruct GGUF; skips when absent.
func TestLoadGGUF_byteLevelMatchesJSON(t *testing.T) {
	if _, err := os.Stat(llama32GGUF); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no byte-level GGUF at %s", llama32GGUF)
	}
	if _, err := os.Stat(llama32JSON); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tokenizer.json model at %s", llama32JSON)
	}

	gg, err := LoadGGUF(llama32GGUF)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	jj, err := Load(llama32JSON)
	if err != nil {
		t.Fatalf("Load(json): %v", err)
	}

	// Structural: the GGUF pre→knob mapping must match what Load derived from the
	// tokenizer.json (byte-level, digit-run 3, no NFC, ignore_merges for Llama-3).
	if gg.mode != modeByteLevel {
		t.Fatalf("GGUF mode = %d, want modeByteLevel", gg.mode)
	}
	if gg.maxDigits != jj.maxDigits || gg.normOn != jj.normOn || gg.ignoreMerges != jj.ignoreMerges {
		t.Errorf("knobs differ: gguf{digits=%d nfc=%v ignoreMerges=%v} json{digits=%d nfc=%v ignoreMerges=%v}",
			gg.maxDigits, gg.normOn, gg.ignoreMerges, jj.maxDigits, jj.normOn, jj.ignoreMerges)
	}
	if gg.special.BOS != jj.special.BOS {
		t.Errorf("BOS: gguf %d, json %d", gg.special.BOS, jj.special.BOS)
	}

	prompts := []string{
		"Hello world", " Hello", "  two  spaces", "trailing   ",
		"The quick brown fox jumps over the lazy dog.",
		"café 中文 — naïve façade", "𝕳ello", "emoji 🦄 and 🏳️‍🌈",
		"a\tb\nc\n\nd", "don't can't I'LL we've",
		"func main() { fmt.Println(\"hi\") }",
		"Number 1234567 and 56 and 8", "year 2024, pi 3.14159, 1000000",
		"<|begin_of_text|>hi<|eot_id|>", "<|begin_of_text|>", "",
	}
	for _, p := range prompts {
		t.Run(caseName(p), func(t *testing.T) {
			for _, bos := range []bool{false, true} {
				gi, err := gg.Encode(p, bos)
				if err != nil {
					t.Fatalf("gguf Encode: %v", err)
				}
				ji, jerr := jj.Encode(p, bos)
				if jerr != nil {
					t.Fatalf("json Encode: %v", jerr)
				}
				if !equalInts(gi, ji) {
					t.Errorf("Encode(%q, bos=%v) disagree\n  gguf %v\n  json %v", p, bos, gi, ji)
				}
			}
			// Decode round-trips identically too.
			ids, _ := gg.Encode(p, false)
			gd, _ := gg.Decode(ids)
			jd, _ := jj.Decode(ids)
			if gd != jd {
				t.Errorf("Decode(%q) disagree: gguf %q json %q", p, gd, jd)
			}
		})
	}
}

// checkGoldenCases asserts a loaded tokenizer matches every golden case:
// Encode without/with BOS, and Decode back to HF's rendering. (Mirrors the
// byte-level/Gemma parity runners; lives here to validate either loader.)
func checkGoldenCases(t *testing.T, tk *Tokenizer, g tokGolden) {
	t.Helper()
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

// loadGoldenAt reads a tokGolden from path, skipping when absent.
func loadGoldenAt(t *testing.T, path string) tokGolden {
	t.Helper()
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — regenerate with scripts/pin_tinyllama_tokenizer.py", path)
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
