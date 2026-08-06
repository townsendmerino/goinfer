package tokenizer

import (
	"os"
	"reflect"
	"testing"
)

// TestGemma3SPMScoresParity gates the SentencePiece unigram-scores encode path (gemma-3 GGUFs ship
// tokenizer.ggml.scores but no tokenizer.ggml.merges, so the BPE merge-rank path can't run). The
// `want` ids are HF AutoTokenizer(gemma-3-4b-it).encode(add_special_tokens=False) — regenerate with:
//
//	g4venv/bin/python -c 'from transformers import AutoTokenizer as A; \
//	  t=A.from_pretrained("~/models/gemma-3-4b-it"); print(t.encode("...", add_special_tokens=False))'
//
// Covers prose, code, mixed whitespace, unicode (accents / CJK / emoji) and a digit run — the segments
// where the greedy best-score merge order matters. Skips when the gitignored GGUF fixture is absent.
func TestGemma3SPMScoresParity(t *testing.T) {
	g := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(g); err != nil {
		t.Skipf("no gemma-3-4b GGUF at %s", g)
	}
	tk, err := LoadGGUF(g)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	if tk.scoreRank == nil {
		t.Fatal("gemma-3 GGUF loaded without scoreRank — the scores fallback did not engage")
	}
	cases := []struct {
		text string
		want []int
	}{
		{"The capital of France is Paris.", []int{818, 5279, 529, 7001, 563, 9079, 236761}},
		{"Hello, world!", []int{9259, 236764, 1902, 236888}},
		{"def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
			[]int{2063, 10779, 78113, 236769, 236749, 1473, 107, 140, 2060, 538, 768, 538, 655, 236743, 236778, 1663, 10779, 78113, 236769, 236749, 236772, 236770, 7064, 73368, 78113, 236769, 236749, 236772, 236778, 236768}},
		{"  leading spaces and\ttabs\nand newlines", []int{138, 26016, 9952, 532, 255968, 39218, 107, 624, 861, 8721}},
		{"Ünïcodé: café, naïve, 日本語, 🎉", []int{238194, 236749, 238527, 15111, 236859, 236787, 33443, 236764, 120362, 236764, 33375, 238582, 236764, 204906}},
		{"1234567890 numbers", []int{236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819, 236771, 4945}},
	}
	for _, c := range cases {
		got, err := tk.Encode(c.text, false)
		if err != nil {
			t.Fatalf("Encode(%q): %v", c.text, err)
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("MISMATCH %q\n got=%v\nwant=%v", c.text, got, c.want)
		}
		// roundtrip: decode must reproduce the input (byte-fallback + merges compose back)
		if dec, _ := tk.Decode(got); dec != c.text {
			t.Errorf("roundtrip %q → %q", c.text, dec)
		}
	}
}
