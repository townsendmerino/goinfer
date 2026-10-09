package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
)

// LoadVocabMerges (S14.3 of docs/tasks/task-multimodal-support-2026-10.md): a Qwen2-class tokenizer assembled from vocab.json + merges.txt + tokenizer_config.json must encode and decode
// exactly like the tokenizer.json the same three files make. The committed Qwen2.5 tokenizer.json is taken apart into those three files, reassembled, and compared on a corpus built to
// hit the pipeline's parts: the split regex's alternatives (contractions, letters, single digits, punctuation runs, newlines, trailing whitespace), non-ASCII, and the added tokens.

var vocabMergesCorpus = []string{
	"", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "I'm sure they'll say we've seen it; she'd go, it's fine.",
	"def f(x):\n    return x * 2  # double\n\n\nprint(f(21))\n", "2026-10-08 12:34:56, 3.14159, 1e-9, 0xFF, 1,000,000", "naïve café — “quotes” … 日本語のテキスト 한국어 😀🙂 العربية",
	"line one\r\nline two\r\n\r\n   indented\ttab   \n\n", "trailing spaces    ", "    leading spaces", "a b c", "URL https://example.com/a?b=c&d=e#frag",
	"<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "x<|endoftext|>y", "language English<asr_text>Mr. Quilter is the apostle of the middle classes.",
	strings.Repeat("lorem ipsum dolor sit amet, ", 200),
}

// splitTokenizerJSON writes tokenizerJSON's vocab, merges and added tokens as the three files a Qwen2Tokenizer repo ships.
func splitTokenizerJSON(t *testing.T, srcJSON, dir, class string) {
	t.Helper()
	raw, err := os.ReadFile(srcJSON)
	if err != nil {
		t.Skipf("no %s: %v", srcJSON, err)
	}
	var tj struct {
		Model struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		}
		AddedTokens []struct {
			ID                                              int    `json:"id"`
			Content                                         string `json:"content"`
			LStrip, RStrip, Normalized, SingleWord, Special bool
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		t.Fatal(err)
	}
	var lines []string
	var pairs [][]string
	if json.Unmarshal(tj.Model.Merges, &pairs) == nil && len(pairs) > 0 && len(pairs[0]) == 2 {
		for _, p := range pairs {
			lines = append(lines, p[0]+" "+p[1])
		}
	} else if err := json.Unmarshal(tj.Model.Merges, &lines); err != nil {
		t.Fatal(err)
	}
	must := func(name string, v any) {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	must("vocab.json", tj.Model.Vocab)
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("#version: 0.2\n"+strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	dec := map[string]any{}
	for _, a := range tj.AddedTokens {
		dec[fmt.Sprint(a.ID)] = map[string]any{"content": a.Content, "lstrip": a.LStrip, "rstrip": a.RStrip, "normalized": a.Normalized, "single_word": a.SingleWord, "special": a.Special}
	}
	must("tokenizer_config.json", map[string]any{"tokenizer_class": class, "added_tokens_decoder": dec})
}

func compareTokenizers(t *testing.T, want, got *Tokenizer) {
	t.Helper()
	// goinfer's byte-level walker classifies the declared split regex into a shape it implements; a regex it does not recognise is recorded as a decline and walked as the nearest shape, which
	// leaves the ids unchanged on easy inputs. So the declared regex is held to the original's classification, not only the ids.
	if wd, gd := want.PreTokenizerDecline(), got.PreTokenizerDecline(); wd != gd || gd != "" {
		t.Errorf("pre-tokenizer decline: tokenizer.json %q, vocab+merges %q (both must be empty: the Qwen2 split regex is a shape the walker implements)", wd, gd)
	}
	for _, s := range vocabMergesCorpus {
		w, err := want.Encode(s, false)
		if err != nil {
			t.Fatal(err)
		}
		g, err := got.Encode(s, false)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(w, g) {
			t.Errorf("encode(%.40q): tokenizer.json gives %d ids, vocab+merges gives %d (first difference at %d)", s, len(w), len(g), firstDiff(w, g))
			continue
		}
		wd, _ := want.Decode(w)
		gd, _ := got.Decode(g)
		if wd != gd {
			t.Errorf("decode(%.40q): %q against %q", s, wd, gd)
		}
	}
}

func firstDiff(a, b []int) int {
	for i := 0; i < min(len(a), len(b)); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

func TestLoadVocabMerges_matchesTokenizerJSON(t *testing.T) {
	src := filepath.Join("..", "testdata", "qwen2.5-0.5b", "tokenizer.json")
	// The regex constant is the class's identity and the walker only classifies it loosely, so hold the TEXT to a real artifact: it must be the one Qwen2.5's own tokenizer.json declares.
	if raw, err := os.ReadFile(src); err == nil {
		var tj struct {
			Pre struct {
				Pretokenizers []struct {
					Pattern struct{ Regex string }
				}
			} `json:"pre_tokenizer"`
		}
		if json.Unmarshal(raw, &tj) != nil || len(tj.Pre.Pretokenizers) == 0 || tj.Pre.Pretokenizers[0].Pattern.Regex != qwen2SplitRegex {
			t.Errorf("qwen2SplitRegex is not the pattern Qwen2.5's tokenizer.json declares: %q", tj.Pre.Pretokenizers)
		}
	}
	dir := t.TempDir()
	splitTokenizerJSON(t, src, dir, "Qwen2Tokenizer")
	got, err := LoadVocabMerges(dir)
	if err != nil {
		t.Fatalf("LoadVocabMerges: %v", err)
	}
	want, err := Load(src)
	if err != nil {
		t.Fatal(err)
	}
	compareTokenizers(t, want, got)
	// the directory loader falls back to it when there is no tokenizer.json
	viaLoad, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(dir without tokenizer.json): %v", err)
	}
	compareTokenizers(t, want, viaLoad)
}

func TestLoadVocabMerges_refusesOtherClasses(t *testing.T) {
	src := filepath.Join("..", "testdata", "qwen2.5-0.5b", "tokenizer.json")
	for _, class := range []string{"GPT2Tokenizer", "LlamaTokenizer", ""} {
		dir := t.TempDir()
		splitTokenizerJSON(t, src, dir, class)
		if _, err := LoadVocabMerges(dir); !errors.Is(err, ErrNoVocabMerges) {
			t.Errorf("tokenizer_class %q: error %v, want ErrNoVocabMerges (the split regex is part of the class)", class, err)
		}
		if _, err := Load(dir); err == nil {
			t.Errorf("tokenizer_class %q: Load succeeded on a directory with no tokenizer.json and an unsupported class", class)
		}
	}
	if _, err := Load(t.TempDir()); err == nil {
		t.Error("Load succeeded on an empty directory")
	}
}

// The real Qwen3-ASR repo's three files against the tokenizer.json transformers builds from them (written beside them by scripts/pin_qwen3asr_real.py).
func TestLoadVocabMerges_realQwen3ASR(t *testing.T) {
	dir := os.Getenv("GOINFER_QWEN3ASR_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3-asr-0.6b")
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("no generated tokenizer.json to compare with in %s", dir)
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	want, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadVocabMerges(dir)
	if err != nil {
		t.Fatal(err)
	}
	compareTokenizers(t, want, got)
	// every id decodes the same
	ids := make([]int, 0, 2000)
	for i := 0; i < 151700; i += 77 {
		ids = append(ids, i)
	}
	sort.Ints(ids)
	wd, _ := want.Decode(ids)
	gd, _ := got.Decode(ids)
	if wd != gd {
		t.Error("a sweep of ids decodes differently")
	}
}
