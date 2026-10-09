package tokenizer

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// G-S14e1 (docs/tasks/task-multimodal-support-2026-10.md): the Tekken tokenizer against mistral_common 1.12.0's own ids, pinned by scripts/pin_tekken.py into
// testdata/tekken_golden.json.gz. The tekken.json (14.9 MB) is not committed: it is read from the Voxtral Mini directory under GOINFER_MODELS_DIR (default ~/models) and the
// test is skipped without it (and refuses a file whose sha256 is not the golden's).

type tekkenGolden struct {
	SHA     string `json:"tekken_sha256"`
	NVocab  int    `json:"n_vocab"`
	BOS     int    `json:"bos"`
	EOS     int    `json:"eos"`
	Strings []struct {
		Text    string `json:"text"`
		IDs     []int  `json:"ids"`
		Decoded string `json:"decoded"`
	} `json:"strings"`
	Requests []struct {
		Seconds  float64 `json:"seconds"`
		Language *string `json:"language"`
		IDs      []int   `json:"ids"`
	} `json:"requests"`
}

func loadTekkenGolden(t *testing.T) (*tekkenGolden, []byte) {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "testdata", "tekken_golden.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g tekkenGolden
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(modelPath("voxtral-mini-3b-2507"), "tekken.json"))
	if err != nil {
		t.Skipf("no tekken.json under %s: %v", modelPath("voxtral-mini-3b-2507"), err)
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != g.SHA {
		t.Fatalf("tekken.json sha256 %s is not the golden's %s", got, g.SHA)
	}
	return &g, raw
}

// plainIDs encodes text as ordinary text: a special-token spelling in it is NOT parsed (mistral_common does not parse them either).
func plainIDs(t *Tokenizer, s string) ([]int, error) {
	return t.EncodeSegments([]Segment{{Text: s, Special: false}}, false)
}

// stringMismatches counts how many golden strings tok encodes to different ids, and the first one's index.
func stringMismatches(t *testing.T, tok *Tokenizer, g *tekkenGolden, encode func(*Tokenizer, string) ([]int, error)) (n, first int) {
	t.Helper()
	first = -1
	for i, s := range g.Strings {
		got, err := encode(tok, s.Text)
		if err != nil {
			t.Fatalf("string %d: %v", i, err)
		}
		if !equalInts(got, s.IDs) {
			n++
			if first < 0 {
				first = i
			}
		}
	}
	return n, first
}

func TestTekken_matchesMistralCommon(t *testing.T) {
	g, raw := loadTekkenGolden(t)
	tok, err := LoadTekken(raw)
	if err != nil {
		t.Fatalf("LoadTekken: %v", err)
	}
	if tok.preShape != shapeTekken {
		t.Fatalf("the pre-tokenizer was classified %q, want tekken", tok.preShape)
	}
	if tok.special.BOS != g.BOS || tok.special.EOS != g.EOS {
		t.Errorf("BOS/EOS = %d/%d, want %d/%d", tok.special.BOS, tok.special.EOS, g.BOS, g.EOS)
	}
	for i, s := range g.Strings {
		got, err := plainIDs(tok, s.Text)
		if err != nil {
			t.Fatalf("string %d %q: %v", i, s.Text, err)
		}
		if !equalInts(got, s.IDs) {
			t.Errorf("string %d %.60q: ids differ\n got %v\nwant %v", i, s.Text, head(got), head(s.IDs))
			continue
		}
		for _, id := range got {
			if id < 0 || id >= g.NVocab {
				t.Errorf("string %d: id %d is outside the model's %d-id vocabulary", i, id, g.NVocab)
			}
		}
		back, err := tok.Decode(got)
		if err != nil {
			t.Fatalf("string %d: decode: %v", i, err)
		}
		if back != s.Text || s.Decoded != s.Text {
			t.Errorf("string %d %.60q: decode(encode) = %.60q (mistral_common: %.60q)", i, s.Text, back, s.Decoded)
		}
	}
	if len(g.Strings) < 400 {
		t.Fatalf("golden has only %d strings", len(g.Strings))
	}
}

func head(a []int) []int {
	if len(a) > 24 {
		return a[:24]
	}
	return a
}

// TestTekken_plantedDefects: each plausible mistake in the build must put some string over the bar, or the gate is blind to it (G-S14e1's registered list; the fifth, "the o200k
// pattern's walker", replaces the registered "marks dropped" one, see the amendment in the task doc).
func TestTekken_plantedDefects(t *testing.T) {
	g, raw := loadTekkenGolden(t)
	zero, offByHundred := 0, 100
	cases := []struct {
		name   string
		build  func() (*Tokenizer, error)
		encode func(*Tokenizer, string) ([]int, error)
		min    int // at least this many of the golden strings must differ
	}{
		{"no rank offset (ids from 0 instead of 1000)", func() (*Tokenizer, error) { return loadTekken(raw, tekkenOptions{rankOffsetOverride: &zero}) }, plainIDs, 50},
		{"rank offset 100", func() (*Tokenizer, error) { return loadTekken(raw, tekkenOptions{rankOffsetOverride: &offByHundred}) }, plainIDs, 50},
		{"vocabulary not truncated at 130,072", func() (*Tokenizer, error) { return loadTekken(raw, tekkenOptions{noTruncate: true}) }, plainIDs, 1},
		{"merges ordered by length, not rank (a longest-match flavour)", func() (*Tokenizer, error) { return loadTekken(raw, tekkenOptions{sortBySize: true}) }, plainIDs, 1},
		{"special-token spellings parsed out of plain text", func() (*Tokenizer, error) { return LoadTekken(raw) }, func(tk *Tokenizer, s string) ([]int, error) { return tk.Encode(s, false) }, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, err := c.build()
			if err != nil {
				t.Fatal(err)
			}
			n, first := stringMismatches(t, tok, g, c.encode)
			t.Logf("%d of %d strings differ (first: %d)", n, len(g.Strings), first)
			if n < c.min {
				t.Errorf("the planted defect %q is invisible to the golden: %d strings differ, want at least %d", c.name, n, c.min)
			}
		})
	}
	t.Run("the o200k walker (attached contractions, three-digit runs) instead of Tekken's", func(t *testing.T) {
		tok, err := LoadTekken(raw)
		if err != nil {
			t.Fatal(err)
		}
		tok.preShape = shapeO200k
		n, first := stringMismatches(t, tok, g, plainIDs)
		t.Logf("%d of %d strings differ (first: %d)", n, len(g.Strings), first)
		if n < 1 {
			t.Error("the o200k walker reads the same on every golden string: the golden cannot tell the two patterns apart")
		}
	})
}

func TestTekken_splitMatchesPattern(t *testing.T) {
	// Tekken's own alternation, spelled out: contractions do NOT attach and a digit is one pre-token.
	cases := []struct {
		in   string
		want []string
	}{
		{"don't", []string{"don", "'t"}},
		{"2026", []string{"2", "0", "2", "6"}},
		{"CamelCase", []string{"Camel", "Case"}},
		{"a/b", []string{"a", "/b"}},
		{"x  y", []string{"x", " ", " y"}},
	}
	for _, c := range cases {
		got := splitTekken(c.in)
		if len(got) != len(c.want) {
			t.Errorf("splitTekken(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("splitTekken(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}
