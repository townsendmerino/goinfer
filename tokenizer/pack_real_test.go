//go:build realckpt

package tokenizer

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// TestPackReal_embeddingGemma2: the pack of google/embeddinggemma-2's tokenizer.json (GOINFER_EG2_DIR, default
// ~/models/embeddinggemma-2) loads to a tokenizer that gives the reference's input ids for the 48 golden texts and the
// same ids as the tokenizer.json on a seeded random sample. It logs the pack's size and the load times of both forms.
func TestPackReal_embeddingGemma2(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Skipf("no checkpoint tokenizer: %v", err)
	}
	t0 := time.Now()
	fromJSON, err := LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	jsonLoad := time.Since(t0)
	t1 := time.Now()
	pack, err := BuildPack(raw)
	if err != nil {
		t.Fatal(err)
	}
	build := time.Since(t1)
	var best time.Duration
	var fromPack *Tokenizer
	for i := range 5 {
		t2 := time.Now()
		tk, err := LoadPack(pack)
		if err != nil {
			t.Fatal(err)
		}
		if d := time.Since(t2); i == 0 || d < best {
			best = d
		}
		fromPack = tk
	}
	t.Logf("tokenizer.json %d bytes, loaded in %v; pack %d bytes, built in %v, loaded in %v (best of 5)",
		len(raw), jsonLoad, len(pack), build, best)

	golden := os.Getenv("GOINFER_EG2_GOLDEN")
	if golden == "" {
		golden = "../testdata/embeddinggemma2-real/golden.json"
	}
	var g struct {
		Items []struct {
			Prompt string `json:"prompt"`
			Text   string `json:"text"`
			IDs    []int  `json:"ids"`
		} `json:"items"`
	}
	graw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(graw, &g); err != nil {
		t.Fatal(err)
	}
	var st struct {
		Prompts map[string]string `json:"prompts"`
	}
	if sraw, err := os.ReadFile(filepath.Join(dir, "config_sentence_transformers.json")); err == nil {
		_ = json.Unmarshal(sraw, &st)
	}
	for i, it := range g.Items {
		ids, err := fromPack.Encode(st.Prompts[it.Prompt]+it.Text, false)
		if err != nil {
			t.Fatal(err)
		}
		want := it.IDs[1 : len(it.IDs)-1] // the reference wraps <bos> … <eos>
		if !reflect.DeepEqual(ids, want) {
			t.Errorf("golden item %d (prompt %q): pack ids differ from the reference", i, it.Prompt)
		}
	}

	rng := rand.New(rand.NewSource(7))
	alphabet := []string{"a", "e", "the ", "queue", " ", "  ", "\n", "\n\n", "\t", "é", "ß", "日本語", "한국어", "🙂", "<eos>",
		"<bos>", "<|turn>", "func ", "{}", "()", "0", "42", "...", "--", "Ω", "\u00a0", "\u200b", "ﬁ", "<0x41>", "Ż"}
	n := 0
	for range 5000 {
		var b strings.Builder
		for range rng.Intn(60) {
			if rng.Intn(5) == 0 {
				r := rune(rng.Intn(0x2FFFF))
				if utf8.ValidRune(r) {
					b.WriteRune(r)
				}
				continue
			}
			b.WriteString(alphabet[rng.Intn(len(alphabet))])
		}
		s := b.String()
		for _, special := range []bool{true, false} {
			a, errA := fromJSON.encode(s, false, special)
			p, errP := fromPack.encode(s, false, special)
			if errA != nil || errP != nil || !reflect.DeepEqual(a, p) {
				t.Fatalf("sample %q special=%v: json %v %v, pack %v %v", s, special, a, errA, p, errP)
			}
			n++
		}
	}
	t.Logf("%d golden texts and %d sample encodings identical", len(g.Items), n)
}
