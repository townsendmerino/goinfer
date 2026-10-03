package clef

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// clefTokenizer finds Clef-flash's tokenizer.json (20 MB, not in the tree): $CLEF_TOKENIZER (a directory), or ~/models/clef-flash. Under
// GOINFER_HEAVY_TESTS=1 a missing tokenizer FAILS; otherwise it skips, and a skip is not a pass.
func clefTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	home, _ := os.UserHomeDir()
	for _, d := range []string{os.Getenv("CLEF_TOKENIZER"), filepath.Join(home, "models", "clef-flash")} {
		if d == "" {
			continue
		}
		p := filepath.Join(d, "tokenizer.json")
		if _, err := os.Stat(p); err == nil {
			tk, err := tokenizer.Load(p)
			if err != nil {
				t.Fatal(err)
			}
			return tk
		}
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") == "1" {
		t.Fatal("GOINFER_HEAVY_TESTS=1 but no Clef-flash tokenizer.json (set CLEF_TOKENIZER, or put it under ~/models/clef-flash)")
	}
	t.Skip("no Clef-flash tokenizer.json (set CLEF_TOKENIZER, or put it under ~/models/clef-flash)")
	return nil
}

// The tokenizer half of the gate: with goinfer's own tokenizer in place of the replay, the 150 items' token ids and spans must still be
// the reference's. Each fragment is tokenized separately and with special-token text PARSED (Encode, not EncodeSegments), as the
// reference's tokenizer(text, add_special_tokens=False) does.
func TestEncode_realTokenizer(t *testing.T) {
	tk := clefTokenizer(t)
	dumps, recs := loadDump(t)
	tok := func(text string) ([]int, error) { return tk.Encode(text, false) }
	bad := 0
	for _, r := range recs {
		d := dumps[r.ID]
		got, err := Encode(r.Request, tok, 0)
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		same := len(got.InputIDs) == len(d.InputIDs)
		for i := 0; same && i < len(d.InputIDs); i++ {
			same = got.InputIDs[i] == d.InputIDs[i]
		}
		if !same {
			bad++
			t.Errorf("%s: token ids differ from the reference (%d vs %d tokens)", r.ID, len(got.InputIDs), len(d.InputIDs))
			continue
		}
		for i, w := range d.Questions {
			g := got.Questions[i]
			if g.QuestionSpan != w.Span || len(g.OptionSpans) != len(w.OptionSpans) {
				bad++
				t.Errorf("%s question %d: spans differ", r.ID, i)
			}
		}
	}
	t.Logf("150 items through goinfer's tokenizer: %d differ from the reference", bad)
}
