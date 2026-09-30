package decide

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPyReplaceInvalidUTF8 holds the replacement to CPython's bytes.decode("utf-8", errors="replace"), the want column
// printed by CPython 3.12 for each input: truncated sequences, stray continuation bytes, surrogates, overlongs, a code
// point past U+10FFFF, and a truncated sequence followed by a valid one.
func TestPyReplaceInvalidUTF8(t *testing.T) {
	for _, c := range [][2]string{
		{"616263", "abc"}, {"e4b8", "�"}, {"e4b8ad78", "中x"}, {"b8ad", "��"}, {"f09f98", "�"},
		{"f09f9880", "😀"}, {"61f09f62", "a�b"}, {"eda080", "���"}, {"e08080", "���"},
		{"c0af", "��"}, {"f4908080", "����"}, {"ff", "�"}, {"e4b8e4b8ad", "�中"},
		{"808080", "���"}, {"f09f98e4b8ad", "�中"}, {"c3", "�"},
	} {
		b, _ := hex.DecodeString(c[0])
		if got := pyReplaceInvalidUTF8(b); got != c[1] {
			t.Errorf("%s: got %q, want %q", c[0], got, c[1])
		}
	}
}

// jevTokenizer finds autotrust/JEV-9B's tokenizer.json: $JEV_TOKENIZER, ~/models/JEV-9B, or a tokenizer-only copy at
// ~/models/JEV-9B-tokenizer. It is 20 MB and not in the tree.
func jevTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	home, _ := os.UserHomeDir()
	for _, d := range []string{os.Getenv("JEV_TOKENIZER"), filepath.Join(home, "models", "JEV-9B"), filepath.Join(home, "models", "JEV-9B-tokenizer")} {
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
	t.Skip("no JEV-9B tokenizer.json (set JEV_TOKENIZER, or put it under ~/models/JEV-9B)")
	return nil
}

// TestTruncateState_matchesReference: Route B's state truncation against jev_core's, run by transformers with JEV-9B's
// own tokenizer (scripts/pin_decisions_truncation.py). Eight states: ASCII, common CJK, emoji and accented text cut
// mid-character, rare ideographs that tokenize to raw bytes, a seam with multi-byte text on both sides, and exactly
// 1024 and 1025 tokens. Each must give the reference's kept text byte for byte, and the whole bare-v1 prompt must
// encode to the reference's token ids. Two of the cases put U+FFFD in the kept text, so the replacement is exercised,
// but each leaves a single stray byte, where CPython's rule and Go's strings.ToValidUTF8 agree (checked: the swap passes
// here). TestPyReplaceInvalidUTF8 is the test that tells the two apart.
func TestTruncateState_matchesReference(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/decisions/truncation_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden: run scripts/pin_decisions_truncation.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases []struct {
			Name, State, Kept string
			StateTokens       int   `json:"state_tokens"`
			Truncated         bool  `json:"truncated"`
			PromptIDs         []int `json:"prompt_ids"`
			HasReplacement    bool  `json:"has_replacement"`
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	tok := NewPlainTokenizer(jevTokenizer(t))
	repl := 0
	for _, c := range g.Cases {
		ids, err := tok.EncodePlain(c.State)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != c.StateTokens {
			t.Errorf("%s: the state is %d tokens here, %d in transformers", c.Name, len(ids), c.StateTokens)
			continue
		}
		kept, cut, err := truncateState(tok, c.State)
		if err != nil {
			t.Fatal(err)
		}
		if cut != c.Truncated || kept != c.Kept {
			t.Errorf("%s: cut %v (want %v); kept text differs from the reference (%d vs %d bytes)", c.Name, cut, c.Truncated, len(kept), len(c.Kept))
			continue
		}
		pids, err := tok.EncodePlain(Render(KindNoul, kept, "Is the order late?", []string{"false", "true"}))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(pids, c.PromptIDs) {
			t.Errorf("%s: the prompt encodes to %d ids, the reference's to %d, or they differ", c.Name, len(pids), len(c.PromptIDs))
		}
		if c.HasReplacement {
			repl++
		}
	}
	if repl < 2 {
		t.Fatalf("only %d cases carry a U+FFFD; the replacement would go untested", repl)
	}
}
