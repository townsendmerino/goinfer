package tokenizer

import (
	"math/rand"
	"slices"
	"testing"
)

// cutInputs are valid text, then what a client's bytes can be that is not: every malformed shape UTF-8 has.
func cutInputs() []string {
	inputs := []string{
		"", "a", "plain ascii words", "héllo wörld 日本語 🦊", "�", "a�b",
		"\xff", "\xff\xfe abc", "a\xc3", "\xc3\x28", "\xed\xa0\x80 x", "ok \xe2\x82 tail\n\n", "\xf0\x9f\x98", "\xf4\x90\x80\x80",
		string([]byte{0, 1, 2, 127, 128, 200, 255}),
	}
	r := rand.New(rand.NewSource(3))
	for range 2000 {
		b := make([]byte, r.Intn(24))
		r.Read(b)
		inputs = append(inputs, string(b))
	}
	return inputs
}

// A pretoken cut from runeOffsets' string is the string its runes re-encode to, on any input: without the invalid-byte
// rewrite a cut would keep the raw bytes, and the token ids of a malformed prompt would change.
func TestRuneOffsets_cutIsTheRunesReencoded(t *testing.T) {
	for _, s := range cutInputs() {
		want := []rune(s)
		rs, off, cut := runeOffsets(s)
		if !slices.Equal(rs, want) {
			t.Fatalf("%q: runes %q, want %q", s, rs, want)
		}
		if len(off) != len(rs)+1 || off[len(rs)] != len(cut) {
			t.Fatalf("%q: %d offsets for %d runes, last %d of %d bytes", s, len(off), len(rs), off[len(off)-1], len(cut))
		}
		for a := range len(rs) + 1 {
			for b := a; b <= len(rs); b++ {
				if got, want := cut[off[a]:off[b]], string(rs[a:b]); got != want {
					t.Fatalf("%q [%d:%d]: cut %q, re-encoded %q", s, a, b, got, want)
				}
			}
		}
	}
}

func TestRuneAt_isTheRuneAsAString(t *testing.T) {
	for _, s := range cutInputs() {
		for i, r := range s {
			if got := runeAt(s, i, r); got != string(r) {
				t.Fatalf("%q at %d (%U): %q, want %q", s, i, r, got, string(r))
			}
		}
	}
}

// Encode allocates per gap and per pretoken at most, not per merge, per byte or per heap operation.
func TestEncode_allocationsBounded(t *testing.T) {
	tk := tkLoad(t, "testdata/qwen2-0.5b-coder-meta.gguf")
	got := testing.AllocsPerRun(5, func() {
		if _, err := tk.Encode(tkCorpus, false); err != nil {
			t.Fatal(err)
		}
	})
	const limit = 300
	if got > limit {
		t.Errorf("Encode allocated %v times on a %d-byte corpus, want at most %d", got, len(tkCorpus), limit)
	}
}
