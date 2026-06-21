package constrain

import "testing"

// TestForcedRun gates the 01 grammar-fused drafter primitive: ForcedRun must return
// exactly the tokens the grammar determines, and nothing where a choice remains. With
// a byte-level vocab "one legal token" == "one legal byte", so this also demonstrates
// the whitespace caveat directly: forcing fires INSIDE a fixed key literal but NOT at
// a structural boundary where optional whitespace is also legal.
func TestForcedRun(t *testing.T) {
	type One struct {
		K string `json:"k"`
	}
	tokens, eos := fullByteVocab()
	bytesOf := func(run []int) string {
		b := make([]byte, len(run))
		for i, id := range run {
			b[i] = byte(id) // fullByteVocab: token id == byte value
		}
		return string(b)
	}
	masker := func(prefix string) *Masker {
		g, err := GrammarFromStruct(One{})
		if err != nil {
			t.Fatal(err)
		}
		m := NewMasker(g, tokens, []int{eos})
		gen := make([]int, len(prefix))
		for i := range prefix {
			gen[i] = int(prefix[i])
		}
		m.Process(gen, make([]float32, len(tokens))) // fold the prefix into the grammar
		return m
	}

	// Inside the only property's key: after `{"`, the single property "k" forces the
	// key character then its closing quote — a real forced run.
	if got := bytesOf(masker(`{"`).ForcedRun(8)); got != `k"` {
		t.Errorf("ForcedRun after `{\"` = %q, want `k\"`", got)
	}

	// Caveat: after `{`, optional whitespace tokens are legal alongside `\"`/`}`, so
	// nothing is single-forced at the structural boundary.
	if got := masker(`{`).ForcedRun(8); len(got) != 0 {
		t.Errorf("ForcedRun after `{` = %q, want empty (ws optional ⇒ not forced)", bytesOf(got))
	}

	// After the colon (`{"k":`), the value of a string field opens with `"` — but
	// optional whitespace before it means it isn't forced either; confirms the run
	// stops at the boundary rather than over-forcing.
	if got := masker(`{"k":`).ForcedRun(8); len(got) != 0 {
		t.Errorf("ForcedRun after `{\"k\":` = %q, want empty (ws before value)", bytesOf(got))
	}

	// max is honored: cap the forced run length.
	if got := bytesOf(masker(`{"`).ForcedRun(1)); got != `k` {
		t.Errorf("ForcedRun(max=1) after `{\"` = %q, want `k`", got)
	}
}

// TestForcedBytesRun gates the BYTE-level forced-run primitive (the BPE-appropriate
// one). After `{"` the only property "k" forces the key char + its closing quote at
// the byte level — the same bytes ForcedRun finds, but byte-level forcing keeps
// firing where token-level forcing wouldn't on a real vocab (inc-2 finding).
func TestForcedBytesRun(t *testing.T) {
	type One struct {
		K string `json:"k"`
	}
	tokens, eos := fullByteVocab()
	masker := func(prefix string) *Masker {
		g, err := GrammarFromStruct(One{})
		if err != nil {
			t.Fatal(err)
		}
		m := NewMasker(g, tokens, []int{eos})
		gen := make([]int, len(prefix))
		for i := range prefix {
			gen[i] = int(prefix[i])
		}
		m.Process(gen, make([]float32, len(tokens)))
		return m
	}
	if got := string(masker(`{"`).ForcedBytesRun(16)); got != `k"` {
		t.Errorf("ForcedBytesRun after `{\"` = %q, want `k\"`", got)
	}
	if got := masker(`{`).ForcedBytesRun(16); len(got) != 0 {
		t.Errorf("ForcedBytesRun after `{` = %q, want empty (ws optional)", string(got))
	}
	if got := string(masker(`{"`).ForcedBytesRun(1)); got != `k` {
		t.Errorf("ForcedBytesRun(max=1) = %q, want `k`", got)
	}
}
