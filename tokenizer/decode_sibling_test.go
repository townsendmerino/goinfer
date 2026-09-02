package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDecodePiece_noSequenceStrip_M13 gates M-13: DecodePiece renders ONE token for streaming
// and must not apply the whole-sequence dummy-prefix strip. On a Llama-2/Mistral tokenizer
// (stripLeadingSpace) DecodePiece(id("▁word")) previously returned "word", so a caller printing
// piece-by-piece emitted "Theanswer". Decode (whole sequence) still strips the one leading space.
func TestDecodePiece_noSequenceStrip_M13(t *testing.T) {
	tk := &Tokenizer{idToPiece: []string{"▁The", "▁answer"}, stripLeadingSpace: true}
	// Whole-sequence decode strips exactly one leading space.
	if got, _ := tk.Decode([]int{0, 1}); got != "The answer" {
		t.Errorf("Decode = %q, want %q", got, "The answer")
	}
	// Per-piece decode keeps each token's leading space, so streaming concatenation is faithful.
	p0, _ := tk.DecodePiece(0)
	p1, _ := tk.DecodePiece(1)
	if p0 != " The" || p1 != " answer" {
		t.Errorf("DecodePiece pieces = %q,%q; want %q,%q (M-13: no per-piece strip)", p0, p1, " The", " answer")
	}
	if p0+p1 != " The answer" {
		t.Errorf("streamed %q, want %q (internal space must survive)", p0+p1, " The answer")
	}
}

// TestLoadJSONBytes_noCWDSibling_M14 gates M-14: LoadJSONBytes must NOT read a
// tokenizer_config.json from the process CWD. A stray config in CWD used to be adopted
// silently (wrong BOS/template). Here a byte-level tokenizer.json with a poison config in CWD
// must load with BOS unset, proving the sibling read was skipped.
func TestLoadJSONBytes_noCWDSibling_M14(t *testing.T) {
	// A minimal byte-level tokenizer.json (Decoder.Type ByteLevel → modeByteLevel, which is the
	// path that reads siblings). Vocab includes a "<s>" the poison config would resolve as BOS.
	blob := []byte(`{"model":{"type":"BPE","vocab":{"<s>":0,"a":1},"merges":[]},` +
		`"decoder":{"type":"ByteLevel"},"pre_tokenizer":{"type":"ByteLevel"}}`)
	// Poison tokenizer_config.json in CWD naming <s> as bos_token.
	dir := t.TempDir()
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"bos_token":"<s>"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := LoadJSONBytes(blob)
	if err != nil {
		t.Fatalf("LoadJSONBytes: %v", err)
	}
	if tk.special.BOS != -1 {
		t.Errorf("LoadJSONBytes adopted the CWD tokenizer_config.json (BOS=%d, want -1) — M-14 not closed", tk.special.BOS)
	}
}

// M-25: the serving loop decoded GENERATED ids with Decode, which applies SentencePiece's
// dummy-prefix strip — a SEQUENCE-level rule. The generated ids are a CONTINUATION of the
// prompt, never a sequence, so a response opening with `▁Paris` reached the client as
// "Paris" where OpenAI and llama.cpp both return " Paris".
//
// Built from a hand-made dummy-prefix tokenizer rather than asserting the claim, so the
// premise (that Decode really does strip) is proven in the same test that proves the fix.
func TestDecodeContinuation_keepsTheLeadingSpace(t *testing.T) {
	tk := &Tokenizer{idToPiece: []string{"▁Paris", "▁is"}, stripLeadingSpace: true} // Llama-2 / Mistral
	const paris = 0

	// The premise: Decode strips, which is CORRECT for a whole sequence.
	whole, err := tk.Decode([]int{paris})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if whole != "Paris" {
		t.Fatalf("premise broke: Decode gave %q, want %q — this fixture no longer has the "+
			"dummy-prefix behaviour the defect depends on", whole, "Paris")
	}

	// The fix: a continuation keeps it.
	cont, err := tk.DecodeContinuation([]int{paris})
	if err != nil {
		t.Fatalf("DecodeContinuation: %v", err)
	}
	if cont != " Paris" {
		t.Errorf("DecodeContinuation gave %q, want %q — the generated ids continue the prompt, "+
			"so the sequence-level dummy-prefix strip does not apply to them (M-25)", cont, " Paris")
	}

	// Not just the first token: the strip is once-per-string, so a multi-token continuation
	// must be unaffected beyond its head.
	if got, _ := tk.DecodeContinuation([]int{paris, 1}); got != " Paris is" {
		t.Errorf("multi-token continuation = %q, want %q", got, " Paris is")
	}
}

// A tokenizer that does NOT strip (Gemma, and every byte-level family) must be unchanged by
// the fix — without this, a change that simply stopped stripping everywhere would pass.
func TestDecodeContinuation_identicalWhenNothingIsStripped(t *testing.T) {
	tk := &Tokenizer{idToPiece: []string{"▁Paris"}, stripLeadingSpace: false}
	d, _ := tk.Decode([]int{0})
	c, _ := tk.DecodeContinuation([]int{0})
	if d != c {
		t.Errorf("Decode %q != DecodeContinuation %q on a non-stripping tokenizer", d, c)
	}
	if c != " Paris" {
		t.Errorf("got %q, want %q", c, " Paris")
	}
}
