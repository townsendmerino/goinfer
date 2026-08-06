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
