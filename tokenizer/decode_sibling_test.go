package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
	"unicode/utf8"
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

// TestDecodeContinuation_isIncrementallyAssociative is audit R-08's gate: streamTokens re-decoded
// the WHOLE generated id sequence on every token (O(n^2) in output length) instead of decoding
// just the new suffix and appending, out of caution that byte-fallback fusion — a run of raw
// bytes accumulated across CONSECUTIVE byte-fallback tokens, only written out as one unit — might
// make an arbitrary split point unsafe.
//
// It does not: decode()'s per-token loop has NO state that depends on chunk boundaries. A
// byte-fallback token appends its raw byte to `pending`; flush() writes pending as-is
// (unconditionally, with no UTF-8 interpretation) whenever a non-byte-fallback token arrives OR
// the slice ends. Splitting a call into several shorter calls just moves WHEN a flush happens,
// never WHAT gets written — string concatenation is associative regardless of where the pending
// bytes get flushed, and a flush that lands mid-multibyte-rune produces the identical bytes a
// later flush of the same bytes would have. So decode(ids) == concat(decode([ids[i]]) for i in
// ids), token by token, for every split point — proven here across a real multi-byte-emoji
// boundary (🦄 = 4 UTF-8 bytes = 4 consecutive byte-fallback tokens) with normal pieces on both
// sides, exactly the shape a real generation produces.
func TestDecodeContinuation_isIncrementallyAssociative(t *testing.T) {
	emoji := "🦄" // F0 9F A6 84 — a 4-byte UTF-8 rune, the classic byte-fallback stress case
	if len(emoji) != 4 {
		t.Fatalf("test setup: %q is %d bytes, want 4", emoji, len(emoji))
	}
	byteToVal := map[int32]byte{}
	idToPiece := []string{"x", "", "", "", "", "y", "", "", "z"}
	// ids 1-4: 🦄's four bytes, byte-fallback. ids 6-7: "€"'s three... use 2 bytes for variety
	// (a truncated/lone continuation byte is still just raw bytes to decode(), no validation).
	for i, b := range []byte(emoji) {
		byteToVal[int32(1+i)] = b
	}
	euro := []byte("€") // E2 82 AC, 3 bytes — a second, separate run later in the sequence
	idToPiece = append(idToPiece, "", "", "")
	for i, b := range euro {
		byteToVal[int32(9+i)] = b
	}
	tk := &Tokenizer{idToPiece: idToPiece, byteToVal: byteToVal}

	ids := []int{0, 1, 2, 3, 4, 5, 9, 10, 11, 8} // x + 🦄 + y + € + z
	full, err := tk.DecodeContinuation(ids)
	if err != nil {
		t.Fatalf("DecodeContinuation: %v", err)
	}
	want := "x" + emoji + "y" + string(euro) + "z"
	if full != want {
		t.Fatalf("test setup: full decode = %q, want %q", full, want)
	}

	var incremental string
	for _, id := range ids {
		piece, err := tk.DecodePiece(id)
		if err != nil {
			t.Fatalf("DecodePiece(%d): %v", id, err)
		}
		incremental += piece
	}
	if incremental != full {
		t.Errorf("incremental DecodePiece concatenation = %q, want %q (DecodeContinuation of the "+
			"whole sequence) — decode is not associative at every split point, so streamTokens "+
			"cannot decode just the new suffix per token", incremental, full)
	}
}

// N-24: byte-level decode pushed ADDED-token content through the byte table. An added token's
// surface is stored VERBATIM (it is not byte-level-encoded), so a rune in U+0080–U+0143 — é, ü,
// ñ, and every chat template that spells a role in a non-ASCII language — mapped back to ONE raw
// byte. The result is invalid UTF-8 and, worse, a wrong surface for the constrained-decoding
// mask, which builds its token table from these strings.
func TestDecodeByteLevel_addedTokensAreVerbatim(t *testing.T) {
	tk := &Tokenizer{
		mode:        modeByteLevel,
		idToPiece:   []string{"Hello", "<|café|>", "Ġworld"},
		byteDecoder: map[rune]byte{},
	}
	// A minimal byte-level table: ASCII maps to itself, and Ġ is the GPT-2 space marker. The
	// accented rune is deliberately IN the table, because that is exactly the situation — the
	// byte-level alphabet covers U+0080–U+0143 and an added token's é collides with it.
	for r := rune(33); r < 127; r++ {
		tk.byteDecoder[r] = byte(r)
	}
	tk.byteDecoder['Ġ'] = ' '
	tk.byteDecoder['é'] = 0xE9 // the collision: é is a byte-level alphabet member

	// Premise: without the added-token check, é decodes to the single byte 0xE9 — invalid UTF-8.
	tk.isAdded = nil
	if got, _ := tk.Decode([]int{1}); utf8.ValidString(got) {
		t.Fatalf("premise broke: %q is valid UTF-8, so this fixture no longer reproduces N-24", got)
	}

	// With the token marked as added, its surface comes back verbatim.
	tk.markAdded(1)
	got, err := tk.Decode([]int{1})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got != "<|café|>" {
		t.Errorf("added token decoded to %q, want %q — its surface is stored verbatim and must "+
			"not go through the byte table (N-24)", got, "<|café|>")
	}
	if !utf8.ValidString(got) {
		t.Error("added-token decode produced invalid UTF-8")
	}

	// Ordinary byte-level tokens must be UNAFFECTED — a fix that emitted everything verbatim
	// would break the byte-level decoding this mode exists for.
	if got, _ := tk.Decode([]int{2}); got != " world" {
		t.Errorf("byte-level token decoded to %q, want %q — the byte table must still apply to "+
			"non-added tokens", got, " world")
	}
}
