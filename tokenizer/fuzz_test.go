package tokenizer

import "testing"

// Track 2.5 (testing campaign): tokenizer.json is untrusted input (it ships in a
// model directory). parseTokenizerJSON is the parse surface; it must turn a
// malformed/hostile file into a typed error, never a panic or OOM.

var tokSeeds = []string{
	`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}`,
	`{"model":{"type":"BPE","vocab":{"<s>":0,"a":1},"merges":[]},"added_tokens":[{"id":1,"content":"a"}]}`,
	`{"model":{"vocab":{"x":0},"merges":[]}}`, // empty type (GPT-2)
	// hostile ids — exercise the idToPiece sizing/indexing.
	`{"model":{"type":"BPE","vocab":{"x":2147483647}}}`, // maxID+1 overflows int32
	`{"model":{"type":"BPE","vocab":{"x":-5}}}`,         // negative id
	`{"model":{"type":"BPE","vocab":{"x":1073741824}}}`, // huge positive id → big make
}

// FuzzParseTokenizerJSON feeds arbitrary bytes to the tokenizer.json parser: a
// typed error or a built Tokenizer, never a panic. jsonPath names a directory
// with no sibling files, so a byte-level pipeline cleanly errors rather than
// reading the host filesystem.
func FuzzParseTokenizerJSON(f *testing.F) {
	for _, s := range tokSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		tk, err := parseTokenizerJSON(raw, "fuzz-nonexistent-dir/tokenizer.json", "fuzz-nonexistent-dir")
		if err != nil {
			return
		}
		// A built tokenizer must be self-consistent: every vocab id is a valid
		// index into idToPiece (else Decode/TokenText would panic at runtime).
		for _, id := range tk.vocab {
			if int(id) < 0 || int(id) >= len(tk.idToPiece) {
				t.Fatalf("vocab id %d out of idToPiece range [0,%d)", id, len(tk.idToPiece))
			}
		}
	})
}
