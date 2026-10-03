package clef

import "github.com/townsendmerino/goinfer/tokenizer"

// TokenizerFunc adapts a goinfer tokenizer to Tokenize: a fragment is tokenized one fragment at a time, without BOS, and parsed for special-token text only
// when the encoder says so (the two fixed templates), so request text stays literal (the M25 split, owner decision 2026-10-02 in
// docs/measurements/decisions-d12-clef-encoder-2026-10-02.md).
func TokenizerFunc(tk *tokenizer.Tokenizer) Tokenize {
	return func(text string, parseSpecial bool) ([]int, error) {
		return tk.EncodeSegments([]tokenizer.Segment{{Text: text, Special: parseSpecial}}, false)
	}
}
