package decide

import (
	"errors"
	"unicode/utf8"
)

// MaxStateTokens is jev_core's MAX_STATE_TOKENS: the reference decide() cuts a longer state to its first 60% and last
// 40% of tokens, decodes that back to text and renders the prompt from it. A trained head reads prompts built that way,
// so Route B does the same (scripts/pin_decisions_truncation.py pins it against transformers).
const MaxStateTokens = 1024

// StateDecoder is what Route B needs from a tokenizer beyond Tokenizer: decoding plain-encoded ids back to text the way
// transformers does, so that a truncated state is byte-identical to the reference's.
type StateDecoder interface {
	DecodePlain(ids []int) (string, error)
}

// truncateState returns the state a Route B prompt is rendered from: state itself when it is at most MaxStateTokens
// tokens, else the decoded first 60% and last 40% of its tokens (int(1024*0.6) = 614 and 410, as jev_core computes
// them). The bool says whether it cut.
func truncateState(tok Tokenizer, state string) (string, bool, error) {
	ids, err := tok.EncodePlain(state)
	if err != nil {
		return "", false, err
	}
	if len(ids) <= MaxStateTokens {
		return state, false, nil
	}
	dec, ok := tok.(StateDecoder)
	if !ok {
		return "", false, errors.New("decide: this state is over the decision head's 1024 tokens and the tokenizer cannot decode it back to cut it")
	}
	head := MaxStateTokens * 6 / 10
	kept := append(append([]int(nil), ids[:head]...), ids[len(ids)-(MaxStateTokens-head):]...)
	s, err := dec.DecodePlain(kept)
	return s, true, err
}

// pyReplaceInvalidUTF8 is CPython's bytes.decode("utf-8", errors="replace"), which transformers' byte-level decoder
// applies to the joined bytes: each maximal subpart of an ill-formed sequence becomes one U+FFFD (a lead byte and the
// valid continuation bytes that follow it, or one stray byte). Go's strings.ToValidUTF8 collapses a whole invalid run
// into one U+FFFD instead, and utf8.DecodeRune reports every byte of a truncated sequence separately; neither matches.
func pyReplaceInvalidUTF8(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	out := make([]byte, 0, len(b)+8)
	for i := 0; i < len(b); {
		c := b[i]
		if c < 0x80 {
			out = append(out, c)
			i++
			continue
		}
		n, lo, hi := 0, byte(0x80), byte(0xBF) // sequence length and the allowed range of its second byte
		switch {
		case c >= 0xC2 && c <= 0xDF:
			n = 2
		case c == 0xE0:
			n, lo = 3, 0xA0
		case c >= 0xE1 && c <= 0xEC, c == 0xEE, c == 0xEF:
			n = 3
		case c == 0xED:
			n, hi = 3, 0x9F
		case c == 0xF0:
			n, lo = 4, 0x90
		case c >= 0xF1 && c <= 0xF3:
			n = 4
		case c == 0xF4:
			n, hi = 4, 0x8F
		}
		if n == 0 { // a continuation byte or a byte that never starts a sequence
			out = utf8.AppendRune(out, utf8.RuneError)
			i++
			continue
		}
		j := i + 1
		for k := 1; k < n && j < len(b); k++ {
			l, h := byte(0x80), byte(0xBF)
			if k == 1 {
				l, h = lo, hi
			}
			if b[j] < l || b[j] > h {
				break
			}
			j++
		}
		if j-i == n {
			out = append(out, b[i:j]...)
		} else {
			out = utf8.AppendRune(out, utf8.RuneError)
		}
		i = j
	}
	return string(out)
}
