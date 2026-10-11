package tokenizer

import "unicode/utf8"

// Stream turns generated token ids into text one id at a time, for a caller that prints as the tokens arrive.
// Decode per id is wrong for that: it treats every id as a whole sequence and drops each SentencePiece piece's
// leading space ("Theanswerisfour"), and DecodePiece alone can hand back half of a multi-byte character when the
// character is spread over byte-fallback or byte-level tokens. Stream does both jobs: it appends each id's own
// piece, holds back an incomplete trailing UTF-8 sequence until the next id completes it, and removes the
// sequence's single leading space once, on the first text it emits, exactly as Decode does for the whole sequence.
//
// The text written, concatenated, equals Decode of all the ids. A Stream is for one generation and is not safe for
// concurrent use. Create one with Tokenizer.NewStream.
type Stream struct {
	t       *Tokenizer
	held    []byte // trailing bytes that do not yet form a whole UTF-8 sequence
	started bool   // some text has been emitted: the leading-space strip is spent
}

// NewStream starts a streaming decode of one generation's ids.
func (t *Tokenizer) NewStream() *Stream { return &Stream{t: t} }

// Write adds one generated id and returns the text that is now complete: possibly empty (a special token that
// renders nothing, or the first bytes of a character), possibly several characters. An id outside the vocabulary
// is an error and changes nothing.
func (s *Stream) Write(id int) (string, error) {
	piece, err := s.t.DecodePiece(id)
	if err != nil {
		return "", err
	}
	s.held = append(s.held, piece...)
	end := 0
	for end < len(s.held) && utf8.FullRune(s.held[end:]) {
		_, size := utf8.DecodeRune(s.held[end:])
		end += size
	}
	out := string(s.held[:end])
	s.held = append(s.held[:0], s.held[end:]...)
	return s.emit(out), nil
}

// Flush returns whatever Write is still holding: the bytes of an incomplete character at the end of the generation,
// as the whole-sequence Decode would write them. Call it once, after the last id.
func (s *Stream) Flush() string {
	out := string(s.held)
	s.held = s.held[:0]
	return s.emit(out)
}

// emit applies the sequence-level leading-space strip to the first non-empty text.
func (s *Stream) emit(text string) string {
	if text == "" {
		return ""
	}
	if !s.started {
		s.started = true
		if s.t.stripLeadingSpace && text[0] == ' ' {
			text = text[1:]
		}
	}
	return text
}
