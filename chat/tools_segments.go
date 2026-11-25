package chat

import "strings"

// toolBuf builds a tool-aware prompt as segments. A family's control tokens are special spans; every other byte, the template's
// own text and everything a user, a tool, or the model supplied alike, goes into ONE non-special gap between two tokens, which is
// tokenized without the added-token trie: a marker string typed into a message stays literal text and cannot forge a turn.
//
// Template text is written with lit, which finds the family's markers in it and makes them tokens; anything untrusted is written
// with txt, which never does. The concatenation of the segments is the prompt the family's template renders (the goldens under
// testdata/chat_goldens pin it byte for byte through Template.RenderTools).
type toolBuf struct {
	buf   segBuf            // named (not embedded) so callers cannot bypass gap via segBuf.ct or direct segs access
	gap   strings.Builder   // accumulates consecutive non-special text to avoid O(N^2) string concatenation
	tk    func(string) bool // does the vocabulary hold this marker as a token? (Template.has, or always true when unknown)
	marks []string          // the family's control-token strings
}

// newToolBuf returns a builder for a family whose control tokens are marks.
func (t *Template) newToolBuf(marks ...string) *toolBuf {
	tk := t.has
	if tk == nil {
		tk = func(string) bool { return true }
	}
	return &toolBuf{tk: tk, marks: marks}
}

func (b *toolBuf) flushGap() {
	if b.gap.Len() == 0 {
		return
	}
	b.buf.segs = append(b.buf.segs, Segment{Text: b.gap.String()})
	b.gap.Reset()
}

// out flushes any buffered trailing gap text and returns the completed segments.
func (b *toolBuf) out() []Segment {
	b.flushGap()
	return b.buf.segs
}

// sp flushes any open gap text before appending a special token segment.
func (b *toolBuf) sp(s string) {
	b.flushGap()
	b.buf.sp(s)
}

// txt appends untrusted or plain text to the open gap, merging into the previous non-special span so the whole gap between two
// tokens is tokenized once, as the whole-string path tokenizes it.
func (b *toolBuf) txt(s string) {
	if s == "" {
		return
	}
	b.gap.WriteString(s)
}

// mark appends a control token: a special span when the vocabulary has it, otherwise ordinary text in the gap (a vocabulary
// without the token tokenizes the string as text whichever way it is written, and a separate span would only cut the merge across
// its edge).
func (b *toolBuf) mark(s string) {
	if b.tk(s) {
		b.sp(s)
		return
	}
	b.txt(s)
}

// lit appends template text that carries no untrusted bytes: each marker in it becomes a token, the rest is gap text.
func (b *toolBuf) lit(s string) { b.litMarks(s, b.marks) }

// litMarks is lit restricted to the markers given, for text whose only structural markers are those (Gemma's string quote inside a
// rendered schema, whose strings may mention other markers as plain text).
func (b *toolBuf) litMarks(s string, marks []string) {
	for s != "" {
		at, mk := -1, ""
		for _, m := range marks {
			if i := strings.Index(s, m); i >= 0 && (at < 0 || i < at || i == at && len(m) > len(mk)) {
				at, mk = i, m
			}
		}
		if at < 0 {
			b.txt(s)
			return
		}
		b.txt(s[:at])
		b.mark(mk)
		s = s[at+len(mk):]
	}
}
