package chatcore

import (
	"encoding/json"
	"strings"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Segment is a span of the rendered prompt tagged special (tokenize with the added-token trie)
// or plain content (tokenize without it).
type Segment = tokenizer.Segment

// Turn is one conversation message in internal dialect form.
type Turn struct {
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolName   string
	ToolCallID string
	Reasoning  string
	ToolLoop   bool
}

// Tool is a callable function declaration with a JSON Schema parameters object.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ToolCall is one structured function call emitted by or replayed to a model.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolFormat selects how a family's tools are rendered when a native template form exists.
type ToolFormat uint8

const (
	ToolFormatAuto ToolFormat = iota
	ToolFormatHermes
	ToolFormatTemplate
)

// ParseToolFormat parses a tool-format name ("" is auto).
func ParseToolFormat(s string) (ToolFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "auto":
		return ToolFormatAuto, true
	case "hermes":
		return ToolFormatHermes, true
	case "template":
		return ToolFormatTemplate, true
	}
	return ToolFormatAuto, false
}

func (f ToolFormat) String() string {
	switch f {
	case ToolFormatHermes:
		return "hermes"
	case ToolFormatTemplate:
		return "template"
	}
	return "auto"
}

// SegBuf accumulates prompt render segments.
type SegBuf struct {
	Segs []Segment
}

// Sp appends a genuine special-token span (tokenized with the added-token trie).
func (b *SegBuf) Sp(text string) {
	b.Segs = append(b.Segs, Segment{Text: text, Special: true})
}

// Ct appends a non-special content gap (tokenized without the added-token trie).
func (b *SegBuf) Ct(text string) {
	if text != "" {
		b.Segs = append(b.Segs, Segment{Text: text, Special: false})
	}
}

// ToolBuf builds a tool-aware prompt as segments where control tokens become special spans
// and untrusted text merges into single non-special gaps between markers.
type ToolBuf struct {
	buf   SegBuf
	gap   strings.Builder
	tk    func(string) bool
	marks []string
}

// NewToolBuf returns a segment builder for a family whose control tokens are marks.
// A nil hasToken callback treats every declared marker as present in the vocabulary.
func NewToolBuf(hasToken func(string) bool, marks ...string) *ToolBuf {
	if hasToken == nil {
		hasToken = func(string) bool { return true }
	}
	return &ToolBuf{tk: hasToken, marks: marks}
}

func (b *ToolBuf) flushGap() {
	if b.gap.Len() == 0 {
		return
	}
	b.buf.Segs = append(b.buf.Segs, Segment{Text: b.gap.String()})
	b.gap.Reset()
}

// Out flushes any buffered gap text and returns the completed segments.
func (b *ToolBuf) Out() []Segment {
	b.flushGap()
	return b.buf.Segs
}

// Sp flushes any open gap text and appends a special-token span.
func (b *ToolBuf) Sp(s string) {
	b.flushGap()
	b.buf.Sp(s)
}

// Txt appends untrusted or plain text to the open non-special gap.
func (b *ToolBuf) Txt(s string) {
	if s == "" {
		return
	}
	b.gap.WriteString(s)
}

// Mark appends s as a special span when the vocabulary holds it as a token, or as gap text otherwise.
func (b *ToolBuf) Mark(s string) {
	if b.tk(s) {
		b.Sp(s)
		return
	}
	b.Txt(s)
}

// Lit appends trusted template text, turning each marker from marks into a token span.
func (b *ToolBuf) Lit(s string) { b.LitMarks(s, b.marks) }

// LitMarks is Lit restricted to the given marker subset.
func (b *ToolBuf) LitMarks(s string, marks []string) {
	for s != "" {
		at, mk := -1, ""
		for _, m := range marks {
			if i := strings.Index(s, m); i >= 0 && (at < 0 || i < at || i == at && len(m) > len(mk)) {
				at, mk = i, m
			}
		}
		if at < 0 {
			b.Txt(s)
			return
		}
		b.Txt(s[:at])
		b.Mark(mk)
		s = s[at+len(mk):]
	}
}
