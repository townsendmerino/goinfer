package reasoning

import (
	"strings"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

type splitState uint8

const (
	stUndecided splitState = iota
	stReasoning
	stAfterClose
	stContent
)

// ThinkSplitter incrementally separates a streamed reply delimited by open and close markers into reasoning and content.
type ThinkSplitter struct {
	open, close string
	state       splitState
	lead        bool
	buf         string
}

// NewThinkSplitter returns a splitter for open and close delimiters; forcedOpen starts inside Reasoning.
func NewThinkSplitter(open, close string, forcedOpen bool) *ThinkSplitter {
	s := &ThinkSplitter{open: open, close: close}
	if forcedOpen {
		s.state, s.lead = stReasoning, true
	}
	return s
}

// InReasoning reports whether the reply so far is inside an unclosed reasoning block.
func (s *ThinkSplitter) InReasoning() bool { return s.state == stReasoning }

func safeLen(pending, delim string) int {
	if delim == "" {
		return len(pending)
	}
	if i := strings.Index(pending, delim); i >= 0 {
		return i
	}
	maxHold := min(len(delim)-1, len(pending))
	for k := maxHold; k > 0; k-- {
		if strings.HasSuffix(pending, delim[:k]) {
			return len(pending) - k
		}
	}
	return len(pending)
}

// Push feeds the next decoded chunk and returns reasoning and content safe to emit immediately.
func (s *ThinkSplitter) Push(chunk string) (reasoning, content string) {
	s.buf += chunk
	var r, c strings.Builder
	for {
		switch s.state {
		case stUndecided:
			tr := strings.TrimLeft(s.buf, " \t\r\n")
			switch {
			case tr == "":
				return r.String(), c.String()
			case strings.HasPrefix(tr, s.open):
				s.buf, s.state, s.lead = tr[len(s.open):], stReasoning, true
				continue
			case len(tr) < len(s.open) && strings.HasPrefix(s.open, tr):
				return r.String(), c.String()
			}
			s.state = stContent
			continue
		case stReasoning:
			if s.lead {
				s.buf = strings.TrimLeft(s.buf, "\r\n")
				if s.buf == "" {
					return r.String(), c.String()
				}
				if strings.HasPrefix(s.buf, s.open) {
					s.buf = s.buf[len(s.open):]
					continue
				}
				if len(s.buf) < len(s.open) && strings.HasPrefix(s.open, s.buf) {
					return r.String(), c.String()
				}
				s.lead = false
			}
			if before, after, ok := strings.Cut(s.buf, s.close); ok {
				r.WriteString(strings.TrimRight(before, "\r\n"))
				s.buf, s.state = after, stAfterClose
				continue
			}
			n := safeLen(s.buf, s.close)
			safe := strings.TrimRight(s.buf[:n], "\r\n")
			r.WriteString(safe)
			s.buf = s.buf[len(safe):]
			return r.String(), c.String()
		case stAfterClose:
			s.buf = strings.TrimLeft(s.buf, "\r\n")
			if s.buf == "" {
				return r.String(), c.String()
			}
			s.state = stContent
			continue
		default:
			for {
				before, after, ok := strings.Cut(s.buf, s.close)
				if !ok {
					break
				}
				c.WriteString(before)
				s.buf = after
			}
			n := safeLen(s.buf, s.close)
			c.WriteString(s.buf[:n])
			s.buf = s.buf[n:]
			return r.String(), c.String()
		}
	}
}

// Flush ends the reply and returns any held-back reasoning or content tail.
func (s *ThinkSplitter) Flush() (reasoning, content string) {
	b := s.buf
	s.buf = ""
	switch s.state {
	case stReasoning:
		return strings.TrimRight(b, "\r\n"), ""
	case stUndecided, stContent:
		s.state = stContent
		return "", b
	}
	return "", ""
}

// ReplyCore is the stream-splitting interface consumed by ReplySplitter.
type ReplyCore interface {
	Push(chunk string) (reasoning, content string)
	Flush() (reasoning, content string)
	InReasoning() bool
}

type callSource interface {
	ToolCalls() []chatcore.ToolCall
}

// ReplySplitter wraps a ReplyCore to emit reasoning only on complete UTF-8 rune boundaries.
type ReplySplitter struct {
	sp    ReplyCore
	carry []byte
}

// NewReplySplitter wraps sp in a UTF-8 rune-boundary ReplySplitter.
func NewReplySplitter(sp ReplyCore) *ReplySplitter {
	if sp == nil {
		return nil
	}
	return &ReplySplitter{sp: sp}
}

// Push feeds the next decoded fragment and returns complete-rune reasoning and raw answer text.
func (r *ReplySplitter) Push(fragment string) (reasoning, answer string) {
	rs, a := r.sp.Push(fragment)
	return r.complete(rs), a
}

// Finish ends the reply and returns remaining reasoning, answer text, and whether thinking was truncated.
func (r *ReplySplitter) Finish() (reasoning, answer string, truncated bool) {
	truncated = r.sp.InReasoning()
	rs, a := r.sp.Flush()
	out := string(r.carry) + rs
	r.carry = nil
	return out, a, truncated
}

// CoreCalls returns structured tool calls collected by the underlying splitter when supported.
func (r *ReplySplitter) CoreCalls() []chatcore.ToolCall {
	if r == nil {
		return nil
	}
	if cs, ok := r.sp.(callSource); ok {
		return cs.ToolCalls()
	}
	return nil
}

func (r *ReplySplitter) complete(s string) string {
	if s == "" && len(r.carry) == 0 {
		return ""
	}
	buf := append(r.carry, s...)
	end := 0
	for i := 0; i < len(buf); {
		if !utf8.FullRune(buf[i:]) {
			break
		}
		_, size := utf8.DecodeRune(buf[i:])
		i += size
		end = i
	}
	out := string(buf[:end])
	r.carry = append(r.carry[:0], buf[end:]...)
	return out
}
