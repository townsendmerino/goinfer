package harmony

import (
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
)

const (
	hmStart     = "<|start|>"
	hmChannel   = "<|channel|>"
	hmMessage   = "<|message|>"
	hmConstrain = "<|constrain|>"
	hmEnd       = "<|end|>"
	hmReturn    = "<|return|>"
	hmCall      = "<|call|>"

	hmMaxHeader = 512
)

var (
	hmTopMarkers  = []string{hmStart, hmChannel, hmEnd, hmReturn, hmCall}
	hmBodyMarkers = []string{hmEnd, hmReturn, hmCall}
)

type hmState uint8

const (
	hmTop hmState = iota
	hmHeader
	hmBody
)

type hmRoute uint8

const (
	hmToContent hmRoute = iota
	hmToReasoning
	hmToCall
)

// Call is a Harmony channel message addressed to a recipient.
type Call struct {
	Channel, Recipient, Constrain string
	Args                          string
}

// Splitter incrementally separates a streamed Harmony reply into reasoning, content, and recipient calls.
type Splitter struct {
	state  hmState
	buf    string
	header string

	route   hmRoute
	started bool
	stray   bool

	emitted   [3]bool
	sawReason bool
	answered  bool

	calls       []Call
	curCallArgs strings.Builder
}

// NewSplitter returns a fresh Harmony channel stream splitter.
func NewSplitter() *Splitter { return &Splitter{} }

// Calls returns the raw recipient call messages collected by the splitter.
func (s *Splitter) Calls() []Call { return s.calls }

// ToolCalls converts the splitter's collected recipient messages into structured ToolCalls.
func (s *Splitter) ToolCalls() []chatcore.ToolCall { return ToolCalls(s.calls) }

// InReasoning reports whether the reply has entered analysis reasoning without yet emitting answer content.
func (s *Splitter) InReasoning() bool {
	if s.state == hmBody && s.route == hmToContent {
		return false
	}
	return s.sawReason && !s.answered
}

// Push feeds the next decoded chunk and returns reasoning and content safe to emit immediately.
func (s *Splitter) Push(chunk string) (reasoning, content string) {
	s.buf += chunk
	var out [3]strings.Builder
	emit := func(r hmRoute, text string) {
		if text == "" {
			return
		}
		if !s.started {
			s.started = true
			if s.emitted[r] {
				text = "\n\n" + text
			}
			s.emitted[r] = true
		}
		if r == hmToContent {
			s.answered = true
		}
		out[r].WriteString(text)
	}
	strayText := func(text string) {
		if !s.stray {
			s.stray, s.route, s.started = true, hmToContent, false
		}
		emit(hmToContent, text)
	}
	for {
		switch s.state {
		case hmTop:
			tr := strings.TrimLeft(s.buf, " \t\r\n")
			if tr == "" {
				return out[hmToReasoning].String(), out[hmToContent].String()
			}
			i, which, hold := scanMarkers(s.buf, hmTopMarkers)
			if i < 0 {
				if !strings.HasPrefix(tr, "<") || hold == 0 {
					keep := len(strings.TrimRight(s.buf[:len(s.buf)-hold], " \t\r\n"))
					strayText(s.buf[:keep])
					s.buf = s.buf[keep:]
				}
				return out[hmToReasoning].String(), out[hmToContent].String()
			}
			if lead := strings.TrimRight(s.buf[:i], " \t\r\n"); strings.TrimSpace(lead) != "" {
				strayText(lead)
			}
			s.stray = false
			s.buf = s.buf[i+len(which):]
			switch which {
			case hmStart:
				s.state, s.header = hmHeader, ""
			case hmChannel:
				s.state, s.header = hmHeader, hmChannel
			default:
			}
		case hmHeader:
			i := strings.Index(s.buf, hmMessage)
			if i < 0 {
				if len(s.header)+len(s.buf) > hmMaxHeader {
					s.stray = false
					strayText(s.header + s.buf)
					s.header, s.buf, s.state = "", "", hmTop
					continue
				}
				_, _, hold := scanMarkers(s.buf, []string{hmMessage})
				s.header += s.buf[:len(s.buf)-hold]
				s.buf = s.buf[len(s.buf)-hold:]
				return out[hmToReasoning].String(), out[hmToContent].String()
			}
			s.header += s.buf[:i]
			s.buf = s.buf[i+len(hmMessage):]
			s.openMessage()
		case hmBody:
			i, which, hold := scanMarkers(s.buf, hmBodyMarkers)
			if i < 0 {
				s.body(s.buf[:len(s.buf)-hold], emit)
				s.buf = s.buf[len(s.buf)-hold:]
				return out[hmToReasoning].String(), out[hmToContent].String()
			}
			s.body(s.buf[:i], emit)
			if s.route == hmToCall && len(s.calls) > 0 {
				s.calls[len(s.calls)-1].Args = s.curCallArgs.String()
				s.curCallArgs.Reset()
			}
			s.buf = s.buf[i+len(which):]
			s.state = hmTop
		}
	}
}

func (s *Splitter) body(text string, emit func(hmRoute, string)) {
	if text == "" {
		return
	}
	if s.route == hmToCall {
		s.curCallArgs.WriteString(text)
		return
	}
	emit(s.route, text)
}

func (s *Splitter) openMessage() {
	channel, recipient, constrain := parseHarmonyHeader(s.header)
	s.header = ""
	s.state, s.started, s.stray = hmBody, false, false
	switch {
	case recipient != "":
		s.route = hmToCall
		s.curCallArgs.Reset()
		s.calls = append(s.calls, Call{Channel: channel, Recipient: recipient, Constrain: constrain})
	case channel == "analysis":
		s.route, s.sawReason = hmToReasoning, true
	default:
		s.route = hmToContent
	}
}

// Flush ends the reply and returns any held-back reasoning or content tail.
func (s *Splitter) Flush() (reasoning, content string) {
	defer func() { s.buf, s.header = "", "" }()
	if s.state == hmBody && s.route == hmToCall && len(s.calls) > 0 {
		if strings.TrimSpace(s.buf) != "" {
			s.curCallArgs.WriteString(s.buf)
		}
		s.calls[len(s.calls)-1].Args = s.curCallArgs.String()
		s.curCallArgs.Reset()
		return "", ""
	}
	if s.state != hmBody && s.state != hmTop {
		return "", ""
	}
	if strings.TrimSpace(s.buf) == "" && !(s.state == hmTop && s.stray) {
		return "", ""
	}
	route := s.route
	if s.state == hmTop {
		route = hmToContent
		if !s.stray {
			s.stray, s.started = true, false
		}
	}
	text := s.buf
	if !s.started {
		s.started = true
		if s.emitted[route] {
			text = "\n\n" + text
		}
		s.emitted[route] = true
	}
	if route == hmToReasoning {
		return text, ""
	}
	s.answered = true
	return "", text
}

func parseHarmonyHeader(h string) (channel, recipient, constrain string) {
	if before, after, ok := strings.Cut(h, hmConstrain); ok {
		constrain = firstField(after)
		h = before
	}
	if _, after, ok := strings.Cut(h, hmChannel); ok {
		channel = firstField(after)
	}
	if _, after, ok := strings.Cut(h, "to="); ok {
		recipient = firstField(after)
	}
	return channel, recipient, constrain
}

func firstField(s string) string {
	if i := strings.IndexAny(s, " \t\r\n<"); i >= 0 {
		s = s[:i]
	}
	return s
}

func scanMarkers(s string, markers []string) (idx int, which string, hold int) {
	idx = -1
	for _, m := range markers {
		if i := strings.Index(s, m); i >= 0 && (idx < 0 || i < idx) {
			idx, which = i, m
		}
	}
	if idx >= 0 {
		return idx, which, 0
	}
	for _, m := range markers {
		for k := min(len(m)-1, len(s)); k > hold; k-- {
			if strings.HasSuffix(s, m[:k]) {
				hold = k
				break
			}
		}
	}
	return -1, "", hold
}
