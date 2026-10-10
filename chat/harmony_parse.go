package chat

// The Harmony (gpt-oss) reply parser: the OUTPUT half of the family whose prompt half is Harmony() in templates.go.
//
// A gpt-oss reply is not one delimited span but a sequence of MESSAGES, each routed to a CHANNEL:
//
//	<|channel|>analysis<|message|>…the model's reasoning…<|end|>
//	<|start|>assistant<|channel|>final<|message|>…the answer…            (then the turn stop, <|return|>, which is not in the text)
//
// and, when it calls a function, a message addressed to a recipient, ended by <|call|>:
//
//	<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}
//
// (the first message has no <|start|>assistant because the generation prompt already ends with it). The decoded stream carries these
// markers as literal text and never carries the turn stops, which are stop ids (docs/measurements/harmony-parser-2026-09-30/).
//
// Routing, what a client is shown:
//
//	analysis                      → reasoning
//	final                         → content
//	commentary, no recipient      → content   (a "preamble": text the model wrote to the user before acting)
//	commentary/anything, to=…     → a tool call: kept on the parser (calls), shown in neither stream
//	any other channel name        → content   (never hide text a client may need because a name is new)
//
// Several messages of one kind are joined by a blank line. Text before the first marker (a reply that does not speak Harmony at
// all) is content, so a model or prompt the parser does not recognise degrades to today's behaviour instead of going silent.
//
// Only three strings end a message: <|end|>, <|return|>, <|call|>. Inside a message body nothing else is structural, so a body
// that mentions <|channel|> or <|start|> as text is left alone — the cost is that a body that writes one of the three terminators
// as plain text ends early, which no text-level parser can tell apart from the token (the same limit the think splitter has).
//
// Correctness is chunk-independence, as for ThinkSplitter: any partition of a reply into Pushes produces the same concatenated
// output as one Push (TestHarmonySplitter_chunkIndependent). Every held-back byte is held because it could still become a marker.

import "strings"

const (
	hmStart     = "<|start|>"
	hmChannel   = "<|channel|>"
	hmMessage   = "<|message|>"
	hmConstrain = "<|constrain|>"
	hmEnd       = "<|end|>"
	hmReturn    = "<|return|>"
	hmCall      = "<|call|>"

	// hmMaxHeader bounds how long a header is waited for. A real one is a few dozen bytes; one that runs past this has no
	// <|message|> coming, and holding the reply forever would be worse than showing it.
	hmMaxHeader = 512
)

var (
	hmTopMarkers  = []string{hmStart, hmChannel, hmEnd, hmReturn, hmCall}
	hmBodyMarkers = []string{hmEnd, hmReturn, hmCall}
)

type hmState uint8

const (
	hmTop    hmState = iota // between messages (and before the first)
	hmHeader                // after <|start|> or <|channel|>, until <|message|>
	hmBody                  // the message text, until a terminator
)

type hmRoute uint8

const (
	hmToContent hmRoute = iota
	hmToReasoning
	hmToCall
)

// harmonyCall is a message addressed to a recipient — a function call the model made. The parser keeps it, shown in neither stream;
// harmonyToolCalls turns the kept calls into ToolCalls (ReplySplitter.ToolCalls).
type harmonyCall struct {
	Channel, Recipient, Constrain string
	Args                          string
}

type harmonySplitter struct {
	state  hmState
	buf    string // held-back input: a partial marker, a header being read, or leading whitespace at the top
	header string // the header being read, raw (markers included)

	route   hmRoute
	started bool // the current message has emitted its first byte
	stray   bool // the current "message" is text that arrived outside any header (see strayText)

	emitted   [3]bool // per route: something was emitted by an earlier message (so the next one is separated)
	sawReason bool    // an analysis message was opened
	answered  bool    // content was emitted

	calls []harmonyCall
}

func newHarmonySplitter() *harmonySplitter { return &harmonySplitter{} }

// InReasoning reports whether the reply so far has reasoned but produced no answer — the reply was cut off while thinking, or
// between thinking and answering, so there is no answer to show.
func (s *harmonySplitter) InReasoning() bool {
	if s.state == hmBody && s.route == hmToContent {
		return false
	}
	return s.sawReason && !s.answered
}

// Push feeds the next decoded chunk and returns the reasoning and content that are safe to emit now (either may be "").
func (s *harmonySplitter) Push(chunk string) (reasoning, content string) {
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
	// strayText emits text that is not inside any message — one continuing run of it is one message, however it is chunked.
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
				return out[hmToReasoning].String(), out[hmToContent].String() // whitespace only so far: hold it
			}
			i, which, hold := scanMarkers(s.buf, hmTopMarkers)
			if i < 0 {
				// No marker. Text that is not a marker's beginning is content; a possible beginning waits. (Whitespace before
				// a marker is scaffolding and is dropped with it, so it must wait too — hence the TrimLeft test above.)
				if !strings.HasPrefix(tr, "<") || hold == 0 {
					// Trailing whitespace waits: it is scaffolding if a marker follows and text if the reply ends or goes on.
					keep := len(strings.TrimRight(s.buf[:len(s.buf)-hold], " \t\r\n"))
					strayText(s.buf[:keep])
					s.buf = s.buf[keep:]
				}
				return out[hmToReasoning].String(), out[hmToContent].String()
			}
			if lead := strings.TrimRight(s.buf[:i], " \t\r\n"); strings.TrimSpace(lead) != "" {
				strayText(lead) // stray text in front of a marker (the whitespace between it and the marker is scaffolding)
			}
			s.stray = false
			s.buf = s.buf[i+len(which):]
			switch which {
			case hmStart:
				s.state, s.header = hmHeader, ""
			case hmChannel:
				s.state, s.header = hmHeader, hmChannel
			default: // a terminator with no message open: nothing to end
			}
		case hmHeader:
			i := strings.Index(s.buf, hmMessage)
			if i < 0 {
				if len(s.header)+len(s.buf) > hmMaxHeader { // no <|message|> is coming: show what there is
					s.stray = false
					strayText(s.header + s.buf)
					s.header, s.buf, s.state = "", "", hmTop
					continue
				}
				_, _, hold := scanMarkers(s.buf, []string{hmMessage}) // a <|message|> may be split across two chunks
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
			s.buf = s.buf[i+len(which):]
			s.state = hmTop
		}
	}
}

// body routes one piece of a message body.
func (s *harmonySplitter) body(text string, emit func(hmRoute, string)) {
	if text == "" {
		return
	}
	if s.route == hmToCall {
		s.calls[len(s.calls)-1].Args += text
		return
	}
	emit(s.route, text)
}

// openMessage reads the header that ended at <|message|> and starts the message it introduces.
func (s *harmonySplitter) openMessage() {
	channel, recipient, constrain := parseHarmonyHeader(s.header)
	s.header = ""
	s.state, s.started, s.stray = hmBody, false, false
	switch {
	case recipient != "":
		s.route = hmToCall
		s.calls = append(s.calls, harmonyCall{Channel: channel, Recipient: recipient, Constrain: constrain})
	case channel == "analysis":
		s.route, s.sawReason = hmToReasoning, true
	default: // final, commentary preamble, or a channel this code has not heard of
		s.route = hmToContent
	}
}

// Flush ends the reply and returns what was still held. A header that never reached its <|message|> is scaffolding and is
// dropped; a body's held-back tail (a partial terminator that never completed) is text.
func (s *harmonySplitter) Flush() (reasoning, content string) {
	defer func() { s.buf, s.header = "", "" }()
	if s.state != hmBody && s.state != hmTop {
		return "", ""
	}
	if strings.TrimSpace(s.buf) == "" && !(s.state == hmTop && s.stray) { // whitespace alone is nothing — unless it follows stray text
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
	if route == hmToCall {
		s.calls[len(s.calls)-1].Args += text
		return "", ""
	}
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

// parseHarmonyHeader reads what sits between a message's start and its <|message|>: optionally `assistant`, then
// `<|channel|>NAME`, optionally ` to=RECIPIENT` (it may come before or after the channel) and `<|constrain|>TYPE`.
func parseHarmonyHeader(h string) (channel, recipient, constrain string) {
	if i := strings.Index(h, hmConstrain); i >= 0 {
		constrain = firstField(h[i+len(hmConstrain):])
		h = h[:i]
	}
	if i := strings.Index(h, hmChannel); i >= 0 {
		channel = firstField(h[i+len(hmChannel):])
	}
	if i := strings.Index(h, "to="); i >= 0 {
		recipient = firstField(h[i+len("to="):])
	}
	return channel, recipient, constrain
}

func firstField(s string) string {
	if i := strings.IndexAny(s, " \t\r\n<"); i >= 0 {
		s = s[:i]
	}
	return s
}

// scanMarkers finds the earliest occurrence of any marker in s and returns its index and text; with none it returns -1 and the
// length of the longest suffix of s that is a proper prefix of some marker (the bytes that must be held back).
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
