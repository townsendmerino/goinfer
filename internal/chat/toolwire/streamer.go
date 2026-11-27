package toolwire

import "strings"

// StreamableLen returns how many bytes of pending may be emitted immediately without
// risking that a later tool-call opener turns them into part of a call.
func StreamableLen(pending, opener string) int {
	if opener == "" {
		return len(pending)
	}
	if i := strings.Index(pending, opener); i >= 0 {
		return i
	}
	maxHold := min(len(opener)-1, len(pending))
	for k := maxHold; k > 0; k-- {
		if strings.HasSuffix(pending, opener[:k]) {
			return len(pending) - k
		}
	}
	return len(pending)
}

// ProseStreamer incrementally releases prose ahead of a tool-call opener while holding back partial opener prefixes.
type ProseStreamer struct {
	opener     string
	bareAware  bool
	fenceAware bool
	pending    strings.Builder
	started    bool
	done       bool
	held       bool
}

// NewProseStreamer returns a streamer for a family's tool-call opener.
func NewProseStreamer(opener string) *ProseStreamer { return &ProseStreamer{opener: opener} }

// NewBareAwareProseStreamer returns a streamer that also holds output starting with `{` for families accepting bare calls.
func NewBareAwareProseStreamer(opener string) *ProseStreamer {
	return &ProseStreamer{opener: opener, bareAware: true}
}

// FenceAware instructs the streamer to hold everything from the first code fence onward.
func (p *ProseStreamer) FenceAware() *ProseStreamer {
	p.fenceAware = true
	return p
}

// Done reports whether the tool-call opener has been encountered.
func (p *ProseStreamer) Done() bool { return p.done }

// Push feeds the next generated chunk and returns prose safe to emit immediately.
func (p *ProseStreamer) Push(chunk string) string {
	if p.done || p.held {
		return ""
	}
	p.pending.WriteString(chunk)
	buf := p.pending.String()
	if p.bareAware && !p.started {
		if t := strings.TrimLeft(buf, " \t\r\n"); t != "" && t[0] == '{' {
			p.held = true
			return ""
		}
	}

	n := StreamableLen(buf, p.opener)
	fenced := false
	if p.fenceAware && !strings.Contains(buf, p.opener) {
		if m := StreamableLen(buf, "```"); m < n || strings.Contains(buf, "```") {
			n = min(n, m)
			fenced = strings.Contains(buf, "```")
		}
	}
	safe, rest := buf[:n], buf[n:]
	if strings.Contains(buf, p.opener) {
		p.done = true
		rest = ""
	} else if fenced {
		p.held = true
		rest = ""
	}

	if !p.started {
		safe = strings.TrimLeft(safe, " \t\r\n")
		if safe != "" {
			p.started = true
		}
	}
	if p.done || p.held {
		safe = strings.TrimRight(safe, " \t\r\n")
	} else if tail := len(safe) - len(strings.TrimRight(safe, " \t\r\n")); tail > 0 {
		rest = safe[len(safe)-tail:] + rest
		safe = safe[:len(safe)-tail]
	}

	p.pending.Reset()
	p.pending.WriteString(rest)
	return safe
}
