package serveapp

import (
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// idsFormat is the closed list of differences between a reply's own bytes and the template's render of its turn that are
// template-owned formatting, not content. A carried turn whose bytes differ from the render only by these keeps its ids, the
// model's bytes taking the place of the template's for that span; any other difference falls back.
//
//  1. A run of newlines next to a reasoning delimiter (the template writes "\n" inside the block and strips the reasoning's own).
//  2. JSON whitespace inside a tool-call block, outside string literals and next to a structural character ({}[]:,): the
//     template writes its own separators around the call object.
//
// Each side's run may be empty. Whitespace inside a JSON string, inside the reasoning or answer text, or anywhere else is
// content. Evidence: ids-carried-2026-10-15.md, "G-ids-3: graded".
type idsFormat struct {
	rOpen, rClose string // the reasoning delimiters, "" when the template has none
	cOpen, cClose string // the tool-call block's markers, "" when the family has no JSON call form
}

func newIDsFormat(tm *chat.Template) idsFormat {
	var f idsFormat
	if r := tm.Reasoning(); r != nil {
		f.rOpen, f.rClose = r.Open(), r.Close()
	}
	if pre, suf, _, _, ok := tm.ToolCallWrapper(); ok && strings.TrimSpace(pre) != "" && strings.TrimSpace(suf) != "" {
		f.cOpen, f.cClose = strings.TrimSpace(pre), strings.TrimSpace(suf)
	}
	return f
}

// align walks d (the reply's bytes) against t from t's start. ok reports that d is consumed with every difference on the closed
// list; end is where d ends in t, at[i] is the byte of t that d[i] stands for (at[len(d)] == end), and differs reports that
// some difference was skipped. The first bytes must match: a difference before them would take in the template's bytes
// ahead of the reply (the newline after the assistant header, next to an opening delimiter).
func (f idsFormat) align(d, t string) (end int, at []int, differs, ok bool) {
	at = make([]int, len(d)+1)
	i, j := 0, 0
	inCall, inStr, esc := false, false, false
	for i < len(d) {
		if j < len(t) && d[i] == t[j] {
			at[i] = j
			if c := d[i]; inStr {
				switch {
				case esc:
					esc = false
				case c == '\\':
					esc = true
				case c == '"':
					inStr = false
				}
			} else if inCall && c == '"' {
				inStr = true
			}
			i, j = i+1, j+1
			if !inStr && f.cOpen != "" {
				if strings.HasSuffix(d[:i], f.cOpen) {
					inCall = true
				} else if inCall && strings.HasSuffix(d[:i], f.cClose) {
					inCall = false
				}
			}
			continue
		}
		if i == 0 {
			return 0, nil, false, false
		}
		var wd, wt int
		switch {
		case inCall && !inStr:
			wd, wt = wsRun(d, i, " \t\r\n"), wsRun(t, j, " \t\r\n")
			if (wd == 0 && wt == 0) || !(jsonStructural(prevNonWS(d, i)) || jsonStructural(nextByte(d, i+wd))) {
				return 0, nil, false, false
			}
		case !inStr && f.rOpen != "":
			wd, wt = wsRun(d, i, "\n"), wsRun(t, j, "\n")
			if (wd == 0 && wt == 0) || !f.nearReasoningTag(d, i, wd) || !f.nearReasoningTag(t, j, wt) {
				return 0, nil, false, false
			}
		default:
			return 0, nil, false, false
		}
		for k := i; k < i+wd; k++ {
			at[k] = j
		}
		i, j, differs = i+wd, j+wt, true
	}
	at[len(d)] = j
	return j, at, differs, true
}

// nearReasoningTag reports whether the newline run s[i:i+w], with the newlines just before it, touches a reasoning delimiter.
func (f idsFormat) nearReasoningTag(s string, i, w int) bool {
	lo := i
	for lo > 0 && s[lo-1] == '\n' {
		lo--
	}
	before, after := s[:lo], s[i+w:]
	for _, tag := range []string{f.rOpen, f.rClose} {
		if strings.HasSuffix(before, tag) || strings.HasPrefix(after, tag) {
			return true
		}
	}
	return false
}

func wsRun(s string, i int, set string) int {
	n := 0
	for i+n < len(s) && strings.IndexByte(set, s[i+n]) >= 0 {
		n++
	}
	return n
}

func prevNonWS(s string, i int) byte {
	for i--; i >= 0; i-- {
		if strings.IndexByte(" \t\r\n", s[i]) < 0 {
			return s[i]
		}
	}
	return 0
}

func nextByte(s string, i int) byte {
	if i < len(s) {
		return s[i]
	}
	return 0
}

func jsonStructural(c byte) bool { return c != 0 && strings.IndexByte("{}[]:,", c) >= 0 }

// alignedAt is aligned for a reply whose bytes stand at at[] in the rendered text: each special piece on exactly one special
// segment, and no ordinary run over a special segment.
func (p *segPlan) alignedAt(at []int, ps []idsPiece) bool {
	x := 0
	for _, pc := range ps {
		a, b := at[x], at[x+len(pc.text)]
		if pc.special {
			i := p.segAt(a)
			if p.starts[i] != a || p.starts[i+1] != b || !p.segs[i].Special {
				return false
			}
		} else if b > a {
			for i := p.segAt(a); i < len(p.segs) && p.starts[i] < b; i++ {
				if p.segs[i].Special {
					return false
				}
			}
		}
		x += len(pc.text)
	}
	return true
}
