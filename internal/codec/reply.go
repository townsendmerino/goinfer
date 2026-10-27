package codec

import (
	"errors"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// NewReplySplitter is the splitter that separates a reply to a prompt rendered by tm from turns into reasoning and answer;
// nil when tm has no reasoning spec, and the caller then passes every fragment through as answer.
func NewReplySplitter(tm *chat.Template, turns []chat.Turn) *chat.ReplySplitter {
	return tm.NewReplySplitter(turns)
}

// SettleCalls reads a finished turn's tool calls and the prose before them, trimmed as the parsers return it.
//
// Every family but gpt-oss writes its call inline, so the template's parser finds it in raw. A Harmony call is a message
// addressed to a function, which the reply router routes out of both the reasoning and the answer: routed is what the
// router read (ReplySplitter.ToolCalls), nil without one. A turn with no answer at all is read from its reasoning, the
// whole of what the router routed there ("" without one): a thinking model can write its call inside the block and end
// its turn there. A call drafted in the reasoning of a turn that went on to answer in prose is not taken. fromReasoning
// reports a call read that way.
func SettleCalls(tm *chat.Template, raw string, tools []chat.Tool, routed []chat.ToolCall, reasoning string) (calls []chat.ToolCall, lead string, fromReasoning bool) {
	calls, lead = tm.ParseToolCallsFor(raw, tools)
	if len(calls) == 0 {
		if len(routed) > 0 {
			return routed, strings.TrimSpace(raw), false
		}
		if strings.TrimSpace(raw) == "" && reasoning != "" {
			if inBlock, _ := tm.ParseToolCallsFor(reasoning, tools); len(inBlock) > 0 {
				return inBlock, "", true
			}
		}
	}
	return calls, lead, false
}

// ErrProseDiverged: bytes already streamed as prose are not a prefix of the prose the parser settled on. Unreachable by
// construction (chat.ProseStreamer releases only bytes that precede the family's call opener, and the lead is the raw
// prefix before it), but streamed bytes cannot be recalled, so a front end that ever sees it must say so rather than emit
// a stream that disagrees with itself.
var ErrProseDiverged = errors.New("internal: streamed prose diverged from the parsed lead; the tool-call " +
	"stream for this family is not prefix-safe (G21)")

// ReconcileProse settles a finished turn's prose against what already streamed. lead is the turn's prose for the final
// message: the parser's lead when there are calls, the raw output when there are none. rest is what the front end still
// has to send so the streamed prose adds up to the turn's prose.
//
// What streamed is held against the parser's lead, never the raw output: the prose streamer trims leading whitespace and
// withholds trailing whitespace exactly as every family's parser trims its lead, so a no-call answer that opens with a
// newline streams "Hello" where the raw text is "\nHello". When nothing streamed (a family that cannot stream prose, or a
// non-streaming request) rest is lead itself.
func ReconcileProse(raw string, hasCalls bool, parsedLead, streamed string) (lead, rest string, err error) {
	lead = raw
	if hasCalls {
		lead = parsedLead
	}
	if streamed == "" {
		return lead, lead, nil
	}
	target := parsedLead // for a no-call turn this is raw as the parser trims it: what the streamer matches
	if !strings.HasPrefix(target, streamed) {
		return lead, "", ErrProseDiverged
	}
	return lead, target[len(streamed):], nil
}
