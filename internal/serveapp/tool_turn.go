package serveapp

import (
	"context"
	"errors"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// toolTurn is the outcome of one tool-bearing generation. /v1/chat/completions, /v1/responses and
// /v1/messages each used to run this turn their own way — drive, buffer, parse, reconcile — and only
// the OpenAI route streamed prose while the model wrote it (G21); /v1/messages, the route Claude
// Code uses, sent nothing but heartbeats until the whole generation was done. runToolTurn is the one
// implementation; each front end keeps only its wire format.
type toolTurn struct {
	raw   string          // everything the model generated
	calls []chat.ToolCall // parsed calls; empty when the model answered in prose
	lead  string          // the parser's prose before the first call (raw when there are no calls)
	// rest is the part of lead NOT yet handed to onProse: the front end sends it before closing its
	// prose and emitting the calls. On a family that cannot stream prose safely it is all of lead.
	rest                          string
	finish, stopSeq, cancelReason string
	nComp, reused                 int
}

// errProseDiverged: bytes already streamed as prose are not a prefix of the prose the parser settled
// on. Unreachable by construction — the prose streamer only releases bytes that precede the family's
// call opener, and lead is the raw prefix before it — but streamed bytes cannot be recalled, so a
// front end that ever sees it must say so rather than emit a stream that disagrees with itself.
var errProseDiverged = errors.New("internal: streamed prose diverged from the parsed lead; the tool-call " +
	"stream for this family is not prefix-safe (G21)")

// runToolTurn runs one tool-bearing generation.
//
//   - ss non-nil: a heartbeat covers the silence before the first token and inside a call (G19).
//   - onProse non-nil and the family can stream prose safely (Template.ToolCallOpener): prose goes out
//     incrementally as the model writes it (G21), each chunk guaranteed to be a prefix of the lead the
//     parser will compute. Every other family keeps the buffered behaviour exactly.
//
// A generation error is returned as the error. A cancelled turn returns with cancelReason set and no
// parse (the buffer is partial). errProseDiverged is returned alongside a filled toolTurn.
func (s *server) runToolTurn(ctx context.Context, lm *loadedModel, gr genRequest, tools []chat.Tool, ss *sseWriter, onProse func(string)) (toolTurn, error) {
	var prose *chat.ProseStreamer
	if onProse != nil && lm.tmpl != nil {
		if opener, ok := lm.tmpl.ToolCallOpener(); ok {
			// A family that also accepts a bare call must not stream an output that opens with '{'
			// — it may yet parse as a call with an empty lead (ParseToolCallsFor).
			if lm.tmpl.AcceptsBareToolCall() {
				prose = chat.NewBareAwareProseStreamer(opener)
			} else {
				prose = chat.NewProseStreamer(opener)
			}
		}
	}
	var sb, streamed strings.Builder
	var stopBeat func()
	if ss != nil {
		stopBeat = sseHeartbeat(ss)
	}
	var turn toolTurn
	var gerr error
	turn.finish, turn.nComp, _, turn.stopSeq, turn.reused, turn.cancelReason, gerr = lm.drive(ctx, gr, s.gens, s.jobs, func(t string) {
		sb.WriteString(t)
		if prose == nil {
			return
		}
		if out := prose.Push(t); out != "" {
			streamed.WriteString(out)
			onProse(out)
		}
	})
	if stopBeat != nil {
		stopBeat() // joins the ticker goroutine before anything else writes to w
	}
	turn.raw = sb.String()
	if gerr != nil {
		return turn, gerr
	}
	if turn.cancelReason != "" {
		turn.lead = turn.raw // K1: a partial buffer — never parse a call out of it
		return turn, nil
	}
	var parsedLead string
	turn.calls, parsedLead = lm.tmpl.ParseToolCallsFor(turn.raw, tools)
	var err error
	turn.lead, turn.rest, err = reconcileProse(turn.raw, len(turn.calls) > 0, parsedLead, streamed.String())
	return turn, err
}

// reconcileProse settles a finished turn's prose against what already streamed. lead is the turn's
// prose for the final message: the parser's lead when there are calls, the raw output when there are
// none (as every route has always returned it). rest is what the front end still has to send so the
// streamed prose adds up to the turn's prose.
//
// What streamed is held against the PARSER's lead, never the raw output: the prose streamer trims the
// leading whitespace and withholds trailing whitespace exactly as every family's parser trims its
// lead (chat.ProseStreamer), so a no-call answer that opens with a newline streams "Hello" where the
// raw text is "\nHello". Checking that against the raw text — what the OpenAI route did before this
// was shared — reported a false divergence and ended the stream in an error. When nothing streamed
// (a family that cannot stream prose, or a non-streaming request) rest is lead itself, unchanged.
func reconcileProse(raw string, hasCalls bool, parsedLead, streamed string) (lead, rest string, err error) {
	lead = raw
	if hasCalls {
		lead = parsedLead
	}
	if streamed == "" {
		return lead, lead, nil
	}
	target := parsedLead // for a no-call turn this is raw as the parser trims it — what the streamer matches
	if !strings.HasPrefix(target, streamed) {
		return lead, "", errProseDiverged
	}
	return lead, target[len(streamed):], nil
}
