package serveapp

import (
	"context"
	"errors"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
)

// toolTurn is the outcome of one tool-bearing generation. /v1/chat/completions, /v1/responses and /v1/messages share
// runToolTurn (drive, buffer, parse, reconcile); each front end keeps only its wire format.
type toolTurn struct {
	raw   string          // everything the model generated
	calls []chat.ToolCall // parsed calls; empty when the model answered in prose
	lead  string          // the parser's prose before the first call (raw when there are no calls)
	// rest is the part of lead NOT yet handed to onProse: the front end sends it before closing its
	// prose and emitting the calls. On a family that cannot stream prose safely it is all of lead.
	rest                          string
	finish, stopSeq, cancelReason string
	nComp, reused                 int
	// fromReasoning: calls were read from the reasoning block, not the answer (settleCalls). Every route marks such a turn
	// with goinfer_tool_calls_from_reasoning.
	fromReasoning bool
}

// errNoForcedCall is a post-checked must-call turn (genRequest.callPostCheck) whose reply holds no tool call.
var errNoForcedCall = errors.New("tool_choice required a tool call and the model answered without one (this model's calls " +
	"are checked after generation, not constrained: forced_tool_call \"post-check\" in /v1/models)")

// errProseDiverged is codec.ErrProseDiverged: streamed prose that is not a prefix of the settled lead.
var errProseDiverged = codec.ErrProseDiverged

// runToolTurn runs one tool-bearing generation.
//
//   - ss non-nil: a heartbeat covers the silence before the first token and inside a call.
//   - onProse non-nil and the family can stream prose safely (Template.ToolCallOpener): prose goes out incrementally as
//     the model writes it, each chunk guaranteed to be a prefix of the lead the parser will compute. Every other family
//     keeps the buffered behaviour.
//
// A generation error is returned as the error. A cancelled turn returns with cancelReason set and no parse (the buffer
// is partial). errProseDiverged is returned alongside a filled toolTurn.
func (s *server) runToolTurn(ctx context.Context, lm *loadedModel, gr genRequest, tools []chat.Tool, ss *sseWriter, onProse func(string)) (toolTurn, error) {
	var prose *chat.ProseStreamer
	if onProse != nil && lm.tmpl != nil && !gr.callPostCheck { // a post-checked turn may yet be an error: no prose leaves before the parse
		if opener, ok := lm.tmpl.ToolCallOpener(); ok {
			// A family that also accepts a bare call must not stream an output that opens with '{'
			// — it may yet parse as a call with an empty lead (ParseToolCallsFor).
			if lm.tmpl.AcceptsBareToolCall() {
				prose = chat.NewBareAwareProseStreamer(opener)
			} else {
				prose = chat.NewProseStreamer(opener)
			}
			// With -lenient-tool-calls a fenced block may turn out to be the call, so nothing from the fence on may stream as prose.
			if lm.tmpl.LenientToolCalls() {
				prose.FenceAware()
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
		turn.lead = turn.raw // a partial buffer: never parse a call out of it
		return turn, nil
	}
	var parsedLead string
	turn.calls, parsedLead, turn.fromReasoning = settleCalls(lm.tmpl, turn.raw, tools, gr.think)
	var err error
	turn.lead, turn.rest, err = reconcileProse(turn.raw, len(turn.calls) > 0, parsedLead, streamed.String())
	if err == nil && gr.callPostCheck && len(turn.calls) == 0 {
		err = errNoForcedCall
	}
	return turn, err
}

// reconcileProse is codec.ReconcileProse: a finished turn's prose, and what is still to send after what streamed.
func reconcileProse(raw string, hasCalls bool, parsedLead, streamed string) (lead, rest string, err error) {
	return codec.ReconcileProse(raw, hasCalls, parsedLead, streamed)
}

// settleCalls is codec.SettleCalls with th's routed calls and reasoning (none for a nil router).
func settleCalls(tmpl *chat.Template, raw string, tools []chat.Tool, th *thinkOut) (calls []chat.ToolCall, lead string, fromReasoning bool) {
	return codec.SettleCalls(tmpl, raw, tools, th.toolCalls(), th.reasoningText())
}
