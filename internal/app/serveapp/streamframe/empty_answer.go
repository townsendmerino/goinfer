package streamframe

import "fmt"

// FromReasoningField is the extension field marking a turn whose tool calls were extracted from the reasoning block.
const FromReasoningField = "goinfer_tool_calls_from_reasoning"

// EmptyAnswer describes why a completed turn produced neither answer text nor tool calls.
type EmptyAnswer struct {
	Reason string `json:"reason"` // ended_in_reasoning | reasoning_only | stop_sequence | no_output
	Detail string `json:"detail"`
}

// TurnOutcome records answer presence, generated token IDs, and stop-string state across a generation turn.
type TurnOutcome struct {
	Answered bool
	IDs      []int
	Done     bool
	StopHit  bool
}

// EmptyAnswerOf classifies why a finished turn has no answer, returning nil when the turn answered, produced tool calls,
// or ended with a non-"stop" finish reason such as "length" or "cancelled".
func EmptyAnswerOf(outcome *TurnOutcome, th *ThinkOut, finish, stopHit string, calls, nComp int) *EmptyAnswer {
	if finish != "stop" || calls > 0 || outcome == nil || outcome.Answered {
		return nil
	}
	switch {
	case stopHit != "":
		return &EmptyAnswer{"stop_sequence", fmt.Sprintf("the stop sequence %q ended the turn before any answer text", stopHit)}
	case th != nil && th.EndedInReasoning():
		return &EmptyAnswer{"ended_in_reasoning", fmt.Sprintf("the model ended its turn inside its reasoning block, without closing it, after %d tokens: all of it is reasoning and there is no answer", nComp)}
	case th != nil && th.ReasoningLen() > 0:
		return &EmptyAnswer{"reasoning_only", fmt.Sprintf("the model closed its reasoning block and ended its turn without an answer (%d tokens)", nComp)}
	case nComp == 0:
		return &EmptyAnswer{"no_output", "the model ended its turn at its first token"}
	}
	return &EmptyAnswer{"no_output", fmt.Sprintf("the model ended its turn after %d tokens with no answer text", nComp)}
}

// MarkEmptyResponse stamps status="incomplete" and incomplete_details onto a /v1/responses object when e is non-nil.
func MarkEmptyResponse(resp map[string]any, e *EmptyAnswer) map[string]any {
	if e != nil {
		resp["status"] = "incomplete"
		resp["incomplete_details"] = map[string]any{"reason": e.Reason, "detail": e.Detail}
	}
	return resp
}
