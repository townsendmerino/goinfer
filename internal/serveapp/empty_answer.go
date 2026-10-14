package serveapp

import "fmt"

// emptyAnswer is why a finished turn carries no answer: no answer text and no tool call. Every chat route says so rather
// than return an empty reply that looks like a clean one (docs/server.md, "Empty answers say why"): /v1/responses as status
// "incomplete" with incomplete_details, /v1/chat/completions and /v1/messages in a goinfer_empty_answer object.
type emptyAnswer struct {
	Reason string `json:"reason"` // ended_in_reasoning | reasoning_only | stop_sequence | no_output
	Detail string `json:"detail"`
}

// turnOutcome is what streamTokens records about a generation's answer as it runs. prepare gives every request one.
type turnOutcome struct {
	answered bool // answer text other than whitespace reached onText
}

// emptyAnswerOf is the reason a finished turn has no answer, or nil when it has answer text or a tool call, or when its
// finish already says why (length: the token limit; cancelled). calls is the number of tool calls the turn returns.
func emptyAnswerOf(gr genRequest, finish, stopHit string, calls, nComp int) *emptyAnswer {
	if finish != "stop" || calls > 0 || gr.outcome == nil || gr.outcome.answered {
		return nil
	}
	th := gr.think
	switch {
	case stopHit != "":
		return &emptyAnswer{"stop_sequence", fmt.Sprintf("the stop sequence %q ended the turn before any answer text", stopHit)}
	case th != nil && th.endedInReasoning:
		return &emptyAnswer{"ended_in_reasoning", fmt.Sprintf("the model ended its turn inside its reasoning block, without closing it, after %d tokens: all of it is reasoning and there is no answer", nComp)}
	case th != nil && th.reasoning.Len() > 0:
		return &emptyAnswer{"reasoning_only", fmt.Sprintf("the model closed its reasoning block and ended its turn without an answer (%d tokens)", nComp)}
	case nComp == 0:
		return &emptyAnswer{"no_output", "the model ended its turn at its first token"}
	}
	return &emptyAnswer{"no_output", fmt.Sprintf("the model ended its turn after %d tokens with no answer text", nComp)}
}

// markEmptyResponse records e on a Responses object: an empty answer is "incomplete", and incomplete_details names why.
func markEmptyResponse(resp map[string]any, e *emptyAnswer) map[string]any {
	if e != nil {
		resp["status"] = "incomplete"
		resp["incomplete_details"] = map[string]any{"reason": e.Reason, "detail": e.Detail}
	}
	return resp
}

// finishChunk is a chat stream's finish_reason chunk, carrying e as goinfer_empty_answer when the turn has no answer.
func finishChunk(id string, created int64, model, finish string, e *emptyAnswer) map[string]any {
	c := chatChunk(id, created, model, delta{}, &finish)
	if e != nil {
		c["choices"].([]any)[0].(map[string]any)["goinfer_empty_answer"] = e
	}
	return c
}
