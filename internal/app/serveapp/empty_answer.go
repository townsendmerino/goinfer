package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"

const fromReasoningField = streamframe.FromReasoningField

type (
	emptyAnswer = streamframe.EmptyAnswer
	turnOutcome = streamframe.TurnOutcome
)

var markEmptyResponse = streamframe.MarkEmptyResponse

func emptyAnswerOf(gr genRequest, finish, stopHit string, calls, nComp int) *emptyAnswer {
	return streamframe.EmptyAnswerOf(gr.outcome, gr.think, finish, stopHit, calls, nComp)
}

func finishChunk(id string, created int64, model, finish string, e *emptyAnswer, fromReasoning bool) map[string]any {
	c := chatChunk(id, created, model, delta{}, &finish)
	ch := c["choices"].([]any)[0].(map[string]any)
	if e != nil {
		ch["goinfer_empty_answer"] = e
	}
	if fromReasoning {
		ch[fromReasoningField] = true
	}
	return c
}
