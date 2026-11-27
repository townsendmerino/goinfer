package serveapp

import (
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

const fromReasoningField = streamframe.FromReasoningField

type (
	emptyAnswer = streamframe.EmptyAnswer
	turnOutcome = streamframe.TurnOutcome
)

var (
	markEmptyResponse = streamframe.MarkEmptyResponse
	finishChunk       = wirecore.FinishChunk
)

func emptyAnswerOf(gr genRequest, finish, stopHit string, calls, nComp int) *emptyAnswer {
	return streamframe.EmptyAnswerOf(gr.outcome, gr.think, finish, stopHit, calls, nComp)
}
