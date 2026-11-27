package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"

const reasoningBlobPrefix = streamframe.ReasoningBlobPrefix

type respStream = streamframe.RespStream

var (
	reasoningItemText    = streamframe.ReasoningItemText
	reasoningItemCarried = streamframe.ReasoningItemCarried
	outputMessage        = streamframe.OutputMessage
)

func newRespStream(ss *sseWriter, id string) *respStream {
	return streamframe.NewRespStream(ss, id, nil)
}

func withReasoning(rs *respStream, th *thinkOut, id string, mint *carriedMint, items []any) []any {
	var fn func(string) string
	if mint != nil {
		fn = mint.blob
	}
	return streamframe.WithReasoning(rs, th, id, fn, items)
}
