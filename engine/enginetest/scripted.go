package enginetest

import (
	"context"

	"github.com/townsendmerino/goinfer/engine"
)

// Scripted is an Engine that streams fixed token ids, at most each request's MaxTokens, and keeps no report: a fake
// for testing a serving shell with no model. OnStart, when set, receives each request before its stream starts.
type Scripted struct {
	IDs     []int
	OnStart func(engine.Request)
}

// Start streams s.IDs, cut to req.MaxTokens (none when it is 0 or less), on a closed, buffered channel.
func (s Scripted) Start(_ context.Context, req engine.Request) *engine.Stream {
	if s.OnStart != nil {
		s.OnStart(req)
	}
	ids := s.IDs
	if n := max(req.MaxTokens, 0); len(ids) > n {
		ids = ids[:n]
	}
	ch := make(chan int, len(ids))
	for _, id := range ids {
		ch <- id
	}
	close(ch)
	return engine.NewStream(ch, nil, nil)
}
