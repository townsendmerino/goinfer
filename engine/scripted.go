package engine

import "context"

// Scripted is an Engine that streams fixed token ids, at most each request's MaxTokens, and keeps no report: a fake
// for testing a serving shell with no model. OnStart, when set, receives each request before its stream starts.
type Scripted struct {
	IDs     []int
	OnStart func(Request)
}

// Start streams s.IDs, cut to req.MaxTokens, on a closed, buffered channel.
func (s Scripted) Start(_ context.Context, req Request) (<-chan int, func() *Report, func()) {
	if s.OnStart != nil {
		s.OnStart(req)
	}
	ids := s.IDs
	if len(ids) > req.MaxTokens {
		ids = ids[:req.MaxTokens]
	}
	ch := make(chan int, len(ids))
	for _, id := range ids {
		ch <- id
	}
	close(ch)
	return ch, func() *Report { return nil }, func() {}
}
