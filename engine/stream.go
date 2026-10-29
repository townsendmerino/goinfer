package engine

import (
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
)

// Stream is one generation as Start returns it: its token ids, its report, and its close. Read Tokens to its end, then
// call Close once the stream is drained (a CPU session returns to its pool there); Report gives the generation's account
// as it stands when called, nil for an engine that keeps none. A nil *Stream has no tokens and no report.
type Stream struct {
	tokens <-chan int
	report func() *Report
	done   func()
	once   sync.Once
}

// NewStream is a Stream over tokens: report, when non-nil, gives its account (its result may be nil), and done, when
// non-nil, runs at the first Close.
func NewStream(tokens <-chan int, report func() *Report, done func()) *Stream {
	return &Stream{tokens: tokens, report: report, done: done}
}

// GenerationStream is a Stream over a decoder generation's tokens, its report read afresh from gen at each call (nil
// for a nil gen); done is as NewStream's.
func GenerationStream(tokens <-chan int, gen *decoder.Generation, done func()) *Stream {
	return NewStream(tokens, func() *Report { return reportOf(gen) }, done)
}

// Tokens is the generation's token ids; the channel closes when the generation ends.
func (s *Stream) Tokens() <-chan int {
	if s == nil {
		return nil
	}
	return s.tokens
}

// Report is the generation's account now, or nil when the engine keeps none.
func (s *Stream) Report() *Report {
	if s == nil || s.report == nil {
		return nil
	}
	return s.report()
}

// Close releases what the generation held, once however often it is called; call it after Tokens has closed.
func (s *Stream) Close() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		if s.done != nil {
			s.done()
		}
	})
}
