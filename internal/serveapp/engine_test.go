package serveapp

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// nilReportEngine streams ids on a stream with no report function at all, which the engine contract allows (a Stream's
// Report is then nil).
type nilReportEngine struct{ ids []int }

func (e nilReportEngine) Start(context.Context, engine.Request) *engine.Stream {
	ch := make(chan int, len(e.ids))
	for _, id := range e.ids {
		ch <- id
	}
	close(ch)
	return engine.NewStream(ch, nil, nil)
}

// TestDrive_engineWithNoReportFunc: drive reads the report through its Stream at every use, so an engine with a nil
// report function streams and finishes like the scripted seam (no logprobs, no reuse, no error) instead of panicking.
func TestDrive_engineWithNoReportFunc(t *testing.T) {
	lm := &loadedModel{tk: &tokenizer.Tokenizer{}, fakeEngine: nilReportEngine{}}
	finish, n, lps, stopHit, reused, cancelReason, err := lm.drive(context.Background(), genRequest{maxTokens: 4}, nil, nil, func(string) {})
	if finish != "stop" || n != 0 || lps != nil || stopHit != "" || reused != 0 || cancelReason != "" || err != nil {
		t.Fatalf("drive on a nil-report engine = (%q, %d, %v, %q, %d, %q, %v), want a clean stop and nothing else",
			finish, n, lps, stopHit, reused, cancelReason, err)
	}
}
