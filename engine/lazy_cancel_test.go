package engine

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestLazyTurn_cancelWithoutDrainingDoesNotLeak(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var d1Called atomic.Bool
	free := func(c context.Context, _ []int, _ int, _ decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		ch := make(chan int, 2)
		ch <- 1
		ch <- 2
		go func() {
			<-c.Done()
			close(ch)
		}()
		return ch, &decoder.Generation{}, func() { d1Called.Store(true) }, nil
	}
	constrained := func(context.Context, []int, int, decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		t.Error("phase 2 should not start after cancel")
		return nil, nil, nil, context.Canceled
	}
	lazy := toyLazy(t)
	st := lazyTurn(ctx, Request{PromptIDs: []int{1}, MaxTokens: 8}, lazy, free, constrained)
	// Give the goroutine a moment to block on out <- 1, then cancel without reading st.Tokens().
	time.Sleep(20 * time.Millisecond)
	cancel()
	deadline := time.After(2 * time.Second)
	for !d1Called.Load() {
		select {
		case <-deadline:
			t.Fatal("lazyTurn leaked goroutine and never ran d1() after cancel")
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}
	// Stream channel must also close promptly.
	select {
	case _, ok := <-st.Tokens():
		if ok {
			t.Fatal("expected closed stream after cancel")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not close after cancel")
	}
}
