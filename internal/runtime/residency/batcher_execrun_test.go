package residency

import (
	"runtime"
	"testing"
	"time"
)

// A batched step that panics fails its waiters either way; a programming bug (a runtime.Error) also crashes loudly, but only
// after the batcher is consistent again: busy clear and b.mu released, so a caller up the stack that recovers it inherits no
// held lock and no resident marked busy forever.
func TestExecRun_bugRepanicsAfterReleasingTheBatcher(t *testing.T) {
	newCoalescer := func(step func([]*BatchReq)) *TokenCoalescer {
		b := &TokenCoalescer{Lo: 1, Hi: 8, Wait: time.Millisecond, Step: step, decoders: 1}
		b.Init()
		return b
	}
	forward := func(b *TokenCoalescer, q *BatchReq) (p any, err error) {
		defer func() { p = recover() }()
		return nil, b.Forward(q)
	}

	var idx []int
	b := newCoalescer(func([]*BatchReq) { _ = idx[3] })
	q := &BatchReq{}
	p, _ := forward(b, q)
	if _, ok := p.(runtime.Error); !ok {
		t.Fatalf("a runtime.Error in a batched Step: recovered %v, want it re-panicked", p)
	}
	if !b.mu.TryLock() {
		t.Fatal("b.mu is still held after the bug's panic")
	}
	busy := b.busy
	b.mu.Unlock()
	if busy || q.Err == nil || !q.done {
		t.Fatalf("after the bug's panic: busy %v, waiter err %v done %v; want busy clear and the waiter failed", busy, q.Err, q.done)
	}

	b = newCoalescer(func([]*BatchReq) { panic("device lost") })
	q = &BatchReq{}
	p, err := forward(b, q)
	if p != nil || err == nil {
		t.Fatalf("a non-bug panic in a batched Step: err %v, panic %v; want an error and no panic", err, p)
	}
}
