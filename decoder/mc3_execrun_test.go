package decoder

import (
	"runtime"
	"testing"
	"time"
)

// A batched step that panics fails its waiters either way; a programming bug (a runtime.Error) also crashes loudly, but only
// after the batcher is consistent again: busy clear and b.mu released, so a caller up the stack that recovers it inherits no
// held lock and no resident marked busy forever.
func TestExecRun_bugRepanicsAfterReleasingTheBatcher(t *testing.T) {
	newCoalescer := func(step func([]*batchReq)) *tokenCoalescer {
		b := &tokenCoalescer{lo: 1, hi: 8, wait: time.Millisecond, step: step, decoders: 1}
		b.init()
		return b
	}
	forward := func(b *tokenCoalescer, q *batchReq) (p any, err error) {
		defer func() { p = recover() }()
		return nil, b.forward(q)
	}

	var idx []int
	b := newCoalescer(func([]*batchReq) { _ = idx[3] })
	q := &batchReq{}
	p, _ := forward(b, q)
	if _, ok := p.(runtime.Error); !ok {
		t.Fatalf("a runtime.Error in a batched step: recovered %v, want it re-panicked", p)
	}
	if !b.mu.TryLock() {
		t.Fatal("b.mu is still held after the bug's panic")
	}
	busy := b.busy
	b.mu.Unlock()
	if busy || q.err == nil || !q.done {
		t.Fatalf("after the bug's panic: busy %v, waiter err %v done %v; want busy clear and the waiter failed", busy, q.err, q.done)
	}

	b = newCoalescer(func([]*batchReq) { panic("device lost") })
	q = &batchReq{}
	p, err := forward(b, q)
	if p != nil || err == nil {
		t.Fatalf("a non-bug panic in a batched step: err %v, panic %v; want an error and no panic", err, p)
	}
}
