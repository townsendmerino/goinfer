package decoder

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ErrClosed is what Generation.Err reports for a generation that Model.Close ended or that was started after Close. The
// channel of a refused generation is closed with no token.
var ErrClosed = errors.New("decoder: model is closed")

// modelLife tracks a Model's in-flight generations so Close can end them before it releases the weights and the device.
// The zero value is ready to use.
type modelLife struct {
	mu     sync.Mutex
	closed bool
	freed  bool // the weights and the device have been released
	next   uint64
	cancel map[uint64]context.CancelFunc
	wg     sync.WaitGroup
}

// lifeState is the model's modelLife, made on first use. It is a raw pointer, like resBusy, so Model stays copyable for the
// value-copy test seam; a copy shares the original's state, as it shares its weights.
func (m *Model) lifeState() *modelLife {
	if p := atomic.LoadPointer(&m.lifeP); p != nil {
		return (*modelLife)(p)
	}
	n := &modelLife{}
	if atomic.CompareAndSwapPointer(&m.lifeP, nil, unsafe.Pointer(n)) {
		return n
	}
	return (*modelLife)(atomic.LoadPointer(&m.lifeP))
}

// runGen runs body as one generation on its own goroutine: it owns closing out, hands body a context that Model.Close
// cancels, and counts the generation as in flight until the goroutine ends. A Model that is already closed refuses it:
// g.err is ErrClosed, out closes with no token, and refused (a scope the caller took before this call, if any) runs
// instead of body.
//
// A generation that Close cancels reports ErrClosed rather than context.Canceled, unless the caller's own ctx was
// already done.
func (m *Model) runGen(ctx context.Context, out chan int, g *Generation, refused func(), body func(ctx context.Context)) {
	l := m.lifeState()
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		if refused != nil {
			refused()
		}
		g.err = ErrClosed
		close(out)
		return
	}
	gctx, cancel := context.WithCancel(ctx)
	id := l.next
	l.next++
	if l.cancel == nil {
		l.cancel = map[uint64]context.CancelFunc{}
	}
	l.cancel[id] = cancel
	l.wg.Add(1)
	l.mu.Unlock()
	go func() {
		defer func() {
			l.mu.Lock()
			delete(l.cancel, id)
			l.mu.Unlock()
			cancel()
			l.wg.Done()
		}()
		defer close(out)
		body(gctx)
		if g.err != nil && errors.Is(g.err, context.Canceled) && ctx.Err() == nil {
			l.mu.Lock()
			closed := l.closed
			l.mu.Unlock()
			if closed {
				g.err = ErrClosed
			}
		}
	}()
}

// endGenerations marks the model closed, cancels every generation in flight and waits for each to finish, so nothing
// is running on the weights or the device when Close releases them. Later calls find it closed and only wait.
func (m *Model) endGenerations() {
	l := m.lifeState()
	l.mu.Lock()
	l.closed = true
	for _, c := range l.cancel {
		c()
	}
	l.mu.Unlock()
	l.wg.Wait()
}

// beginRelease reports whether the caller is the one that releases the model's resources: true once, false for every
// later Close.
func (l *modelLife) beginRelease() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.freed {
		return false
	}
	l.freed = true
	return true
}
