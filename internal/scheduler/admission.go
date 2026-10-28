package scheduler

import (
	"container/list"
	"context"
	"sync"
)

// Record is what a waiter carries into admission. Admission keeps strict FIFO and never reads PromptIDs; ID names the
// waiter for Position ("" for requests nothing will ask about, which Position never matches).
type Record struct {
	PromptIDs []int
	ID        string
}

// turnWaiter is one FIFO entry: ready is closed exactly once, by admission itself, when this waiter is granted the
// turn.
type turnWaiter struct {
	ready chan struct{}
	rec   Record
}

// Admission is a FIFO, context-aware turn granter: up to Capacity holders at once, one by default (the zero value is
// ready to use and admits one). Enter drops out the instant its context ends, granted or not, so a cancelled or halted
// waiter does not hold its place. Waiters are never reordered.
type Admission struct {
	mu      sync.Mutex
	held    int       // turns currently granted
	cap     int       // max concurrent holders; 0 ⇒ 1
	waiters list.List // of *turnWaiter, oldest (next in line) at the front
}

// SetCap sets how many holders may run at once (at least 1). Raising it wakes no waiter, and lowering it takes hold only
// as the queue empties, since Release hands a turn straight to the next waiter: call it before serving.
func (a *Admission) SetCap(n int) {
	a.mu.Lock()
	a.cap = max(1, n)
	a.mu.Unlock()
}

// Capacity is the number of holders admitted at once.
func (a *Admission) Capacity() int { return max(1, a.cap) }

// Load is how many hold a turn or wait for one right now.
func (a *Admission) Load() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.held + a.waiters.Len()
}

// Enter blocks until this waiter holds a turn or ctx ends first. The immediate-admit fast path (a turn free, nobody
// waiting) neither allocates nor blocks. Whoever gets true must call Release exactly once when done.
func (a *Admission) Enter(ctx context.Context, rec Record) (ok bool) {
	a.mu.Lock()
	if a.held < a.Capacity() && a.waiters.Len() == 0 {
		a.held++
		a.mu.Unlock()
		return true
	}
	w := &turnWaiter{ready: make(chan struct{}), rec: rec}
	elem := a.waiters.PushBack(w)
	a.mu.Unlock()

	select {
	case <-ctx.Done():
		a.mu.Lock()
		select {
		case <-w.ready:
			// Granted in the race against cancellation: release it straight back. The caller sees ctx done and runs
			// nothing, so the cost is one uncontested promotion.
			a.mu.Unlock()
			a.Release()
		default:
			a.waiters.Remove(elem)
			a.mu.Unlock()
		}
		return false
	case <-w.ready:
		return true
	}
}

// Position reports where the waiter with this id stands: its place among the waiters (1 = next to be granted), how many
// wait in all, and whether a turn is held now. ok is false when no waiter has this id: it was never queued, has been
// granted, or left. Order is arrival order.
func (a *Admission) Position(id string) (place, waiting int, running, ok bool) {
	if id == "" {
		return 0, 0, false, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	i := 0
	for e := a.waiters.Front(); e != nil; e = e.Next() {
		i++
		if e.Value.(*turnWaiter).rec.ID == id {
			place, ok = i, true
		}
	}
	return place, i, a.held > 0, ok
}

// Release hands the turn to the front waiter, if any, or gives it back. A hand-off leaves the held count unchanged, so
// a concurrent Enter never sees a free turn with waiters queued and admits itself out of order.
func (a *Admission) Release() {
	a.mu.Lock()
	if front := a.waiters.Front(); front != nil {
		a.waiters.Remove(front)
		w := front.Value.(*turnWaiter)
		a.mu.Unlock()
		close(w.ready)
		return
	}
	a.held--
	a.mu.Unlock()
}
