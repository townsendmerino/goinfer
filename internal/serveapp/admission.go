package serveapp

import (
	"container/list"
	"context"
	"sync"
)

// admissionRecord is what a scheduler would read to decide ordering; admission keeps strict FIFO and never inspects
// it. promptIDs is this repo's session/prefix key: sessionLRU matches purely by longest common prefix over a
// request's prompt token ids against the resident sessions (bestExtend, sessions.go) and has no client-visible
// session id, so a prefix-aware scheduler would run that same match against the LRU at pick time.
type admissionRecord struct {
	promptIDs []int
	// id names the waiter for position() (a job's id: a page shows its own job's place in line). "" for requests
	// nothing will ask about; position() never matches them.
	id string
}

// turnWaiter is one FIFO entry: ready is closed exactly once, by admission itself, when this
// waiter is granted the turn.
type turnWaiter struct {
	ready chan struct{}
	rec   admissionRecord
}

// admission is a FIFO, context-aware turn-granter: up to cap holders at once, one by default (the zero value is ready
// to use and admits one; setCap widens it for -max-concurrent). It replaces a plain mutex, which cannot notice a
// waiter's client disconnecting, or a halt, until the waiter is granted the lock, by which point the dead request has
// held its place for nothing and delayed everyone behind it. enter drops out the instant ctx ends, granted or not.
//
// Strict FIFO: waiters are not reordered on anything in admissionRecord.
//
// Modeled on golang.org/x/sync/semaphore.Weighted's Acquire, specialized to weight 1. The mechanism is reused, not
// the package: a dependency is the wrong trade for a three-method primitive this repo can own and test directly (the
// work-queue task doc's rule: no new root module dependency).
type admission struct {
	mu      sync.Mutex
	held    int       // turns currently granted
	cap     int       // max concurrent holders; 0 ⇒ 1
	waiters list.List // of *turnWaiter, oldest (next in line) at the front
}

// setCap sets how many holders may run at once (at least 1). Call it before the model serves requests.
func (a *admission) setCap(n int) {
	a.mu.Lock()
	a.cap = max(1, n)
	a.mu.Unlock()
}

func (a *admission) capacity() int { return max(1, a.cap) }

// load is how many generations hold a turn or wait for one right now.
func (a *admission) load() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.held + a.waiters.Len()
}

// enter blocks until this waiter holds a turn or ctx ends first. The immediate-admit fast path (a turn free, nobody
// else waiting) neither allocates nor blocks. Whoever gets ok=true must call release exactly once when done (see
// loadedModel.exit). Turns are interchangeable, so no per-request closure is needed.
func (a *admission) enter(ctx context.Context, rec admissionRecord) (ok bool) {
	a.mu.Lock()
	if a.held < a.capacity() && a.waiters.Len() == 0 {
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
			// Granted the turn in the race against cancellation. Rather than fight the queue to
			// hand it back to exactly the right next waiter, accept the grant and release it
			// right back immediately: drive()/driveVL() will see this same ctx already done and
			// stop before running anything real (their own per-token check), so the cost is one
			// wasted, uncontested promotion — never a wasted generation.
			a.mu.Unlock()
			a.release()
		default:
			a.waiters.Remove(elem)
			a.mu.Unlock()
		}
		return false
	case <-w.ready:
		return true
	}
}

// position reports where the waiter with this id stands: its place among the waiters (1 = next to be granted the
// turn), how many are waiting in all, and whether one is running now. ok is false when no waiter has this id: it was
// never queued, has already been granted the turn, or dropped out. Order is exactly arrival order.
func (a *admission) position(id string) (place, waiting int, running, ok bool) {
	if id == "" {
		return 0, 0, false, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	i := 0
	for e := a.waiters.Front(); e != nil; e = e.Next() {
		i++
		if e.Value.(*turnWaiter).rec.id == id {
			place, ok = i, true
		}
	}
	return place, i, a.held > 0, ok
}

// release hands the turn to the next waiter (if any) or gives it back. The held count is unchanged across a
// direct hand-off (remove the front waiter, close its ready channel, done) rather than being
// decremented and re-incremented, so a concurrent enter() can never observe a free turn with no waiters in the gap
// and admit itself out of FIFO order ahead of an already-queued waiter.
func (a *admission) release() {
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
