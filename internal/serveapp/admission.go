package serveapp

import (
	"container/list"
	"context"
	"sync"
)

// admissionRecord is what a future scheduler reads to decide ordering — J1 keeps strict FIFO and
// never inspects this itself (task-work-queue-2026-09.md: "J1 itself keeps strict FIFO — it
// establishes the structure and changes no order"). promptIDs is this repo's own "session/prefix
// key": sessionLRU (sessions.go's bestExtend) has no client-visible session id at all — it matches
// purely by longest-common-prefix over a request's actual prompt token ids against the sessions
// currently resident — so the ids themselves are what a later prefix-aware scheduler (J6) would
// need to run that same match against the LRU at pick time.
type admissionRecord struct {
	promptIDs []int
	// id names the waiter for position() — a job's id (W28: a page shows its own job's place in line).
	// "" for requests nothing will ever ask about; position() simply never matches them.
	id string
}

// turnWaiter is one FIFO entry: ready is closed exactly once, by admission itself, when this
// waiter is granted the turn.
type turnWaiter struct {
	ready chan struct{}
	rec   admissionRecord
}

// admission is a size-1, FIFO, context-aware turn-granter (J1, task-work-queue-2026-09.md),
// replacing loadedModel's plain sync.Mutex. A mutex has no notion of context: a waiter blocked on
// Lock() cannot notice its own client disconnecting, or a K2 halt landing, until it is actually
// GRANTED the lock — at which point it has already occupied its place in line for nothing, and
// everything behind it waits that much longer. admission.enter drops out the instant ctx ends,
// whether or not the turn has been granted yet, so a dead request never delays a live one.
//
// FIFO order only; it does not reorder waiters on anything in admissionRecord — that is left for
// J6 (prefix-aware scheduling) and J7 (classes/priorities) to build on top of, not decided here.
//
// Modeled on golang.org/x/sync/semaphore.Weighted's Acquire, specialized to weight-1 (one holder
// at a time — ground rule: "one decode worker per model stays," task-work-queue-2026-09.md's own
// "Ground rules" §1). The MECHANISM is reused, not the package: pulling in a dependency for one
// specialized case would be the wrong trade for a three-method primitive this repo can own and
// test directly (task-work-queue-2026-09.md's own ground rule 4: "no new root module dependency").
//
// Zero value is ready to use — no constructor needed.
type admission struct {
	mu      sync.Mutex
	held    bool
	waiters list.List // of *turnWaiter, oldest (next in line) at the front
}

// enter blocks until this waiter holds the turn or ctx ends first. The immediate-admit fast path
// (nothing held, nobody else waiting) never allocates or blocks. release() must be called exactly
// once by whoever gets ok=true, when done running — see loadedModel.exit(), which calls it
// unconditionally for the request that currently holds the turn (there is only ever one, by
// construction, so no per-request closure is needed here).
func (a *admission) enter(ctx context.Context, rec admissionRecord) (ok bool) {
	a.mu.Lock()
	if !a.held && a.waiters.Len() == 0 {
		a.held = true
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

// position reports where the waiter with this id stands: its place among the waiters (1 = next to be
// granted the turn), how many are waiting in all, and whether one is running now. ok is false when no
// waiter has this id — it was never queued, has already been granted the turn, or dropped out. Order is
// exactly arrival order: J1 is strict FIFO, and J6's reordering was measured and not shipped.
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
	return place, i, a.held, ok
}

// release hands the turn to the next waiter (if any) or clears held. Held stays true across a
// direct hand-off (remove the front waiter, close its ready channel, done) rather than being
// cleared and re-set, so a concurrent enter() can never observe (not held, no waiters) in the gap
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
	a.held = false
	a.mu.Unlock()
}
