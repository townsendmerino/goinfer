package scheduler

import "context"

// A model's queue bound is a channel of slots: one per request it holds, running or waiting. A nil queue is unbounded.

// NewQueue is the bound for turns running requests plus maxQueue waiting ones, or nil (unbounded) when maxQueue <= 0.
func NewQueue(turns, maxQueue int) chan struct{} {
	if maxQueue <= 0 {
		return nil
	}
	return make(chan struct{}, turns+maxQueue)
}

// QueueFull reports whether every slot of q is held; never for an unbounded q.
func QueueFull(q chan struct{}) bool { return q != nil && len(q) == cap(q) }

// MaxQueued is the -max-queue q was made with, behind turns running requests.
func MaxQueued(q chan struct{}, turns int) int { return cap(q) - max(1, turns) }

// Admit claims a slot of q, then waits for a turn of a. A full q refuses at once, as (false, ""), and so does a ctx that
// ends while waiting, with the slot given back. Once granted it asks halted, when non-nil: a halt gives the turn and the
// slot back and is returned as (false, reason). The halt is checked after the grant because one can land while the
// request waits. Whoever gets true calls Exit once.
func Admit(ctx context.Context, a *Admission, q chan struct{}, rec Record, halted func() (reason string, ok bool)) (bool, string) {
	if q != nil {
		select {
		case q <- struct{}{}:
		default:
			return false, ""
		}
	}
	if !a.Enter(ctx, rec) {
		if q != nil {
			<-q
		}
		return false, ""
	}
	if halted != nil {
		if reason, ok := halted(); ok {
			Exit(a, q)
			return false, reason
		}
	}
	return true, ""
}

// Exit gives back the turn, then the slot (the reverse of Admit's order).
func Exit(a *Admission, q chan struct{}) {
	a.Release()
	if q != nil {
		<-q
	}
}
