package scheduler

import (
	"context"
	"runtime"
	"testing"
)

// TestAdmit_haltWithNoReasonRefuses: a halt is the halted callback's ok, not a non-empty reason, so a halt with no reason
// still refuses after the grant and gives back both the turn and the queue slot.
func TestAdmit_haltWithNoReasonRefuses(t *testing.T) {
	var a Admission
	q := NewQueue(1, 2)
	ok, reason := Admit(context.Background(), &a, q, Record{}, func() (string, bool) { return "", true })
	if ok || reason != "" {
		t.Fatalf("Admit under a halt with no reason = (%v, %q), want (false, \"\")", ok, reason)
	}
	if a.Load() != 0 || len(q) != 0 {
		t.Fatalf("after the refusal: load %d, slots held %d, want 0 and 0", a.Load(), len(q))
	}
	if ok, _ := Admit(context.Background(), &a, q, Record{}, func() (string, bool) { return "", false }); !ok {
		t.Fatal("with no halt the next request must be admitted")
	}
	Exit(&a, q)
	if a.Load() != 0 || len(q) != 0 {
		t.Fatalf("after Exit: load %d, slots held %d, want 0 and 0", a.Load(), len(q))
	}
}

// TestNewQueue_unboundedAtZero: -max-queue 0 is no bound, which QueueFull never reports full.
func TestNewQueue_unboundedAtZero(t *testing.T) {
	if q := NewQueue(3, 0); q != nil || QueueFull(q) {
		t.Fatalf("NewQueue(3, 0) = %v (full %v), want nil and never full", q, QueueFull(q))
	}
	if q := NewQueue(3, 2); cap(q) != 5 || MaxQueued(q, 3) != 2 {
		t.Fatalf("NewQueue(3, 2): cap %d, MaxQueued %d, want 5 and 2", cap(q), MaxQueued(q, 3))
	}
	for _, turns := range []int{0, -2} { // counted as one turn, as Admission counts them
		if q := NewQueue(turns, 2); cap(q) != 3 || MaxQueued(q, turns) != 2 {
			t.Fatalf("NewQueue(%d, 2): cap %d, MaxQueued %d, want 3 and 2", turns, cap(q), MaxQueued(q, turns))
		}
	}
}

// Admit refuses at once on a full queue, without waiting for a turn, and a waiter whose context ends gives its queue slot
// back: neither refusal holds a slot or a turn.
func TestAdmit_refusalsGiveBackTheirSlot(t *testing.T) {
	var a Admission
	q := NewQueue(1, 1) // one running, one waiting
	if ok, _ := Admit(context.Background(), &a, q, Record{}, nil); !ok {
		t.Fatal("the first request was refused")
	}
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan bool, 1)
	go func() { ok, _ := Admit(ctx, &a, q, Record{}, nil); result <- ok }()
	for len(q) < 2 || a.Load() < 2 { // the second holds a slot and waits for the turn
		runtime.Gosched()
	}
	if ok, _ := Admit(context.Background(), &a, q, Record{}, nil); ok || !QueueFull(q) {
		t.Fatalf("a third request on a full queue: admitted %v, full %v; want refused at once", ok, QueueFull(q))
	}
	cancel()
	if <-result {
		t.Fatal("the cancelled waiter was admitted")
	}
	if len(q) != 1 || a.Load() != 1 {
		t.Fatalf("after the cancelled waiter left: %d slots held, load %d; want 1 and 1 (the running request's)", len(q), a.Load())
	}
	Exit(&a, q)
	if len(q) != 0 || a.Load() != 0 {
		t.Fatalf("after Exit: %d slots held, load %d", len(q), a.Load())
	}
}
