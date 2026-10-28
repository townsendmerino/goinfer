package scheduler

import (
	"context"
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
}
