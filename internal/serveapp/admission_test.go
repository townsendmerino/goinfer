package serveapp

import (
	"context"
	"sync"
	"testing"
	"time"
)

// TestAdmission_waiterLeavesOnContextCancelWithoutWaitingForHolder is J1's own gate
// (task-work-queue-2026-09.md): "with -max-queue 1 and two concurrent requests, the second's
// cancellation is observed at the server within one decode step, not at the end of the first
// generation." Made deterministic and fast: no HTTP, no model — the defect is in the WAIT
// mechanism itself, so it is provable (and provably fixed) without either.
//
// A plain sync.Mutex fails this test: Lock() cannot be interrupted by a context, so a waiter
// behind an indefinitely-held mutex would not return until release() unblocks it — exactly the
// bug this type exists to fix. See TestAdmission_mutexWouldFailThisGate below for the same
// scenario run against a bare mutex, proving the test discriminates.
func TestAdmission_waiterLeavesOnContextCancelWithoutWaitingForHolder(t *testing.T) {
	var a admission
	if !a.enter(context.Background(), admissionRecord{}) {
		t.Fatal("first enter (nothing held, no waiters) should succeed immediately")
	}
	defer a.release()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool, 1)
	go func() { done <- a.enter(ctx, admissionRecord{}) }()

	// Give the goroutine time to actually reach the wait (queue as a waiter), not race the
	// PushBack. 20ms is generous relative to the assertion window below (200ms) and tiny relative
	// to what "the end of the first generation" would mean in production (seconds+).
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case ok := <-done:
		if ok {
			t.Fatal("expected enter to report not-admitted after its own context was cancelled")
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("enter did not return promptly after context cancellation — waiter blocked on the holder instead of its own ctx")
	}
}

// TestAdmission_mutexWouldFailThisGate is the "red without the fix" half: the exact same scenario
// as above, run against a bare sync.Mutex standing in for the pre-J1 mechanism. It must time out
// (Lock() has no way to observe ctx), proving the previous test's assertion actually discriminates
// the defect rather than passing by construction. This test is expected — and required — to hit
// its own timeout branch; it reports that as success, since observing the hang IS the proof.
func TestAdmission_mutexWouldFailThisGate(t *testing.T) {
	var mu sync.Mutex
	mu.Lock()

	ctx, cancel := context.WithCancel(context.Background())
	proceeded := make(chan struct{})
	go func() {
		mu.Lock() // the old mechanism: blocks with no way to notice ctx at all
		close(proceeded)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	_ = ctx // unused by a plain mutex — that IS the defect

	select {
	case <-proceeded:
		t.Fatal("a bare sync.Mutex somehow noticed context cancellation — this test's premise is wrong")
	case <-time.After(200 * time.Millisecond):
		// Expected: the mutex-blocked waiter is still stuck. Release the lock THIS goroutine took
		// at the top (unblocking the other one's Lock() above) so it doesn't leak past the test.
		mu.Unlock()
		<-proceeded
	}
}

// TestAdmission_fifoOrderPreserved: N waiters enter in submission order and are released one at a
// time in that same order — J1's own "changes no order" promise.
func TestAdmission_fifoOrderPreserved(t *testing.T) {
	var a admission
	if !a.enter(context.Background(), admissionRecord{}) {
		t.Fatal("first enter should succeed immediately")
	}

	const n = 5
	order := make(chan int, n)
	for i := range n {
		i := i
		go func() {
			if a.enter(context.Background(), admissionRecord{promptIDs: []int{i}}) {
				order <- i
			}
		}()
		time.Sleep(5 * time.Millisecond) // enqueue in a known order before releasing anything
	}

	a.release() // hand off to waiter 0
	for want := range n {
		select {
		case got := <-order:
			if got != want {
				t.Fatalf("FIFO order violated: got waiter %d, want %d", got, want)
			}
			a.release() // hand off to the next waiter (no-op once the list is empty)
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for waiter %d", want)
		}
	}
}

// TestAdmission_immediateAdmitFastPath: entering with nothing held and nobody waiting never
// blocks, regardless of ctx state — the common case must not pay any wait-mechanism cost.
func TestAdmission_immediateAdmitFastPath(t *testing.T) {
	var a admission
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already-cancelled context: must still succeed, since nothing is held
	if !a.enter(ctx, admissionRecord{}) {
		t.Fatal("immediate admit must not consult ctx at all when nothing is held and nobody waits")
	}
	a.release()
}
