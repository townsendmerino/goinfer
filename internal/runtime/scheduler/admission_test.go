package scheduler

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestAdmission_capacityUnderConcurrentSetCap: Capacity reads the cap under the lock SetCap writes it under, so reading
// it while another goroutine sets it, and while requests enter and release, is race-free (run with -race).
func TestAdmission_capacityUnderConcurrentSetCap(t *testing.T) {
	var a Admission
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := range 1000 {
			a.SetCap(1 + i%4)
		}
	}()
	go func() {
		defer wg.Done()
		for range 1000 {
			if c := a.Capacity(); c < 1 || c > 4 {
				t.Errorf("Capacity = %d, want 1 to 4", c)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for range 1000 {
			if a.Enter(context.Background(), Record{}) {
				a.Release()
			}
		}
	}()
	wg.Wait()
	if a.Load() != 0 {
		t.Fatalf("Load = %d after every request released, want 0", a.Load())
	}
}

// A waiter whose context ends as Release hands it the turn either holds the turn, and gives it back, or leaves it to the next in
// line: the hand-off and the cancel decide under one lock, so no interleaving loses a turn. A lost turn is lost for good, and at
// capacity 1 the model then admits no one. The window is a few instructions wide, so this races the two many times (run with -race).
func TestAdmission_cancelDuringHandoffLosesNoTurn(t *testing.T) {
	rounds := 200_000
	if testing.Short() {
		rounds = 20_000
	}
	var a Admission
	for i := range rounds {
		if !a.Enter(context.Background(), Record{}) {
			t.Fatalf("round %d: the holder was refused a free turn", i)
		}
		ctx, cancel := context.WithCancel(context.Background())
		granted := make(chan bool)
		go func() { granted <- a.Enter(ctx, Record{}) }()
		for a.Load() < 2 { // the waiter is queued behind the holder
			runtime.Gosched()
		}
		go cancel()
		a.Release()
		if <-granted {
			a.Release()
		}
		cancel()
		if n := a.Load(); n != 0 {
			t.Fatalf("round %d: Load %d after the holder released and the waiter left or released: a turn was lost", i, n)
		}
	}
}

// queueWaiters queues one Enter per id behind a held turn, in order, and returns each one's result channel.
func queueWaiters(t *testing.T, a *Admission, ctx context.Context, ids ...string) map[string]chan bool {
	t.Helper()
	out := map[string]chan bool{}
	for _, id := range ids {
		ch := make(chan bool, 1)
		out[id] = ch
		n := a.Load()
		go func() { ch <- a.Enter(ctx, Record{ID: id}) }()
		for a.Load() == n { // queued before the next one starts, so arrival order is ids' order
			runtime.Gosched()
		}
	}
	return out
}

// Waiters are granted in arrival order, and Position reports each one's place among them (1 is next), how many wait, and
// whether a turn is held; an id nobody waits under, a granted one, or "" is not found, though a waiter with no id is queued.
// Whether a later waiter is still waiting is read from the queue (Position and Load, under its lock), not from its channel,
// whose send would race the waiter's wakeup.
func TestAdmission_fifoAndPosition(t *testing.T) {
	var a Admission
	if !a.Enter(context.Background(), Record{}) {
		t.Fatal("the holder was refused a free turn")
	}
	order := []string{"w1", "", "w2", "w3"} // "" is a waiter nothing asks about
	waiters := queueWaiters(t, &a, context.Background(), order...)
	if place, waiting, running, ok := a.Position("w2"); place != 3 || waiting != 4 || !running || !ok {
		t.Errorf("Position(w2) = (%d, %d, %v, %v), want (3, 4, true, true)", place, waiting, running, ok)
	}
	for _, id := range []string{"", "nobody"} {
		if _, _, _, ok := a.Position(id); ok {
			t.Errorf("Position(%q) found a waiter", id)
		}
	}
	for i, id := range order {
		a.Release() // the holder's turn, then each waiter's in turn
		select {
		case ok := <-waiters[id]:
			if !ok {
				t.Fatalf("%q was refused when its turn came", id)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("%q was not granted within 5 s of its turn", id)
		}
		rest := order[i+1:]
		if n := a.Load(); n != 1+len(rest) {
			t.Fatalf("after %q's grant: Load %d, want %d (its turn and %d still waiting)", id, n, 1+len(rest), len(rest))
		}
		for place, later := range rest {
			if later == "" {
				continue
			}
			if p, _, _, ok := a.Position(later); !ok || p != place+1 {
				t.Fatalf("after %q's grant, %s is at place %d (found %v), want still waiting at %d", id, later, p, ok, place+1)
			}
		}
		if id != "" {
			if _, _, _, ok := a.Position(id); ok {
				t.Errorf("Position(%s) found it after its grant", id)
			}
		}
	}
	a.Release()
	if a.Load() != 0 {
		t.Fatalf("Load %d after every holder released", a.Load())
	}
}
