package scheduler

import (
	"context"
	"sync"
	"testing"
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
