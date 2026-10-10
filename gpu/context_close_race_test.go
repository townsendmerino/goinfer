//go:build gpu

package gpu

import (
	"runtime"
	"testing"
	"time"
)

// TestContextClose_finalizerRace reproduces an intermittent hang: a test leaks a buffer, its Context is closed, and the buffer's finalizer later drops the LAST reference on that device on
// the finalizer goroutine, concurrently with another Context's Close releasing its own device on the test goroutine. wgpu-native deadlocks when two devices are dropped at once. The loop
// leaks one matrix per round, closes its context, opens a second, forces the collection that queues the finalizer, and closes the second at once, under a deadline: a hang is a failure that
// names the cause, not a stuck package.
func TestContextClose_finalizerRace(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no gpu: %v", err)
	} else {
		c.Close()
	}
	const rounds = 150
	leakExempt.Store(true) // the leak per round is the point of this test
	defer leakExempt.Store(false)
	done := make(chan error, 1)
	go func() {
		for range rounds {
			a, err := New()
			if err != nil {
				done <- err
				return
			}
			if _, err := a.UploadMatrix(make([]float32, 64), 8, 8); err != nil { // leaked on purpose: never Closed
				done <- err
				return
			}
			a.Close()
			b, err := New()
			if err != nil {
				done <- err
				return
			}
			runtime.GC() // queues the leaked buffer's finalizer
			b.Close()
		}
		done <- nil
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(90 * time.Second):
		t.Fatal("deadlock: a Context.Close and a finalizer-driven device release were in wgpuDeviceRelease at once for 90 s")
	}
}
