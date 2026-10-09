//go:build gpu

package gpu

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// TestContextClose_lateWrapperClose is TestContextClose_finalizerRace's other half: a wrapper closed by the CALLER after its Context was closed holds the last reference on that device, so its Close
// destroys the device on the caller's goroutine -- and if the finalizer goroutine is destroying another closed device at that moment (a buffer a test leaked), wgpu-native deadlocks. releaseOwned
// sends that Close to the finalizer goroutine, where it is serialized with the rest. The planted defect (releaseOwnedHopOff) must hang this test.
//
// Per round: context A's matrix outlives A.Close; context B's matrix is leaked on purpose and B is closed; a collection queues the leak's finalizer; then A's matrix is closed explicitly.
func lateWrapperCloseRounds(t *testing.T, rounds int) error {
	leakExempt.Store(true) // the leaked matrix per round is the point of this test
	defer leakExempt.Store(false)
	for range rounds {
		a, err := New()
		if err != nil {
			return err
		}
		ma, err := a.UploadMatrix(make([]float32, 64), 8, 8)
		if err != nil {
			return err
		}
		a.Close() // ma now holds the last reference on a's device
		b, err := New()
		if err != nil {
			return err
		}
		if _, err := b.UploadMatrix(make([]float32, 64), 8, 8); err != nil { // leaked on purpose
			return err
		}
		b.Close()
		runtime.GC() // queues the leaked matrix's finalizer
		ma.Close()   // the caller's goroutine drops a's device while the finalizer goroutine drops b's
	}
	return nil
}

func runLateClose(t *testing.T, rounds int, limit time.Duration) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- lateWrapperCloseRounds(t, rounds) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(limit):
		t.Fatalf("deadlock: a wrapper's Close after its Context and a finalizer-driven device release were in wgpuDeviceRelease at once for %v", limit)
	}
}

func TestContextClose_lateWrapperClose(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no gpu: %v", err)
	} else {
		c.Close()
	}
	runLateClose(t, 150, 90*time.Second)
}

func TestContextClose_lateWrapperClose_plantedDefect(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no gpu: %v", err)
	} else {
		c.Close()
	}
	releaseOwnedHopOff = true
	defer func() { releaseOwnedHopOff = false }()
	// With the hop off the test must hang: the deadlocked goroutine cannot be cancelled, so this runs LAST in the package order and is skipped unless asked for.
	if testing.Short() || envOff("GOINFER_GPU_PLANTED") {
		t.Skip("the planted defect deadlocks the process by design: set GOINFER_GPU_PLANTED=1 and run it alone with a short -timeout")
	}
	runLateClose(t, 150, 60*time.Second)
}

func envOff(name string) bool { return os.Getenv(name) != "1" }
