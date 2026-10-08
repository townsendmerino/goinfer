//go:build gpu

package gpu

import (
	"sync/atomic"
	"testing"
	"time"
)

// TestReleaseOwned pins the routing of a wrapper's release, with no device: a live Context (or none) releases in place; a closed one releases on the finalizer goroutine, exactly once, and a
// release that is itself running there (a Close that closes other wrappers) runs in place instead of waiting on itself.
func TestReleaseOwned(t *testing.T) {
	here := curGID()
	var nilCtx *Context
	ran := 0
	nilCtx.releaseOwned(func() { ran++ })
	live := &Context{}
	live.releaseOwned(func() { ran++ })
	if ran != 2 {
		t.Fatalf("a nil or live Context must release in place: ran %d of 2", ran)
	}

	closed := &Context{}
	closed.closed.Store(true)
	var onGID int64
	var nestedGID, nestedSerial int64
	var nestedElapsed time.Duration
	calls := atomic.Int32{}
	closed.releaseOwned(func() {
		calls.Add(1)
		onGID = curGID()
		t0 := time.Now()
		closed.releaseOwned(func() { nestedGID, nestedSerial = curGID(), serialGID.Load() }) // from inside: must not wait on itself
		nestedElapsed = time.Since(t0)
	})
	if calls.Load() != 1 {
		t.Fatalf("the release ran %d times, want exactly 1", calls.Load())
	}
	if onGID == here || onGID == 0 {
		t.Errorf("a closed Context's release ran on goroutine %d, the caller's own (%d): not on the finalizer goroutine", onGID, here)
	}
	if nestedGID != onGID || nestedSerial != onGID {
		t.Errorf("the nested release ran on goroutine %d (serial %d), want the serializing goroutine %d, in place", nestedGID, nestedSerial, onGID)
	}
	if nestedElapsed > 500*time.Millisecond {
		t.Errorf("the nested release took %v: it waited on the finalizer goroutine it was running on", nestedElapsed)
	}
	if serialGID.Load() != 0 {
		t.Errorf("serialGID left at %d after the release", serialGID.Load())
	}
}
