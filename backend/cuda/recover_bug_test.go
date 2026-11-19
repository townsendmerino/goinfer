//go:build cuda

package cuda

import (
	"errors"
	"runtime"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// mustPanicWithBug runs f and returns the runtime.Error it panicked with, failing the test when it returned instead.
func mustPanicWithBug(t *testing.T, what string, f func()) (bug runtime.Error) {
	t.Helper()
	defer func() {
		p := recover()
		if p == nil {
			t.Fatalf("%s returned: a programming bug was absorbed, where it must panic", what)
		}
		var ok bool
		if bug, ok = p.(runtime.Error); !ok {
			t.Fatalf("%s panicked with %T %v, want the runtime.Error itself", what, p, p)
		}
	}()
	f()
	return nil
}

// A nil dereference inside BuildResident is a bug, not a device that declined: it panics out of the build instead of becoming a
// decline that sends the model to the CPU path.
func TestBuildResident_aBugPanicsInsteadOfDeclining(t *testing.T) {
	mustPanicWithBug(t, "BuildResident on a nil model", func() {
		_, _, _ = (&cudaBackend{}).buildResident(nil, true)
	})
}

// newTestExecutor is a cudaResident with only its pinned executor: the loop newResident starts.
func newTestExecutor(t *testing.T) *cudaResident {
	r := &cudaResident{reqCh: make(chan func() error), ackCh: make(chan error)}
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range r.reqCh {
			r.ackCh <- runJob(j)
		}
	}()
	t.Cleanup(func() { close(r.reqCh) })
	return r
}

// A bug on the executor goroutine is raised again on the goroutine that called do, with the executor's stack in its message, and the
// executor keeps running the next job.
func TestDo_aBugOnTheExecutorPanicsOnTheCaller(t *testing.T) {
	r := newTestExecutor(t)
	bug := mustPanicWithBug(t, "do with a job that dereferences nil", func() {
		_ = r.do(func() error {
			var z *cudaResident
			return z.launchErr
		})
	})
	if msg := bug.Error(); !strings.Contains(msg, "executor goroutine stack") || !strings.Contains(msg, "runJob") {
		t.Errorf("the re-raised bug does not carry the executor's stack: %s", msg)
	}
	if mem.DeviceAllocFailed(bug) {
		t.Error("mem.DeviceAllocFailed takes an executor bug for an allocation failure")
	}
	if err := r.do(func() error { return nil }); err != nil {
		t.Errorf("the executor's next job: %v, want it to run", err)
	}
}

// A typed allocation failure on the executor stays an error the prefill decline can classify, with its type, and does not panic.
func TestDo_anAllocFailureIsAnErrorThatStillClassifies(t *testing.T) {
	r := newTestExecutor(t)
	err := r.do(func() error { panic(allocFailureForTest()) })
	var f *gpu.AllocFailure
	if !errors.As(err, &f) || !mem.DeviceAllocFailed(err) {
		t.Errorf("do's error %v: errors.As finds the AllocFailure %v, DeviceAllocFailed %v; want both", err, f != nil, mem.DeviceAllocFailed(err))
	}
}

// A KV pool allocation that fails with the allocator's typed failure still declines: ok=false, no error.
func TestKVTryAllocPools_aTypedAllocFailureDeclines(t *testing.T) {
	kvSlotAllocFailAtForTest = 1
	defer func() { kvSlotAllocFailAtForTest = 0 }()
	r := &cudaResident{}
	ok, err := r.kvTryAllocPools(2)
	if ok || err != nil {
		t.Errorf("kvTryAllocPools at a failing allocation: ok=%v err=%v, want a decline (false, nil)", ok, err)
	}
}
