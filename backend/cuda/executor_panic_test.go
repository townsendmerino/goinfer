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

// Audit C-24 / C-25 — the executor goroutine must survive a panicking job, and the expert-cache
// sizing must not divide by zero on a dense-prefix MoE.
//
// Both are code the design says should DECLINE and that instead kills the process, on the pinned executor
// goroutine where no caller's `defer recover()` can reach (BuildResident's recover runs on the CALLING
// goroutine and cannot catch it). C-24: `gpu.NewBufferLenOf` panics on OOM per its own contract, and
// prefillCore allocates hundreds of MB for a long prompt, so a long prompt on a nearly-full card must fall
// back to the sequential path rather than kill serve. C-25: GLM/DeepSeek/Kimi put dense layers first, so
// layer 0 has no expert strides and `budget / len(moeLayers) / perLayer` divides by zero.

// TestRunJob_recoversPanic is the C-24 gate: a panicking job becomes an error, not a dead process.
func TestRunJob_recoversPanic(t *testing.T) {
	err := runJob(func() error { panic("simulated device OOM") })
	if err == nil {
		t.Fatal("runJob returned nil for a panicking job — the panic escaped to the executor " +
			"goroutine, which kills the process instead of declining")
	}
	if !strings.Contains(err.Error(), "simulated device OOM") {
		t.Errorf("recovered error drops the panic value (undiagnosable): %v", err)
	}
	if !strings.Contains(err.Error(), "panicked") {
		t.Errorf("recovered error should be identifiable as a panic so prefillCore can turn it "+
			"into a decline: %v", err)
	}
}

// TestRunJob_passesThroughNormalResults: the boundary must be transparent otherwise — a recover
// wrapper that swallowed or rewrote ordinary errors would hide every real device failure.
func TestRunJob_passesThroughNormalResults(t *testing.T) {
	if err := runJob(func() error { return nil }); err != nil {
		t.Errorf("runJob altered a nil result: %v", err)
	}
	sentinel := errPrefillDeclined
	if err := runJob(func() error { return sentinel }); err != sentinel {
		t.Errorf("runJob rewrote an ordinary error %v → %v", sentinel, err)
	}
}

// TestSlotBytesPerLayer_densePrefix is the C-25 gate: sizing must come from a ROUTED layer. A
// dense layer 0 (first_k_dense_replace) reports zero, and zero is what the caller divides by.
func TestSlotBytesPerLayer_densePrefix(t *testing.T) {
	// Layer 0 dense (no expert strides), layer 1 routed — the GLM/DeepSeek/Kimi shape.
	r := &cudaResident{layers: []cudaLayer{
		{},
		{expGU: cudaWQ{perExpertW: 1024, perExpertS: 128}, expDown: cudaWQ{perExpertW: 512, perExpertS: 64}},
	}}

	if got := r.slotBytesPerLayer(0); got != 0 {
		t.Fatalf("dense layer 0 reports %d bytes/slot, want 0 — the fixture does not reproduce the finding", got)
	}
	want := (1024+512)*4 + (128+64)*2
	if got := r.slotBytesPerLayer(1); got != want {
		t.Errorf("routed layer 1 = %d bytes/slot, want %d", got, want)
	}
	// The bug was structural: sizing off layer 0 yields 0, and allocSlots divides by it. Assert the
	// property that makes the division safe, since allocSlots itself needs a device to reach.
	if r.slotBytesPerLayer(0) != 0 || r.slotBytesPerLayer(1) == 0 {
		t.Fatal("fixture inverted")
	}
}

// TestSlotBytesPerLayer_outOfRange: the guard must not itself panic on a bad index.
func TestSlotBytesPerLayer_outOfRange(t *testing.T) {
	r := &cudaResident{layers: []cudaLayer{{}}}
	for _, l := range []int{-1, 1, 99} {
		if got := r.slotBytesPerLayer(l); got != 0 {
			t.Errorf("slotBytesPerLayer(%d) = %d, want 0", l, got)
		}
	}
}

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
