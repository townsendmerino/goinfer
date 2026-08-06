//go:build cuda

package cuda

import (
	"strings"
	"testing"
)

// Audit C-24 / C-25 — the executor goroutine must survive a panicking job, and the expert-cache
// sizing must not divide by zero on a dense-prefix MoE.
//
// WHY THESE EXIST. Both findings are the same shape: code that the design says should DECLINE
// instead kills the process, and both do it on the pinned executor goroutine where no caller's
// `defer recover()` can reach. C-24: `gpu.NewBufferLenOf` panics on OOM per its own contract, and
// prefillCore allocates hundreds of MB at M=3000 — so a long prompt on a nearly-full card killed
// serve at the exact seam whose job is to fall back to the sequential path. Two comments claimed
// this was handled; `BuildResident`'s recover runs on the CALLING goroutine and cannot catch it.
// C-25 is a reachable trigger for the same crash: GLM/DeepSeek/Kimi put dense layers first, so
// layer 0 has no expert strides and `budget / len(moeLayers) / perLayer` divides by zero.
//
// NO DEVICE NEEDED: runJob is a pure function, and slotBytesPerLayer reads struct state only.

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
