//go:build gpu && goinfer_testhooks

package gpu

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSetAdapter_partialBindErrorRestoresSteps pins that a SetAdapter that fails partway through
// its conversion loop restores r.steps to r.baseSteps' length. SetAdapter clears and releases
// r.loraLayers unconditionally at the top of the call, so a mid-loop error that left r.steps (the
// flat, Go-side dispatch-step list Run walks) pointing at the previously bound adapter, whose
// bind groups were just released, would make a later Run() dispatch against freed WebGPU
// resources.
//
// This backend has no buffer ledger to assert a leak count directly (unlike Metal's
// d.LedgerLen(); metal's TestSetAdapter_partialBindErrorReleasesBuffers is the leak gate), so
// this test pins the more severe, directly observable half: r.steps must not stay at a stale
// adapter's larger spliced-in step count. Background:
// docs/code-notes/gpu.md#TestSetAdapter_partialBindErrorRestoresSteps.
func TestSetAdapter_partialBindErrorRestoresSteps(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident: decline=%s", m.ResidentDecline())
	}
	ra, ok := rf.(decoder.ResidentAdapter)
	if !ok {
		t.Fatalf("webgpu resident runner does not implement decoder.ResidentAdapter")
	}
	rd, ok := rf.(*residentDecoder)
	if !ok {
		t.Fatalf("resident runner is %T, not *residentDecoder", rf)
	}
	r := rd.runner
	if err := m.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	layers := m.ResidentAdapterLayersForTest("a")
	if len(layers) < 2 {
		t.Fatalf("llama-tiny adapter has %d layers, need >= 2 to exercise a PARTIAL bind", len(layers))
	}

	baseLen := len(r.baseSteps)

	// First bind: a real, fully valid adapter across every layer — r.steps must grow past
	// r.baseSteps (the LoRA down/up dispatch pairs spliced in), giving a non-trivial "stale"
	// step count to detect if a later failed rebind leaves it untouched.
	if err := ra.SetAdapter(layers); err != nil {
		t.Fatalf("SetAdapter (valid): %v", err)
	}
	boundLen := len(r.steps)
	if boundLen <= baseLen {
		t.Fatalf("test setup: bound step count %d did not grow past base %d — the fixture isn't "+
			"exercising a real splice, this test proves nothing", boundLen, baseLen)
	}

	// Second bind: layer 1's Q rank is invalid, but layer 0 converts cleanly first — this
	// exercises the actual partial-progress path (mk succeeds several times before failing),
	// not a first-projection failure.
	broken := make([]decoder.ResidentAdapterLayer, len(layers))
	copy(broken, layers)
	badQ := *layers[1].Q
	badQ.R = 0 // R<=0: mk refuses
	broken[1].Q = &badQ

	if err := ra.SetAdapter(broken); err == nil {
		t.Fatal("SetAdapter accepted an invalid rank on layer 1 — test setup is wrong")
	}

	if r.loraLayers != nil {
		t.Error("r.loraLayers not nil after a failed rebind — a caller checking it would think an adapter is bound")
	}
	if got := len(r.steps); got != baseLen {
		t.Errorf("r.steps has %d entries after a failed rebind, want %d (r.baseSteps, no adapter) — "+
			"still reflects the PREVIOUS successful bind's %d steps, whose bind groups were already "+
			"released at the top of this SetAdapter call: a Run() now dispatches against freed "+
			"WebGPU resources (N-82)", got, baseLen, boundLen)
	}

	// The resident must still be usable after a rejected bind — no adapter, but not broken.
	if err := ra.SetAdapter(nil); err != nil {
		t.Errorf("SetAdapter(nil) after a failed bind: %v", err)
	}
}
