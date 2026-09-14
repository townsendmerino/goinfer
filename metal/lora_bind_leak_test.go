//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSetAdapter_partialBindErrorReleasesBuffers gates C-04 (audit-metal-2026-09-12.md):
// SetAdapter's conversion loop used to return immediately on the first bad projection, leaving
// every earlier layer's already-converted device buffers (A/B plus their uniform buffers) on the
// device ledger — referenced by nothing (r.loraLayers/r.loraCached are never set on an error
// return), so they leaked until Close. Every failed bind attempt — a bad rank is the easy way to
// trigger it, but any mid-loop error does — added to the leak.
//
// llama-tiny (4 layers) lets layer 0 convert cleanly before layer 1's invalid rank fails the bind,
// exercising the actual partial-progress path rather than failing on the very first projection.
func TestSetAdapter_partialBindErrorReleasesBuffers(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	const hidden = 64 // testdata/llama-tiny's hidden_size

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident: decline=%s", m.ResidentDecline())
	}
	ra, ok := rf.(decoder.ResidentAdapter)
	if !ok {
		t.Fatalf("metal resident runner does not implement decoder.ResidentAdapter")
	}
	mr, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("resident runner is %T, not *metalResident", rf)
	}
	nL := mr.r.nL
	if nL < 2 {
		t.Fatalf("llama-tiny has %d layers, need >= 2 to exercise a PARTIAL bind", nL)
	}

	before, beforeObjs := mr.r.d.LedgerLen()

	layers := make([]decoder.ResidentAdapterLayer, nL)
	layers[0].Q = &decoder.ResidentAdapterProj{
		A: make([]float32, 4*hidden), B: make([]float32, hidden*4), R: 4, In: hidden, Out: hidden, Scale: 1,
	}
	layers[1].Q = &decoder.ResidentAdapterProj{
		A: make([]float32, hidden), B: make([]float32, hidden), R: 0, In: hidden, Out: hidden, Scale: 1, // R<=0: conv refuses
	}

	if err := ra.SetAdapter(layers); err == nil {
		t.Fatal("SetAdapter accepted an invalid rank on layer 1 — test setup is wrong")
	}

	after, afterObjs := mr.r.d.LedgerLen()
	if after != before || afterObjs != beforeObjs {
		t.Errorf("device ledger grew across a failed SetAdapter: before=(%d,%d) after=(%d,%d) — "+
			"layer 0's converted A/B/uniform buffers leaked (C-04)", before, beforeObjs, after, afterObjs)
	}

	// The resident must still be usable after a rejected bind — no adapter, but not broken.
	if err := ra.SetAdapter(nil); err != nil {
		t.Errorf("SetAdapter(nil) after a failed bind: %v", err)
	}
}
