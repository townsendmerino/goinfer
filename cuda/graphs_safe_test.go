//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGraphsSafeGate verifies the "byte-identical or decline, never silently mis-run" gate for CUDA
// graphs. On a DEFAULT-compute-mode box with no MPS (the common shared-GPU case, and this box), replay
// is NOT bit-exact under contention, so GOINFER_CUDA_GRAPHS=1 must DECLINE (fall to the live path).
// The GOINFER_CUDA_GRAPHS_UNSAFE override force-enables and the startup self-test must still pass
// bit-exact (idle box, no churn → capture is bit-exact). A box in EXCLUSIVE_PROCESS or MPS would admit
// graphs without the override — not reachable here, so it's asserted via graphsTenancySafe directly.
func TestGraphsSafeGate(t *testing.T) {
	const path = "../testdata/mistral-tiny-window"
	requireDeviceAndFixture(t, path)

	load := func() *cudaResident {
		mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { mc.Close() })
		rf, ok := mc.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatal("resident is not *cudaResident (backend declined residency entirely)")
		}
		return rf
	}

	// 1. GOINFER_CUDA_GRAPHS=1 on a DEFAULT box → declined.
	t.Setenv("GOINFER_CUDA_GRAPHS", "1")
	t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
	rf := load()
	reason, ok := rf.graphsTenancySafe()
	if ok {
		t.Fatalf("expected UNSAFE tenancy on a DEFAULT box, got safe: %s", reason)
	}
	if rf.graphs {
		t.Fatal("SAFETY VIOLATION: graphs enabled under DEFAULT compute mode — the gate must decline")
	}
	t.Logf("DEFAULT box, GRAPHS=1: tenancy UNSAFE (%s) → graphs DECLINED (live path) ✓", reason)

	// 2. GOINFER_CUDA_GRAPHS_UNSAFE=1 → force-enabled, and the startup self-test passed bit-exact
	//    (else admitGraphs would have set graphs=false).
	t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
	rf2 := load()
	if !rf2.graphs {
		t.Fatal("UNSAFE override should force graphs on and the idle-box self-test should pass bit-exact")
	}
	t.Logf("UNSAFE override: graphs FORCED on + startup self-test bit-exact ✓")
}
