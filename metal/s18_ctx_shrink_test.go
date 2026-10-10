//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestS18ShrinkCtxToFit is S18's part 3 on the Mac (docs/tasks/task-multimodal-support-2026-10.md, "S18 on the Mac", registered before the code): with a
// budget between what the resident needs at the 4096 default and at the 2048 floor, an unpinned build shrinks its context to 2048 and goes resident
// instead of declining; an explicit -ctx never shrinks (the build declines); and the ceiling's entry goes with the resident, so it cannot keep a closed
// model alive. Red before S18: the 4096 build declined.
func TestS18ShrinkCtxToFit(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny" // a 4096-position window: llama-tiny's is 128, which clamps a pinned 4096 into fitting
	// The probe is a Metal load too: a CPU load prices more (it keeps arm64-repacked weights a Metal load does not), which would put
	// the budget above the Metal build's 4096 need.
	probe, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "metal"})
	if err != nil {
		t.Skipf("llama-attnfa-tiny: %v", err)
	}
	need4096 := residentNeedBytes(probe)
	metalCtxCeiling.Store(probe, 2048)
	need2048 := residentNeedBytes(probe)
	metalCtxCeiling.Delete(probe)
	probe.Close()
	if need2048 >= need4096 {
		t.Fatalf("need at 2048 (%d) is not under need at 4096 (%d)", need2048, need4096)
	}
	live := metalLiveAvailable
	metalLiveAvailable = func() int64 { return (need2048 + need4096) / 2 }
	t.Cleanup(func() { metalLiveAvailable = live })

	m, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "metal"})
	if err != nil {
		t.Fatal(err)
	}
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("an unpinned build declined instead of shrinking its context: %s", m.ResidentDecline())
	}
	if got, _ := resolveMetalCtxCap(m); got != 2048 || a.ctxCap() != 2048 {
		t.Errorf("context %d (resident %d), want 2048", got, a.ctxCap())
	}
	if a.r.ctxCeilKey != m {
		t.Error("the resident does not hold its ceiling's key")
	}
	m.Close()
	if _, ok := metalCtxCeiling.Load(m); ok {
		t.Error("the ceiling outlived its resident (it would keep the model alive)")
	}

	pinned, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer pinned.Close()
	if _, ok := pinned.ResidentForwardForTest().(*metalResident); ok {
		t.Error("an explicit -ctx 4096 was shrunk to fit")
	}
	if _, ok := metalCtxCeiling.Load(pinned); ok {
		t.Error("a pinned decline left a ceiling behind")
	}
}
