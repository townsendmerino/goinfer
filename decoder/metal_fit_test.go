package decoder

import (
	"fmt"
	"strings"
	"testing"
)

// TestPlan_metalPlansWhatMetalAllocates (C-C01, docs/audit-metal-2026-09-30.md): Plan("metal") never promises a context
// Metal will not allocate. Unpinned, Metal allocates MetalCtxDefault however much memory is free; an explicit -ctx
// reaches past it up to MetalCtxCeiling, above which Metal refuses, so Plan declines it with the ceiling named.
func TestPlan_metalPlansWhatMetalAllocates(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const free = int64(64) << 30
	if p := m.Plan("metal", free, PlanRequest{Ctx: 8192}); p.Placement == PlacementDecline || p.Ctx != MetalCtxDefault {
		t.Errorf("unpinned 8192 on metal: %v ctx %d (%s); want resident at %d, what an unpinned Metal load allocates", p.Placement, p.Ctx, p.Reason, MetalCtxDefault)
	}
	if p := m.Plan("metal", free, PlanRequest{Ctx: 16384, CtxPinned: true}); p.Placement == PlacementDecline || p.Ctx != 16384 {
		t.Errorf("-ctx 16384 on metal: %v ctx %d (%s); want resident at 16384", p.Placement, p.Ctx, p.Reason)
	}
	p := m.Plan("metal", free, PlanRequest{Ctx: MetalCtxCeiling + 1, CtxPinned: true})
	if p.Placement != PlacementDecline || !strings.Contains(p.Reason, fmt.Sprintf("metal's %d-position ceiling", MetalCtxCeiling)) {
		t.Errorf("-ctx %d on metal: %v (%s); want a decline naming the ceiling", MetalCtxCeiling+1, p.Placement, p.Reason)
	}
}

// TestGuardGIWFit_onMetalStillPricesTheCPUWorstCase: the load-time guard runs before the backend decides whether it can
// host the model, and a model Metal declines runs on the CPU, whose KV grows to the request (R13). So the guard keeps
// pricing the window at the CPU cache's precision on a Metal load too; resolveMetalCtxCap then treats its auto-pin as
// a ceiling (metal/ctxcap_autopin_test.go). The cost, recorded in docs/tasks/task-metal-audit-2026-10.md: a machine too
// tight for the 2048-position floor at f32 is refused a load Metal could have run at f16.
func TestGuardGIWFit_onMetalStillPricesTheCPUWorstCase(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 131072}
	injectHostRAM(t, int64(giwMemMargin+prefillAttnScratchBudget)+estimateKVBytes(cfg, 40000, false, false))
	for _, backend := range []string{"cpu", "metal"} {
		if pin, err := guardGIWFit(cfg, Options{Backend: backend}); err != nil || pin != 40000 {
			t.Errorf("%s: guardGIWFit = %d, %v; want the pin 40000 the CPU worst case fits", backend, pin, err)
		}
	}
}

// TestGuardGIWFit_floorRefusalNamesTheFloorsNeed: the refusal says what the floor needs, not the whole window's
// need, under "even at the 2048-token floor".
func TestGuardGIWFit_floorRefusalNamesTheFloorsNeed(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 131072}
	injectHostRAM(t, int64(giwMemMargin)+prefillAttnScratchBudget/2)
	_, err := guardGIWFit(cfg, Options{})
	if err == nil {
		t.Fatal("no refusal below the floor; the test proves nothing")
	}
	floor := fmt.Sprintf("~%.2f GB", float64(estimateKVBytes(cfg, ctxFloor, false, false)+prefillAttnScratchBudget)/fitGB)
	if !strings.Contains(err.Error(), floor) {
		t.Errorf("refusal %q does not give the floor's need %s", err, floor)
	}
}
