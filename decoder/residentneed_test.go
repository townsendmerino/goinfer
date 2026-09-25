package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// docs/tasks/task-memory-accounting-2026-09.md item 2: `fit`'s verdict (Plan) and Metal's resident guard
// (ResidentNeedBytes, via metal/backend.go's residentNeedBytes) must be the same number by construction. Before,
// Plan("metal") had no host-copy term and priced Metal's KV at f32, so on a directly loaded model it asked for
// about half of what the guard would then demand (measured: 2.34 vs 4.33 GB on qwen2.5-coder-1.5b).

func loadAccountingFixture(t *testing.T, name string) *Model {
	t.Helper()
	dir := filepath.Join("..", "testdata", name)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture %s: %v", name, err)
	}
	m, err := Load(dir, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestPlan_metalAgreesWithResidentNeedBytes(t *testing.T) {
	const ctx = 100 // not a multiple of Metal's 8-position KV padding
	for _, name := range []string{"llama-tiny", "mixtral-tiny"} {
		t.Run(name, func(t *testing.T) {
			m := loadAccountingFixture(t, name)
			if m.ResidentHostCopyBytes(0) == 0 {
				t.Fatalf("%s: a heap-loaded model reports no host copy — the case this pins is not exercised", name)
			}
			p := m.Plan("metal", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true})
			if p.Placement != PlacementResident {
				t.Fatalf("with unlimited memory: placement %v (%s)", p.Placement, p.Reason)
			}
			if want := m.ResidentNeedBytes("metal", p.Slots, p.Ctx, false, false); p.NeedBytes() != want {
				t.Errorf("resident: Plan.NeedBytes %d != ResidentNeedBytes %d (dense %d experts %d host %d kv %d)",
					p.NeedBytes(), want, p.DenseBytes, p.ExpertBytesUsed, p.HostCopyBytes, p.KVBytes)
			}
			if p.HostCopyBytes == 0 {
				t.Error("Plan(\"metal\") priced no host copy for a heap-loaded model")
			}
			if p.KVBytes != m.ResidentKVBytes("metal", ctx, false, false) {
				t.Errorf("Plan KV %d != Metal's allocation-exact KV %d", p.KVBytes, m.ResidentKVBytes("metal", ctx, false, false))
			}
			if _, _, isMoE := m.moeGeometry(); isMoE {
				// Just short of fully resident: Plan has to cache experts, and must price the SAME slot count
				// the way the guard would.
				full := m.ResidentNeedBytes("metal", 0, ctx, false, false)
				pc := m.Plan("metal", full-1, PlanRequest{Ctx: ctx, CtxPinned: true})
				if pc.Placement != PlacementExpertCached {
					t.Fatalf("one byte short of resident: placement %v (%s), want expert-cached", pc.Placement, pc.Reason)
				}
				if want := m.ResidentNeedBytes("metal", pc.Slots, ctx, false, false); pc.NeedBytes() != want {
					t.Errorf("expert-cached (%d slots): Plan.NeedBytes %d != ResidentNeedBytes %d", pc.Slots, pc.NeedBytes(), want)
				}
				if pc.NeedBytes() > full-1 {
					t.Errorf("expert-cached plan needs %d B, more than the %d B it was given", pc.NeedBytes(), full-1)
				}
			}
		})
	}
}

// The disagreement itself, on a dense model: given one byte less than the guard needs, Plan("metal") must not
// say resident. The old Plan (no host copy, f32 KV) asked for far less and said resident here.
func TestPlan_metalDeclinesWhereTheGuardWould(t *testing.T) {
	m := loadAccountingFixture(t, "llama-tiny")
	const ctx = 64
	guard := m.ResidentNeedBytes("metal", 0, ctx, false, false)
	p := m.Plan("metal", guard-1, PlanRequest{Ctx: ctx, CtxPinned: true})
	if p.Placement == PlacementResident {
		t.Fatalf("Plan(\"metal\") says resident with %d B free; Metal's guard needs %d B for this load (%s)", guard-1, guard, p.Reason)
	}
	if ok := m.Plan("metal", guard, PlanRequest{Ctx: ctx, CtxPinned: true}); ok.Placement != PlacementResident {
		t.Errorf("with exactly the guard's %d B, Plan(\"metal\") says %v (%s)", guard, ok.Placement, ok.Reason)
	}
	// Off Metal nothing changed: no host copy, and KV at the requested precision.
	pc := m.Plan("cuda", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true})
	if pc.HostCopyBytes != 0 || pc.KVBytes != m.kvBytesPerPositionAllLayers(false, false)*ctx {
		t.Errorf("Plan(\"cuda\") changed: host copy %d, KV %d", pc.HostCopyBytes, pc.KVBytes)
	}
}
