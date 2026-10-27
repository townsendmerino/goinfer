package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// docs/tasks/task-memory-accounting-2026-09.md item 2: `fit`'s verdict (Plan) and Metal's resident guard (ResidentNeedBytes,
// via residentNeedBytes in metal/backend.go) must be the same number by construction. Plan("metal") once had no host-copy term
// and priced Metal's KV at f32, so on a directly loaded model it asked for about half of what the guard then demanded.

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
	// Off Metal nothing changed: no host copy, and KV as the CUDA pricing gives it at the requested precision.
	pc := m.Plan("cuda", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true})
	if pc.HostCopyBytes != 0 || pc.KVBytes != m.ResidentKVBytes("cuda", ctx, false, false) {
		t.Errorf("Plan(\"cuda\") changed: host copy %d, KV %d", pc.HostCopyBytes, pc.KVBytes)
	}
}

// The "cuda" layout, pinned where CI can run it (the allocation-level check, cuda's
// TestResidentKVBytes_matchesCUDAAllocation, needs a GPU): f32 unless f16 is asked for AND the f16 kernels cover the family (then half,
// kv16_cuda.go), K and V on every attention layer, and ONE latent buffer on an MLA layer (which has no f16 form, so f16 prices as f32). int8 is not a CUDA
// cache, and Plan's old per-position formula cut it to ~0.28x and doubled MLA (docs/measurements/memory-accounting-cuda-2026-09-25.md). The storage is
// paged (cuda/kvpool.go): a one-slot load holds the context's whole 64-row blocks, one null block, and two tables of 4 B a block.
func TestResidentKVBytes_cudaLayout(t *testing.T) {
	// The CUDA backend's policy lives in the cuda package, which this binary does not link: stand in for it with the one fact the pricing reads (dense, no MLA).
	saved := kvPolicies["cuda"]
	RegisterKVF16Policy("cuda", KVF16Policy{Decline: func(f KVFacts, _ Options) string {
		if f.MLA {
			return "an MLA latent cache has no f16 form"
		}
		return ""
	}, Auto: func() bool { return false }})
	defer func() { kvPolicies["cuda"] = saved }() // restore decoder's own default (kvresolve.go's init), which later tests read
	const ctx = 1000
	for _, c := range []struct {
		dir string
		mla bool
	}{{"../testdata/llama-tiny", false}, {"../testdata/deepseek-tiny", true}} {
		m, err := Load(c.dir, Options{Quant: "f32"})
		if err != nil {
			t.Skipf("no fixture at %s: %v", c.dir, err)
		}
		a := m.w.arch
		_, nLayers, _, _, _, _, _ := m.Dims()
		var perRow int64 // bytes of one position across the layers
		for l := range nLayers {
			per := int64(a.kvDimAt(l)) * 4
			if !c.mla {
				per *= 2
			}
			perRow += per
		}
		blocks := int64((ctx + KVBlockRows - 1) / KVBlockRows)
		rows, tables := (blocks+1)*KVBlockRows, 2*4*blocks
		if (a.mla != nil) != c.mla {
			t.Fatalf("%s: fixture MLA=%v, expected %v", c.dir, a.mla != nil, c.mla)
		}
		for _, p := range []struct{ f16, i8 bool }{{false, false}, {true, false}, {false, true}} {
			w := perRow*rows + tables
			if p.f16 && kvF16Decline("cuda", a, Options{Backend: "cuda"}) == "" {
				w = perRow/2*rows + tables
			}
			if got := m.ResidentKVBytes("cuda", ctx, p.f16, p.i8); got != w {
				t.Errorf("%s (f16=%v i8=%v): ResidentKVBytes(\"cuda\") = %d, want %d (%s)", c.dir, p.f16, p.i8, got, w,
					map[bool]string{true: "one latent buffer per layer", false: "K and V per layer"}[c.mla])
			}
		}
		m.Close()
	}
}
