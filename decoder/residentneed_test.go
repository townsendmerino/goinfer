package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
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

// TestPlan_metalNeedCountsDecodeScratch is MM5b's gate (docs/measurements/mm5-decode-scratch-2026-11/): on every fixture G-MS3 holds
// Metal's decode list to, Plan("metal").NeedBytes and the resident guard's number count decode's scratch (MetalDecodeScratchBytes)
// beside the weights, KV, host copy and state; the CPU's do not.
func TestPlan_metalNeedCountsDecodeScratch(t *testing.T) {
	const ctx = 256
	for _, c := range []struct{ name, quant string }{
		{"llama-tiny", "int8int8"}, {"qwen35-tiny", "int4"}, {"gemma2-tiny", "int8int8"}, {"deepseek-tiny", "int4"}, {"nemotron-tiny", "int4"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", c.name)
			m, err := Load(dir, Options{Quant: c.quant})
			if err != nil {
				t.Fatalf("load %s: %v", c.name, err)
			}
			defer m.Close()
			scratch := m.MetalDecodeScratchBytes(0)
			if scratch <= 0 {
				t.Fatalf("%s: Metal's decode scratch is priced at %d bytes", c.name, scratch)
			}
			lines := func(p Plan) int64 {
				return p.DenseBytes + p.StateBytes + p.AllocSlackBytes + p.ExpertBytesUsed + p.KVBytes + p.HostCopyBytes + p.ExtraBytes
			}
			p := m.Plan("metal", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true})
			if got := p.NeedBytes() - lines(p); got != scratch {
				t.Errorf("%s: Plan(\"metal\").NeedBytes is %d bytes past its weight, KV, host-copy and state lines; want the decode scratch, %d",
					c.name, got, scratch)
			}
			if want := m.ResidentNeedBytes("metal", p.Slots, p.Ctx, false, false); p.NeedBytes() != want {
				t.Errorf("%s: Plan(\"metal\").NeedBytes %d, the resident guard's %d", c.name, p.NeedBytes(), want)
			}
			if cp := m.Plan("cpu", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true}); cp.NeedBytes() != lines(cp) {
				t.Errorf("%s: Plan(\"cpu\").NeedBytes counts %d bytes past its lines; the CPU holds no Metal scratch", c.name, cp.NeedBytes()-lines(cp))
			}
		})
	}
}

// TestPlan_metalScratchFollowsTheSlots: Metal's decode scratch holds the routed experts' slot indices, 4 bytes for each of the top k,
// exactly when the experts page (slots above 0 and under the expert count). The resident guard's figure counts them at a paged slot
// count and not at an unpaged one, and Plan("metal") counts them on an expert cache that pages and not on a resident placement or a
// cache that holds every expert; written as arithmetic on each fixture's top k, on a generic MoE and on Gemma 4's.
func TestPlan_metalScratchFollowsTheSlots(t *testing.T) {
	const ctx = 64
	for _, name := range []string{"mixtral-tiny", "gemma4-moe-kv-tiny32"} {
		t.Run(name, func(t *testing.T) {
			m := loadAccountingFixture(t, name)
			nE, k, isMoE := m.moeGeometry()
			if !isMoE || k >= nE {
				t.Fatalf("%s: %d experts, top %d: the fixture cannot page", name, nE, k)
			}
			index := int64(4 * k)
			resident := m.MetalDecodeScratchBytes(0)
			// The guard's figure past its weight and host-copy terms, which also move with the slots; KV and state do not.
			scratchIn := func(slots int) int64 {
				return m.ResidentNeedBytes("metal", slots, ctx, false, false) - m.ResidentNeedBytes("metal", 0, ctx, false, false) -
					(m.ResidentWeightBytesPaged(slots) - m.ResidentWeightBytesPaged(0)) - (m.ResidentHostCopyBytes(slots) - m.ResidentHostCopyBytes(0)) + resident
			}
			for _, c := range []struct {
				slots int
				want  int64
			}{{0, resident}, {nE, resident}, {k, resident + index}, {nE - 1, resident + index}} {
				if got := m.MetalDecodeScratchBytes(c.slots); got != c.want {
					t.Errorf("%s: MetalDecodeScratchBytes(%d) = %d, want %d", name, c.slots, got, c.want)
				}
				if got := scratchIn(c.slots); got != c.want {
					t.Errorf("%s: the resident guard at %d slots counts %d bytes of scratch, want %d", name, c.slots, got, c.want)
				}
			}
			p := m.Plan("metal", 1<<40, PlanRequest{Ctx: ctx, CtxPinned: true})
			if p.Placement != PlacementResident || p.ScratchBytes != resident {
				t.Errorf("%s: resident plan: placement %v, scratch %d; want resident at %d", name, p.Placement, p.ScratchBytes, resident)
			}
			// One byte short of resident, the cache holds every expert, which does not page; asked for top-k slots, it pages.
			short := m.ResidentNeedBytes("metal", 0, ctx, false, false) - 1
			for _, c := range []struct {
				asked, slots int
				want         int64
			}{{0, nE, resident}, {k, k, resident + index}} {
				pc := m.Plan("metal", short, PlanRequest{Ctx: ctx, CtxPinned: true, Slots: c.asked})
				if pc.Placement != PlacementExpertCached || pc.Slots != c.slots || pc.ScratchBytes != c.want {
					t.Errorf("%s: one byte short of resident, %d slots asked: placement %v at %d slots (%s), scratch %d; want expert-cached at %d slots, %d",
						name, c.asked, pc.Placement, pc.Slots, pc.Reason, pc.ScratchBytes, c.slots, c.want)
				}
				if want := m.ResidentNeedBytes("metal", pc.Slots, ctx, false, false); pc.NeedBytes() != want {
					t.Errorf("%s: expert-cached (%d slots): Plan.NeedBytes %d, the resident guard's %d", name, pc.Slots, pc.NeedBytes(), want)
				}
			}
		})
	}
}

// TestMetalDecodeScratch_q4kSegmentsAreThePartsTheLaneUploads: on a q4k load, each fused group's segments are the runs of one kind
// among the parts the layer holds, one row-count uniform each. A part with no rows is not uploaded: emptied between two Q4_K parts it
// splits no run, and a group with none of its parts adds no segment. On a two-layer q4k llama whose projections are all Q4_K.
func TestMetalDecodeScratch_q4kSegmentsAreThePartsTheLaneUploads(t *testing.T) {
	m, err := Load(q4kGGUF(t, t.TempDir(), 2), Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Quant() != "q4k" {
		t.Fatalf("the load's quant is %q, want q4k", m.Quant())
	}
	base := m.MetalDecodeScratchBytes(0)
	lw := &m.w.Layers[0]
	k := lw.KProj
	lw.KProj = linalg.WeightMat{}
	if got := m.MetalDecodeScratchBytes(0); got != base {
		t.Errorf("K emptied between Q4_K q and v: decode's scratch %d bytes, want %d (q and v stay one run)", got, base)
	}
	lw.KProj = k
	gate, up := lw.GateProj, lw.UpProj
	lw.GateProj, lw.UpProj = linalg.WeightMat{}, linalg.WeightMat{}
	if got, want := m.MetalDecodeScratchBytes(0), base-4; got != want {
		t.Errorf("gate and up emptied: decode's scratch %d bytes, want %d (the group's one segment gone)", got, want)
	}
	lw.GateProj, lw.UpProj = gate, up
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

// TestResidentNeedBytes_countsRecurrentState: Metal's resident guard and Plan price the same bytes on a recurrent model, slot 0's
// state included, so a one-slot load is judged with the state it allocates.
func TestResidentNeedBytes_countsRecurrentState(t *testing.T) {
	for _, dir := range []string{"../testdata/qwen35-tiny", "../testdata/nemotron-tiny", "../testdata/olmo_hybrid-tiny"} {
		m, err := Load(dir, Options{Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		p := m.Plan("metal", 1<<50, PlanRequest{Ctx: 64, CtxPinned: true})
		if p.StateBytes <= 0 {
			t.Fatalf("%s: the plan prices no recurrent state: this test no longer discriminates", dir)
		}
		if got := m.ResidentNeedBytes("metal", p.Slots, p.Ctx, false, false); got != p.NeedBytes() {
			t.Errorf("%s: ResidentNeedBytes %d, Plan.NeedBytes %d (state %d)", dir, got, p.NeedBytes(), p.StateBytes)
		}
	}
}
