package mem

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"
)

// genDevice is a generated DeviceLoad: KV linear per position, or (CtxSearch) whole 64-position blocks beside a constant, so
// non-decreasing either way; a third of the loads are MoE.
func genDevice(r *rand.Rand) (DeviceLoad, PlanRequest) {
	be := []string{"cpu", "cuda", "metal", "webgpu"}[r.IntN(4)]
	l := DeviceLoad{Backend: be, Tier: VRAM, DenseBytes: r.Int64N(16 * genGiB), CtxStep: 1}
	perPos := r.Int64N(1<<18) + 1
	l.KV = func(ctx int) int64 { return perPos * int64(ctx) }
	switch be {
	case "cuda":
		base := r.Int64N(64 << 20)
		l.CtxSearch, l.FixedSlack = true, r.Int64N(256<<20)
		l.KV = func(ctx int) int64 { return base + perPos*64*int64((ctx+63)/64) }
	case "metal":
		l.CtxStep, l.CtxCeiling = 8, []int{16384, 32768}[r.IntN(2)]
		l.HostFixed = r.Int64N(4 * genGiB)
	case "webgpu":
		l.CtxCeiling = []int{16384, 32768, 65536}[r.IntN(3)]
	}
	if r.IntN(3) == 0 {
		l.NExperts, l.TopK = []int{8, 64, 128}[r.IntN(3)], []int{2, 8}[r.IntN(2)]
		l.ExpertBytesFull = int64(l.NExperts) * (r.Int64N(64<<20) + 1)
		extra := l.HostFixed + r.Int64N(genGiB)
		l.HostAll = func() int64 { return extra }
	}
	req := PlanRequest{Ctx: []int{4096, 16384, 32768, 131072}[r.IntN(4)], CtxPinned: r.IntN(3) == 0, Slots: []int{0, 0, 16}[r.IntN(3)],
		ExtraBytes: []int64{0, 256 << 20}[r.IntN(2)]}
	return l, req
}

func planAnswer(p DevicePlan) string {
	return fmt.Sprintf("%d %d %d %d %d %d %q", p.Placement, p.Ctx, p.Slots, p.ExpertBytesUsed, p.KVBytes, p.HostCopyBytes, p.Reason)
}

func metalAnswer(a MetalCtx) string {
	return fmt.Sprintf("%d %d %d %q", a.Ctx, a.Slots, a.KVBytes, a.Why)
}

func genMetalCtx(r *rand.Rand) MetalCtxLoad {
	l, _ := genDevice(r)
	l.Backend, l.Tier, l.CtxStep, l.CtxCeiling, l.CtxSearch = "metal", Unified, 8, 32768, false
	return MetalCtxLoad{Load: l, Floor: 4096, Candidate: []int{512, 8192, 16384}[r.IntN(3)], Lower: []int{0, 0, 2048, 8192}[r.IntN(4)],
		Slots: 1 + r.IntN(4), FitDisabled: r.IntN(20) == 0, PagedMoE: r.IntN(10) == 0, FreeKnown: r.IntN(10) > 0, Ineligible: r.IntN(10) == 0,
		ExtraBytes: []int64{0, 512 << 20}[r.IntN(2)], ExtraKVPerPosition: []int64{0, 4096}[r.IntN(2)], Pad: 8}
}

func deviceAnswers(seed uint64, n int) string {
	r := rand.New(rand.NewPCG(seed, 5))
	p := DefaultPolicy()
	var b strings.Builder
	for range n {
		l, req := genDevice(r)
		m := Machine{Free: map[Tier]int64{VRAM: r.Int64N(64 * genGiB), Unified: r.Int64N(32 * genGiB)}}
		b.WriteString(planAnswer(SolvePlacement(l, req, m, p)) + " | " + metalAnswer(SolveMetalCtx(genMetalCtx(r), m, p)) + "\n")
	}
	return b.String()
}

// TestSolvePlacement_pure (G-MM1b-2, I4) is TestSolve_pure for the device planner and the Metal context fit.
func TestSolvePlacement_pure(t *testing.T) {
	const n = 10000
	want := deviceAnswers(9, n)
	if deviceAnswers(9, n) != want {
		t.Fatal("a second run in the same goroutine answered differently")
	}
	var wg sync.WaitGroup
	got := make([]string, 16)
	for g := range got {
		wg.Add(1)
		go func() { defer wg.Done(); got[g] = deviceAnswers(9, n) }()
	}
	wg.Wait()
	for g, a := range got {
		if a != want {
			t.Fatalf("goroutine %d answered differently", g)
		}
	}
}

// placementRank orders placements from worst to best.
func placementRank(p Placement) int {
	switch p {
	case Declined:
		return 0
	case WeightPaged:
		return 1
	case ExpertCached:
		return 2
	default:
		return 3
	}
}

// TestSolvePlacement_monotone (G-MM1b-2 as amended, I5): over a ladder of free memory, a plan's placement never moves down the
// order, its context never shrinks at the same resident or expert-cached placement, nor its slots at the same context, and the
// Metal context fit never shrinks.
func TestSolvePlacement_monotone(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 7))
	p := DefaultPolicy()
	ladder := []int64{0, 1 << 20, 256 << 20, genGiB, 2 * genGiB, 4 * genGiB, 8 * genGiB, 16 * genGiB, 32 * genGiB, 64 * genGiB, 1 << 44}
	for i := range 10000 {
		l, req := genDevice(r)
		mc := genMetalCtx(r)
		var prev DevicePlan
		var prevM MetalCtx
		for j, f := range ladder {
			m := Machine{Free: map[Tier]int64{VRAM: f, Unified: f}}
			d, a := SolvePlacement(l, req, m, p), SolveMetalCtx(mc, m, p)
			if j > 0 {
				pr, dr := placementRank(prev.Placement), placementRank(d.Placement)
				placed := d.Placement == Resident || d.Placement == ExpertCached // a refusal's context and slots only echo the attempt
				// The planner takes context before experts (SolvePlacement's order), so a larger context may leave fewer slots;
				// slots are held only where the context did not move.
				if dr < pr || (dr == pr && placed && (d.Ctx < prev.Ctx || (d.Ctx == prev.Ctx && d.Slots < prev.Slots))) {
					t.Fatalf("input %d (%+v, %+v): the plan got worse from %d to %d free: %s -> %s", i, l, req, ladder[j-1], f, planAnswer(prev), planAnswer(d))
				}
				if a.Ctx < prevM.Ctx {
					t.Fatalf("input %d: the Metal context fell from %d to %d free: %s -> %s", i, ladder[j-1], f, metalAnswer(prevM), metalAnswer(a))
				}
			}
			prev, prevM = d, a
		}
	}
}
