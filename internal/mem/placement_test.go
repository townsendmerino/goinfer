package mem

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"
)

// genDevice is a generated DeviceLoad on its backend's tier: KV linear per position (on Metal's 8-position grid), or (CUDA,
// CtxSearch) whole 64-position blocks beside a constant, so non-decreasing either way; a third of the loads are MoE. Only Metal
// holds a host copy.
func genDevice(r *rand.Rand) (DeviceLoad, PlanRequest) {
	be := []string{"cpu", "cuda", "metal", "webgpu"}[r.IntN(4)]
	l := DeviceLoad{Backend: be, Tier: map[string]Tier{"cpu": HostAnon, "cuda": VRAM, "metal": Unified, "webgpu": VRAM}[be],
		DenseBytes: r.Int64N(16 * genGiB), CtxStep: 1}
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
		if be == "metal" {
			all := l.HostFixed + r.Int64N(genGiB)
			l.HostAll = func() int64 { return all }
		}
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

// genMetalCtx is a generated Metal context query on a Metal load (linear KV on the 8-position grid). Each early answer (fit off,
// paged experts, an unknown budget, no resident path, a candidate at the floor) is drawn rarely, so most draws run the search.
func genMetalCtx(r *rand.Rand) MetalCtxLoad {
	l, _ := genDevice(r)
	perPos := r.Int64N(1<<18) + 1
	l.Backend, l.Tier, l.CtxStep, l.CtxCeiling, l.CtxSearch, l.FixedSlack = "metal", Unified, 8, 32768, false, 0
	l.KV = func(ctx int) int64 { return perPos * int64((ctx+7)/8*8) }
	if l.HostAll == nil && l.NExperts > 0 {
		all := l.HostFixed + r.Int64N(genGiB)
		l.HostAll = func() int64 { return all }
	}
	rare := func() bool { return r.IntN(20) == 0 }
	return MetalCtxLoad{Load: l, Floor: 4096, Candidate: []int{8192, 16384, 16384, 512}[r.IntN(4)], Lower: []int{0, 0, 0, 8192}[r.IntN(4)],
		Slots: 1 + r.IntN(4), FitDisabled: rare(), PagedMoE: rare(), FreeKnown: !rare(), Ineligible: rare(),
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

// monotoneBreak walks n generated loads over free memory in 16 MiB steps to 64 GiB through solve and SolveMetalCtx and returns
// the first break of G-MM1b-2 as amended (I5 as far as it holds), or "": a dense plan's placement never moves down the order; a
// placed context (resident or expert-cached) never shrinks, MoE included; the slots never fall at the same context and
// placement; an MoE plan's placement never worsens at the same context; and a dense load's Metal context never shrinks. It does
// not hold an MoE plan's placement as the context grows, nor an MoE Metal context: the context grows first, and a step of it can
// leave no room for the top-k experts (docs/tasks/task-subsystems-2026-10.md, "MM1 follow-up"). searched counts the Metal draws whose answer passed the floor.
func monotoneBreak(n int, solve func(DeviceLoad, PlanRequest, Machine, Policy) DevicePlan) (brk string, searched int) {
	r := rand.New(rand.NewPCG(13, 7))
	p := DefaultPolicy()
	const steps, step = 4096, int64(16 << 20)
	placed := func(x DevicePlan) bool { return x.Placement == Resident || x.Placement == ExpertCached }
	for i := range n {
		l, req := genDevice(r)
		mc := genMetalCtx(r)
		var prev DevicePlan
		var prevM MetalCtx
		reached := false
		for j := range steps {
			f := int64(j) * step
			m := Machine{Free: map[Tier]int64{HostAnon: f, VRAM: f, Unified: f}}
			d, a := solve(l, req, m, p), SolveMetalCtx(mc, m, p)
			reached = reached || a.Ctx > mc.Floor
			switch {
			case j == 0:
			case l.NExperts == 0 && placementRank(d.Placement) < placementRank(prev.Placement):
				return fmt.Sprintf("input %d: a dense plan got worse from %d to %d free: %s -> %s", i, f-step, f, planAnswer(prev), planAnswer(d)), searched
			case l.NExperts > 0 && placed(prev) && d.Ctx == prev.Ctx && placementRank(d.Placement) < placementRank(prev.Placement):
				return fmt.Sprintf("input %d: an MoE plan got worse at the same context from %d to %d free: %s -> %s", i, f-step, f, planAnswer(prev), planAnswer(d)), searched
			case placed(d) && placed(prev) && d.Ctx < prev.Ctx:
				return fmt.Sprintf("input %d: a placed context shrank from %d to %d free: %s -> %s", i, f-step, f, planAnswer(prev), planAnswer(d)), searched
			case d.Placement == prev.Placement && d.Ctx == prev.Ctx && d.Slots < prev.Slots:
				return fmt.Sprintf("input %d: slots fell at the same context from %d to %d free: %s -> %s", i, f-step, f, planAnswer(prev), planAnswer(d)), searched
			case mc.Load.NExperts == 0 && a.Ctx < prevM.Ctx:
				return fmt.Sprintf("input %d: the Metal context fell from %d to %d free: %s -> %s", i, f-step, f, metalAnswer(prevM), metalAnswer(a)), searched
			}
			prev, prevM = d, a
		}
		if reached {
			searched++
		}
	}
	return "", searched
}

// TestSolvePlacement_monotone (G-GF-2): monotoneBreak finds no break over 1,500 loads, at least a third of the Metal draws reach
// the search, and a planted defect (a refusal above 32 GiB free) is found.
func TestSolvePlacement_monotone(t *testing.T) {
	const n = 1500
	brk, searched := monotoneBreak(n, SolvePlacement)
	if brk != "" {
		t.Fatal(brk)
	}
	if searched*3 < n {
		t.Errorf("only %d of %d Metal draws reached a context past the floor: the Metal check is mostly vacuous", searched, n)
	}
	t.Logf("%d of %d Metal draws reached a context past the floor", searched, n)
	planted := func(l DeviceLoad, req PlanRequest, m Machine, p Policy) DevicePlan {
		if m.Available(l.Tier) > 32*genGiB {
			return DevicePlan{Placement: Declined}
		}
		return SolvePlacement(l, req, m, p)
	}
	if brk, _ := monotoneBreak(50, planted); brk == "" {
		t.Fatal("a planted refusal above 32 GiB free went unseen")
	}
}

// TestSolvePlacement_unpricedExperts (G-NP2-3): an MoE whose experts could not be priced (under a byte each) is declined on a
// device and paged on the CPU, at any slot count, with a reason naming them, on loads that keep DeviceLoad's invariants (no
// host copy on CUDA; Metal's host copy is its experts').
func TestSolvePlacement_unpricedExperts(t *testing.T) {
	kv := func(c int) int64 { return int64(c) << 10 }
	for _, full := range []int64{0, 7} {
		for _, slots := range []int{0, 2} {
			loads := map[string]DeviceLoad{
				"cuda": {Backend: "cuda", Tier: VRAM, DenseBytes: 1 << 30, CtxStep: 1, KV: kv, NExperts: 8, TopK: 2, ExpertBytesFull: full},
				"metal": {Backend: "metal", Tier: Unified, DenseBytes: 1 << 30, CtxStep: 8, CtxCeiling: 32768, KV: kv, NExperts: 8, TopK: 2,
					ExpertBytesFull: full, HostFixed: 1 << 20, HostAll: func() int64 { return 1<<20 + full }},
				"cpu": {Backend: "cpu", Tier: HostAnon, DenseBytes: 1 << 30, CtxStep: 1, KV: kv, NExperts: 8, TopK: 2, ExpertBytesFull: full},
			}
			for be, l := range loads {
				d := SolvePlacement(l, PlanRequest{Ctx: 4096, Slots: slots}, Machine{Free: map[Tier]int64{l.Tier: 64 << 30}}, DefaultPolicy())
				want := Declined
				if be == "cpu" {
					want = WeightPaged
				}
				if d.Placement != want || !strings.Contains(d.Reason, "could not be priced") {
					t.Errorf("%s, ExpertBytesFull %d over 8 experts, slots %d: %s, want %v naming the unpriced experts", be, full, slots, planAnswer(d), want)
				}
			}
		}
	}
}
