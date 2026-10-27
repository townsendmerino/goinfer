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

// monotoneBreak walks n generated loads over free memory through solve and SolveMetalCtx and returns the first break of I5, or
// "": for every load, MoE included, the placement never moves down the order; within a placement the context never shrinks, nor
// the slots at the same context; and the Metal context never shrinks (G-MA-2). The walk is 16 MiB steps to 64 GiB, then 64 KiB
// steps over 16 MiB before each of a draw's first change points, where a KV step's sawtooth lives. searched counts the Metal
// draws whose answer passed the floor.
func monotoneBreak(n int, solve func(DeviceLoad, PlanRequest, Machine, Policy) DevicePlan) (brk string, searched int) {
	r := rand.New(rand.NewPCG(13, 7))
	p := DefaultPolicy()
	const steps, step, fine, fineSteps = 4096, int64(16 << 20), int64(64 << 10), 256
	for i := range n {
		l, req := genDevice(r)
		mc := genMetalCtx(r)
		reached := false
		walk := func(frees []int64) (string, []int64) {
			var prev DevicePlan
			var prevM MetalCtx
			var changes []int64
			for j, f := range frees {
				m := Machine{Free: map[Tier]int64{HostAnon: f, VRAM: f, Unified: f}}
				d, a := solve(l, req, m, p), SolveMetalCtx(mc, m, p)
				reached = reached || a.Ctx > mc.Floor
				if j == 0 {
					prev, prevM = d, a
					continue
				}
				if d != prev || a != prevM {
					changes = append(changes, f)
				}
				was := frees[j-1]
				switch {
				case placementRank(d.Placement) < placementRank(prev.Placement):
					return fmt.Sprintf("input %d: the plan got worse from %d to %d free: %s -> %s", i, was, f, planAnswer(prev), planAnswer(d)), nil
				case d.Placement == prev.Placement && (d.Placement == Resident || d.Placement == ExpertCached) && d.Ctx < prev.Ctx:
					return fmt.Sprintf("input %d: the context shrank within a placement from %d to %d free: %s -> %s", i, was, f, planAnswer(prev), planAnswer(d)), nil
				case d.Placement == prev.Placement && d.Ctx == prev.Ctx && d.Slots < prev.Slots:
					return fmt.Sprintf("input %d: slots fell at the same context from %d to %d free: %s -> %s", i, was, f, planAnswer(prev), planAnswer(d)), nil
				case a.Ctx < prevM.Ctx:
					return fmt.Sprintf("input %d: the Metal context fell from %d to %d free: %s -> %s", i, was, f, metalAnswer(prevM), metalAnswer(a)), nil
				}
				prev, prevM = d, a
			}
			return "", changes
		}
		coarse := make([]int64, steps)
		for j := range coarse {
			coarse[j] = int64(j) * step
		}
		b, changes := walk(coarse)
		if b != "" {
			return b, searched
		}
		for _, c := range changes[:min(len(changes), 6)] {
			ladder := make([]int64, fineSteps+1)
			for j := range ladder {
				ladder[j] = c - step + int64(j)*fine
			}
			if b, _ := walk(ladder); b != "" {
				return b, searched
			}
		}
		if reached {
			searched++
		}
	}
	return "", searched
}

// TestSolvePlacement_monotone (G-MA-2): monotoneBreak finds no break over 1,500 loads, at least a third of the Metal draws reach
// the search, and two planted defects are found: a refusal above 32 GiB free, and the planner without the MoE reserve.
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
	noReserve := func(l DeviceLoad, req PlanRequest, m Machine, p Policy) DevicePlan {
		return solvePlacement(l, req, m, p, false)
	}
	if brk, _ := monotoneBreak(n, noReserve); brk == "" {
		t.Fatal("the planner without the MoE reserve passed: the walk cannot see the fall the reserve removes")
	} else {
		t.Logf("without the reserve: %s", brk)
	}
}

// TestSolvePlacement_unpricedExperts: an MoE load whose experts could not be priced (under a byte each) declines the expert cache
// with a reason instead of sizing slots on unknown bytes, and never panics.
func TestSolvePlacement_unpricedExperts(t *testing.T) {
	for _, full := range []int64{0, 7} {
		l := DeviceLoad{Backend: "cuda", Tier: VRAM, DenseBytes: 1 << 30, CtxStep: 1, KV: func(c int) int64 { return int64(c) << 10 },
			NExperts: 8, TopK: 2, ExpertBytesFull: full, HostFixed: 1 << 30, HostAll: func() int64 { return 2 << 30 }}
		d := SolvePlacement(l, PlanRequest{Ctx: 4096, CtxPinned: true}, Machine{Free: map[Tier]int64{VRAM: 3 << 30}}, DefaultPolicy())
		if d.Placement != Declined || !strings.Contains(d.Reason, "could not be priced") {
			t.Errorf("ExpertBytesFull %d over 8 experts: %s, want a decline naming the unpriced experts", full, planAnswer(d))
		}
	}
}

// TestSolvePlacement_moeSawtooth (G-MA-3): the two constructed MoE loads whose answers fell with more memory, walked in 64 KiB
// steps. A CUDA load whose experts are under one KV block (it fell from resident with a top-k reserve alone, and to a decline
// with none) never moves down the order; a Metal load (its context fell to the floor 1,535 times before the reserve) never loses
// context. The planner without the reserve is shown to fall on the CUDA load.
func TestSolvePlacement_moeSawtooth(t *testing.T) {
	p := DefaultPolicy()
	perPos := int64(256 << 10)
	cuda := DeviceLoad{Backend: "cuda", Tier: VRAM, DenseBytes: 2 << 30, CtxStep: 1, CtxSearch: true,
		KV: func(c int) int64 { return perPos * 64 * int64((c+63)/64) }, NExperts: 8, TopK: 2, ExpertBytesFull: 8 << 20}
	falls := func(solve func(DeviceLoad, PlanRequest, Machine, Policy) DevicePlan) int {
		n := 0
		var prev DevicePlan
		for f := int64(3 << 30); f < int64(12<<30); f += 64 << 10 {
			d := solve(cuda, PlanRequest{Ctx: 32768}, Machine{Free: map[Tier]int64{VRAM: f}}, p)
			if f > 3<<30 && placementRank(d.Placement) < placementRank(prev.Placement) {
				n++
			}
			prev = d
		}
		return n
	}
	if n := falls(SolvePlacement); n > 0 {
		t.Errorf("the CUDA MoE load's placement fell %d times with more memory", n)
	}
	if n := falls(func(l DeviceLoad, r PlanRequest, m Machine, p Policy) DevicePlan {
		return solvePlacement(l, r, m, p, false)
	}); n == 0 {
		t.Error("the planner without the reserve does not fall on the CUDA load: the case no longer tests the reserve")
	}
	metalPerPos := int64(64 << 10)
	all := int64(3 << 30)
	metal := DeviceLoad{Backend: "metal", Tier: Unified, DenseBytes: 2 << 30, CtxStep: 8, CtxCeiling: 32768,
		KV: func(c int) int64 { return metalPerPos * int64((c+7)/8*8) }, HostFixed: 3 << 30,
		NExperts: 8, TopK: 2, ExpertBytesFull: 8 * (32 << 10), HostAll: func() int64 { return all + 100<<10 }}
	mc := MetalCtxLoad{Load: metal, Floor: 4096, Candidate: 16384, Slots: 1, FreeKnown: true, Pad: 8}
	var prev MetalCtx
	for f := int64(5 << 30); f < int64(9<<30); f += 64 << 10 {
		a := SolveMetalCtx(mc, Machine{Free: map[Tier]int64{Unified: f}}, p)
		if a.Ctx < prev.Ctx {
			t.Fatalf("the Metal MoE context fell from %d to %d at %d KiB free", prev.Ctx, a.Ctx, f>>10)
		}
		prev = a
	}
	if prev.Ctx <= mc.Floor {
		t.Errorf("the Metal MoE load never got past the floor (%d): the walk tests nothing", prev.Ctx)
	}
}
