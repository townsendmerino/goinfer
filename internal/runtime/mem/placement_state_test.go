package mem

import (
	"math/rand/v2"
	"testing"
)

// TestSolvePlacement_statePricedOnEveryPath: state is a fixed term, so a load holding it answers as the same load with those bytes in
// PlanRequest.ExtraBytes, on every placement and in every search. Only the reason and the state line itself differ.
func TestSolvePlacement_statePricedOnEveryPath(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 3))
	p := DefaultPolicy()
	placements := map[Placement]int{}
	for range 4000 {
		l, req := genDevice(r)
		m := Machine{Free: map[Tier]int64{VRAM: r.Int64N(64 * genGiB), Unified: r.Int64N(32 * genGiB), HostAnon: r.Int64N(64 * genGiB)}}
		state := r.Int64N(4*genGiB) + 1
		bare := SolvePlacement(l, PlanRequest{Ctx: req.Ctx, CtxPinned: req.CtxPinned, Slots: req.Slots, ExtraBytes: req.ExtraBytes + state}, m, p)
		l.StateBytes = state
		held := SolvePlacement(l, req, m, p)
		placements[held.Placement]++
		if held.StateBytes != state {
			t.Fatalf("the plan's state line is %d, the load's state is %d", held.StateBytes, state)
		}
		held.StateBytes, held.Reason, bare.Reason = 0, "", ""
		if held != bare {
			t.Fatalf("state %d: held %+v, as extra bytes %+v", state, held, bare)
		}
	}
	for _, want := range []Placement{Resident, ExpertCached, WeightPaged, Declined} {
		if placements[want] == 0 {
			t.Errorf("no draw reached placement %d (%v): the comparison checks nothing there", want, placements)
		}
	}
}

// TestSolvePlacement_stateNamedInTheReason: a plan that holds state says so, and one with none reads as before.
func TestSolvePlacement_stateNamedInTheReason(t *testing.T) {
	l := DeviceLoad{Backend: "webgpu", Tier: VRAM, DenseBytes: 1 << 30, CtxStep: 1, KV: func(ctx int) int64 { return int64(ctx) << 10 }}
	m := Machine{Free: map[Tier]int64{VRAM: 8 << 30}}
	none := SolvePlacement(l, PlanRequest{Ctx: 4096}, m, DefaultPolicy())
	if want := "dense 1.00 GB + KV@4096 0.00 GB fits 8.00 GB free"; none.Reason != want {
		t.Errorf("no state: %q, want %q", none.Reason, want)
	}
	l.StateBytes = 1 << 30
	if got, want := SolvePlacement(l, PlanRequest{Ctx: 4096}, m, DefaultPolicy()).Reason, "dense 1.00 GB + recurrent state 1.00 GB + KV@4096 0.00 GB fits 8.00 GB free"; got != want {
		t.Errorf("state: %q, want %q", got, want)
	}
}
