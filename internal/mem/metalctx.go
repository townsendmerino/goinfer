package mem

import "fmt"

// MetalCtxLoad is what MetalFitCtx decides an unpinned Metal load's context from, every term resolved by the caller.
type MetalCtxLoad struct {
	Load        DeviceLoad // the model on Metal, as SolvePlacement prices it
	Floor       int        // the context an unpinned load never goes under: Metal's default
	Candidate   int        // the context asked for first: Metal's fit default, clamped to the model's window
	Lower       int        // a context the answer may not exceed, which lowers the floor with it; 0 for none
	Slots       int        // resident KV slots, each a whole KV cache at the context; 1 or more
	FitDisabled bool       // --fit=off
	PagedMoE    bool       // the load pages its routed experts, whose pager lives on the memory a context would take
	FreeKnown   bool       // the Machine's Unified reading is real
	Ineligible  bool       // Metal has no resident path for this model: the planner declines every context
	// ExtraBytes and ExtraKVPerPosition are a companion's (a drafter's) weights and K/V per position, priced at the candidate.
	ExtraBytes, ExtraKVPerPosition int64
	Pad                            int // Metal's KV grid, 8 positions
}

// MetalCtx is SolveMetalCtx's answer: the context, the slots, the KV they cost together, and why in a few words.
type MetalCtx struct {
	Ctx, Slots int
	KVBytes    int64
	Why        string
}

// MetalFitWhy is the reason for a context grown past the default: the slots it holds and what their KV costs.
func MetalFitWhy(slots int, kvBytes int64) string {
	plural := "s"
	if slots == 1 {
		plural = ""
	}
	return fmt.Sprintf("fit: %d slot%s, %.1f GB KV", slots, plural, float64(kvBytes)/(1<<30))
}

// SolveMetalCtx is the context an unpinned Metal load takes: it asks the planner for the candidate and gives up context until
// every slot fits beside the weights and the policy's margin, never under the floor (which Lower may only lower). An explicit
// -ctx is not its business.
func SolveMetalCtx(in MetalCtxLoad, m Machine, pol Policy) MetalCtx {
	slots := max(1, in.Slots)
	floor, candidate := in.Floor, in.Candidate
	lowered := false
	if in.Lower > 0 {
		lowered = in.Lower < candidate
		candidate, floor = min(candidate, in.Lower), min(floor, in.Lower)
	}
	answer := func(ctx int, why string) MetalCtx {
		return MetalCtx{Ctx: ctx, Slots: slots, KVBytes: int64(slots) * in.Load.KV(ctx), Why: why}
	}
	switch {
	case in.FitDisabled:
		return answer(floor, "--fit=off keeps the default")
	case in.PagedMoE:
		return answer(floor, "paged MoE keeps the default")
	case candidate <= floor && lowered:
		return answer(floor, "the load-time memory guard's limit")
	case candidate <= floor:
		return answer(floor, "the default")
	case !in.FreeKnown:
		return answer(floor, "the default: this machine's memory could not be read")
	}
	budget := Machine{Free: map[Tier]int64{in.Load.Tier: max(m.Available(in.Load.Tier)-pol.MetalFitMargin, 0)}}
	// A companion's K/V is priced at the candidate, the widest context asked of the planner: it can over-price, never under-price.
	extra := in.ExtraBytes + in.ExtraKVPerPosition*int64(candidate)
	fits := func(ctx int) bool {
		more := int64(slots-1) * in.Load.KV(ctx)
		p := SolvePlacement(in.Load, PlanRequest{Ctx: ctx, CtxPinned: true, ExtraBytes: extra + more}, budget, pol)
		return p.Placement == Resident // an expert cache is a placement the Metal load does not take by itself
	}
	if in.Ineligible {
		return answer(floor, "the default: a longer context does not fit this machine's memory")
	}
	one := SolvePlacement(in.Load, PlanRequest{Ctx: candidate, ExtraBytes: extra}, budget, pol)
	if one.Placement != Resident || one.Ctx <= floor {
		return answer(floor, "the default: a longer context does not fit this machine's memory")
	}
	ctx := one.Ctx
	if !fits(ctx) {
		if !fits(floor) {
			return answer(floor, fmt.Sprintf("the default: %d KV slots leave no room for more", slots))
		}
		pad := max(in.Pad, 1)
		lo, hi := floor/pad, ctx/pad // on Metal's grid: fits(lo), not fits(hi)
		for hi-lo > 1 {
			if mid := lo + (hi-lo)/2; fits(mid * pad) {
				lo = mid
			} else {
				hi = mid
			}
		}
		ctx = lo * pad
	}
	a := answer(ctx, "")
	a.Why = MetalFitWhy(slots, a.KVBytes)
	return a
}
