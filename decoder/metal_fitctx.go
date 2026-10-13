package decoder

import "fmt"

// MetalDefaultKVSlots is how many resident KV slots Metal keeps when the count asked is the caller's default, not the operator's
// (Options.ResidentKVSlotsDefault). `goinfer-chat fit` prices the same count, so its answer is the one a default serve load reaches.
const MetalDefaultKVSlots = 2

// MetalFitMarginBytes is the memory MetalFitCtx leaves under the budget it is given. Metal's load-time guard reads the live budget
// again after the context is chosen, and a context taken at the exact edge would be halved by it on a drift of a few megabytes
// (docs/code-notes/metal.md#metalFitFor).
const MetalFitMarginBytes = 384 << 20

// MetalFitInput is what MetalFitCtx decides from. The caller resolves each field, so the Metal load and `goinfer-chat fit` can hand it
// the same facts and get the same context.
type MetalFitInput struct {
	FreeBytes int64 // the Metal memory budget (the registered "metal" probe: the live ceiling), before the margin; read only when FreeKnown
	FreeKnown bool
	KVSlots   int  // resident KV slots the load asks for, 1 or more: each holds a whole KV cache at the chosen context
	Lower     int  // a context the choice may not exceed and that lowers the floor with it: a guard pin, or what a memory decline shrank to; 0 for none
	PagedMoE  bool // the load pages its routed experts, whose pager lives on the memory a larger context would take
}

// MetalFit is MetalFitCtx's answer: the context, the KV it costs across the slots priced, and why, in a few words for a banner.
type MetalFit struct {
	Ctx     int
	KVSlots int
	KVBytes int64
	Why     string
}

// MetalFitCtx is the resident context an unpinned Metal load takes (fit by default, CUDA's resolveCtxCapFit shape): it asks Plan for
// MetalFitDefaultCtx positions, clamped to the model's window, and gives up context until every requested KV slot fits beside the
// weights and MetalFitMarginBytes. The answer is never under MetalCtxDefault, except that in.Lower caps the candidate and the floor alike: it only lowers.
// It is MetalCtxDefault (or in.Lower under it) when the fit is off, the load pages MoE experts, the memory budget is unknown, or
// nothing larger fits. An explicit -ctx is not this function's business: the caller honours it as given.
//
// KV is priced as Metal allocates it for this model (ResidentKVBytes), and a companion's bytes (ExtraResidentBytes) as CUDA prices them.
func (m *Model) MetalFitCtx(in MetalFitInput) MetalFit {
	slots := max(1, in.KVSlots)
	floor, candidate := MetalCtxDefault, MetalFitDefaultCtx
	if w := m.Config().MaxPositions; w > 0 && candidate > w {
		candidate = w
	}
	lowered := false
	if in.Lower > 0 {
		lowered = in.Lower < candidate
		candidate, floor = min(candidate, in.Lower), min(floor, in.Lower)
	}
	answer := func(ctx int, why string) MetalFit {
		return MetalFit{Ctx: ctx, KVSlots: slots, KVBytes: int64(slots) * m.ResidentKVBytes("metal", ctx, false, false), Why: why}
	}
	switch {
	case m.FitDisabled():
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
	free := max(in.FreeBytes-MetalFitMarginBytes, 0)
	// A companion's K/V is priced at the candidate, the widest context asked of Plan: it can over-price, never under-price.
	extra := m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(candidate)
	fits := func(ctx int) bool {
		more := int64(slots-1) * m.ResidentKVBytes("metal", ctx, false, false)
		p := m.Plan("metal", free, PlanRequest{Ctx: ctx, CtxPinned: true, ExtraBytes: extra + more})
		return p.Placement == PlacementResident // an expert cache is a placement the Metal load does not take by itself
	}
	one := m.Plan("metal", free, PlanRequest{Ctx: candidate, ExtraBytes: extra})
	if one.Placement != PlacementResident || one.Ctx <= floor {
		return answer(floor, "the default: a longer context does not fit this machine's memory")
	}
	ctx := one.Ctx
	if !fits(ctx) {
		if !fits(floor) {
			return answer(floor, fmt.Sprintf("the default: %d KV slots leave no room for more", slots))
		}
		lo, hi := floor/metalKVPad, ctx/metalKVPad // in Metal's 8-position steps: fits(lo), not fits(hi)
		for hi-lo > 1 {
			if mid := lo + (hi-lo)/2; fits(mid * metalKVPad) {
				lo = mid
			} else {
				hi = mid
			}
		}
		ctx = lo * metalKVPad
	}
	a := answer(ctx, "")
	a.Why = MetalFitWhy(slots, a.KVBytes)
	return a
}

// MetalFitWhy is MetalFitCtx's reason for a context it grew past the default: the slots it holds and what their KV costs.
func MetalFitWhy(slots int, kvBytes int64) string {
	plural := "s"
	if slots == 1 {
		plural = ""
	}
	return fmt.Sprintf("fit: %d slot%s, %.1f GB KV", slots, plural, float64(kvBytes)/(1<<30))
}
