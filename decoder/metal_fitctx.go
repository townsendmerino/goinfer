package decoder

import "github.com/townsendmerino/goinfer/internal/mem"

// MetalDefaultKVSlots is how many resident KV slots Metal keeps when the count asked is the caller's default, not the operator's
// (Options.ResidentKVSlotsDefault). `goinfer-chat fit` prices the same count, so its answer is the one a default serve load reaches.
const MetalDefaultKVSlots = 2

// MetalFitMarginBytes is the memory MetalFitCtx leaves under the budget it is given. Metal's load-time guard reads the live budget
// again after the context is chosen, and a context taken at the exact edge would be halved by it on a drift of a few megabytes
// (docs/code-notes/metal.md#metalFitFor).
const MetalFitMarginBytes = mem.MetalFitMargin

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
// The load carries the pinned ceiling (MetalCtxCeiling); the candidate, at most MetalFitDefaultCtx, bounds every context asked.
func (m *Model) MetalFitCtx(in MetalFitInput) MetalFit {
	candidate := MetalFitDefaultCtx
	if w := m.Config().MaxPositions; w > 0 && candidate > w {
		candidate = w
	}
	a := mem.SolveMetalCtx(mem.MetalCtxLoad{
		Load: m.deviceLoad("metal", false, false, true), Floor: MetalCtxDefault, Candidate: candidate, Lower: in.Lower, Slots: in.KVSlots,
		FitDisabled: m.FitDisabled(), PagedMoE: in.PagedMoE, FreeKnown: in.FreeKnown, Ineligible: m.planIneligible("metal", PlanRequest{}) != "",
		ExtraBytes: m.ExtraResidentBytes(), ExtraKVPerPosition: m.ExtraResidentKVPerPosition(), Pad: metalKVPad,
	}, mem.Machine{Free: map[mem.Tier]int64{mem.Unified: in.FreeBytes}}, mem.DefaultPolicy())
	return MetalFit{Ctx: a.Ctx, KVSlots: a.Slots, KVBytes: a.KVBytes, Why: a.Why}
}

// MetalFitWhy is MetalFitCtx's reason for a context it grew past the default: the slots it holds and what their KV costs.
func MetalFitWhy(slots int, kvBytes int64) string { return mem.MetalFitWhy(slots, kvBytes) }
