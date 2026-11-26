//go:build cuda

package cuda

import (
	"github.com/townsendmerino/goinfer/backend/cuda/internal/fitdecide"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// resolveCtxCapFit is the "fit by default" resident context for CUDA:
// an UNPINNED load asks Plan for a larger candidate (fitDefaultCtx, clamped to the model's window) instead of a
// flat cudaCtxCapDefault, and uses whatever Plan lands on. Because of the floor checks below that is never less
// than cudaCtxCapDefault, except under a guard pin (below). A caller with m.FitDisabled() (--fit=off), or whose
// free-VRAM probe is unknown, gets exactly resolveCtxCap, so this can only improve on the historical default.
//
// m.MoECacheExperts() opts out the same way. A MoE-cache-experts load has a second, elastic claimant on free
// VRAM, the host-to-VRAM expert-slot cache, whose design is "whatever is left after everything pinned"; every
// byte handed to KV is a byte the expert cache never sees, and growing the context there cut its slots and
// decode speed sharply (docs/code-notes/cuda.md#resolveCtxCapFit). Unlike a drafter's fixed cost it cannot be
// priced into ExtraBytes, so the context must not grow into that leftover at all.
//
// slots is the requested resident KV slot count (cudaKVSlotsRequest). Above 1, the unpinned choice gives up
// context until every slot fits (ctxForSlots), never below cudaCtxCapDefault, or below request when request is
// itself a guard pin under that default.
//
// request > 0 is NOT always a choice. decoder.Model.ResidentContextPinned() tells a genuine -ctx from the
// load-time fit guard auto-pinning a smaller context for an UNREQUESTED load; request still reports that pin
// (ResidentContextRequest's doc), but it is a ceiling already proved safe against host RAM, not the caller's
// decision. A guard pin gets fit-by-default and the slots rule too, with its ceiling and floor both following
// it down, never up: decoder.ctxFloor (2048) can sit under cudaCtxCapDefault, and raising a guard-shrunk
// context back to cudaCtxCapDefault is exactly the regression the guard pinned it to prevent.
//
// margin is the context margin at a context (cudaResident.ctxMarginFor): the plan subtracts it at the widest context it asks for,
// and the figure at the context returned is returned beside it, for checkKVFits to keep.
func resolveCtxCapFit(m *decoder.Model, request, modelCtx, slots int, margin func(ctx int) int64) (int, int64) {
	return resolveCtxCapFitSlack(m, request, modelCtx, slots, margin, 0)
}

// resolveCtxCapFitSlack is resolveCtxCapFit with the KV pools' rounding bound. The recurrent state and the driver's rounding over the
// weight buffers are not arguments: Plan prices both itself (decoder.BuildDemand.StateBytes, decoder.Model.PlanSlack, planslack.go). The
// state is not in checkKVFits: DeltaNet's and Mamba-2's are allocated at upload, so its probe reads free memory after them.
//
// kvRound bounds what the device's rounding of the KV pools adds to Plan's linear KV figure (kvRoundingBound): the plan reserves
// it beside the margin, so the context it picks fits the build's exact, charged figure (checkKVFits).
func resolveCtxCapFitSlack(m *decoder.Model, request, modelCtx, slots int, margin func(ctx int) int64, kvRound int64) (int, int64) {
	return resolveCtxCapFitFree(m, request, modelCtx, slots, margin, kvRound, liveCtxFree)
}

// liveCtxFree is the context plan's live free-VRAM reading, the device probe; it reports the reading to the test hooks, which
// record the real build's readings and so are not called for a supplied figure.
func liveCtxFree() (int64, bool) {
	free, ok := decoder.FreeBytesFor("cuda")
	if memReadSeenForTest != nil {
		memReadSeenForTest("plan", free, ok)
	}
	if ok && planFreeSeenForTest != nil {
		planFreeSeenForTest(free)
	}
	return free, ok
}

// resolveCtxCapFitFree is resolveCtxCapFitSlack with the free-VRAM reading taken from readFree, which is called only when the
// context is planned: the build passes the device probe, the memory plan (cudaBackend.planMemory) a supplied figure.
func resolveCtxCapFitFree(m *decoder.Model, request, modelCtx, slots int, margin func(ctx int) int64, kvRound int64, readFree func() (int64, bool)) (int, int64) {
	candidate, floor, plans := ctxPlanRange(m, request, modelCtx)
	if !plans {
		c := floor
		if candidate == 0 {
			c = resolveCtxCap(request, modelCtx) // a genuine explicit -ctx is untouched either way
		}
		return c, margin(c)
	}
	// The margin at candidate, the widest context this call asks Plan for; the build keeps the figure at the context returned.
	mb := margin(candidate)
	free, ok := readFree()
	if !ok {
		return floor, margin(floor) // unknown ⇒ the safe floor, never guess
	}
	// Plan's chooseCtx reserves NO margin, but checkKVFits requires the margin beyond dense+KV+extra. Subtracting the same
	// margin here, with the KV pools' rounding bound, before Plan sees the free bytes, makes the two checks agree: whatever ctx
	// Plan picks already leaves room for it (otherwise an interior choice failed checkKVFits and the whole build declined to
	// CPU-only). Floored at 0 on a tiny probe; Plan declines an unfittable budget cleanly.
	marginedFree := max(free-mb-kvRound, 0)
	// ExtraBytes prices what attaches after BuildResident (m.ExtraResidentBytes: a --drafter) so the context Plan
	// chooses leaves it room; zero when nothing attaches. The drafter's device K/V is priced at candidate, the widest
	// ctx this call asks of Plan: Plan only shrinks from the value it is given (fitplan.go, "never GROW past what was
	// asked"), so that can over-estimate the real cost but never under-price it.
	extraBytes := m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(candidate)
	p := m.Plan("cuda", marginedFree, fit.PlanRequest{Ctx: candidate, ExtraBytes: extraBytes})
	if p.Placement == fit.PlacementDecline || p.Ctx < floor {
		return floor, margin(floor) // Plan could not confidently improve on the floor
	}
	c := p.Ctx
	if slots > 1 {
		c = ctxForSlots(m, marginedFree, extraBytes, p.Ctx, slots, floor)
	}
	return c, margin(c)
}

// ctxPlanRange is resolveCtxCapFitSlack's context range: the widest context it asks Plan for and the floor, and whether it asks
// Plan at all (and so reads free VRAM). candidate is 0 when the context is not planned (a genuine pin, fit disabled, the expert
// cache): resolveCtxCap decides it. A MoE build pays its local-memory pool before the plan only when plans is true (openForPlan).
func ctxPlanRange(m *decoder.Model, request, modelCtx int) (candidate, floor int, plans bool) {
	if (request > 0 && m.ResidentContextPinned()) || m.FitDisabled() || m.MoECacheExperts() {
		return 0, 0, false
	}
	candidate = fitDefaultCtx
	if modelCtx > 0 && candidate > modelCtx {
		candidate = modelCtx
	}
	floor = cudaCtxCapDefault
	if request > 0 {
		// A guard pin (request>0, not ResidentContextPinned — the branch above already returned
		// otherwise): the ceiling AND the floor both follow it down, never up. Growing past it
		// asks Plan for more than the guard already determined host RAM can hold; flooring above
		// it (at the historical cudaCtxCapDefault) would do the same by a different route.
		if candidate > request {
			candidate = request
		}
		if floor > request {
			floor = request
		}
	}
	return candidate, floor, candidate > floor // at or below the floor there is nothing to gain from asking Plan
}

// planFreeSeenForTest, when set (tests only), is told the free VRAM resolveCtxCapFitSlack hands Plan.
var planFreeSeenForTest func(free int64)

// cudaExtraSlotKVBytes is the KV one more resident slot of m costs at ctx positions, at the precision the build allocates: what ResolveKV
// delivers on CUDA (the backend's kvI8 reads the same), not the --kv request, which a policy may hold at f32 (a drafter).
func cudaExtraSlotKVBytes(m *decoder.Model, ctx int) int64 {
	return m.CUDAKVSlotBytes(ctx, m.KVCacheF16(), m.KVDeliveredFor("cuda") == mem.KVI8)
}

// ctxForSlots is resolveCtxCapFit's answer when several resident KV slots are requested:
// the largest context in [floor, oneSlot] at which Plan fits
// every requested slot (the build's own KV plus slots-1 more copies, priced as ExtraBytes), so an unpinned load
// gives up context before conversations (owner decision). When not even floor holds them all it returns floor and
// checkKVFits clamps the count. An explicit -ctx never reaches here.
//
// floor is cudaCtxCapDefault, unless resolveCtxCapFit's caller passed a guard-pinned request below it: the slots
// rule may shrink a guard-pinned load like an unpinned one, but never past the guard's own safety floor.
//
// KV is exactly linear in the context and Plan's other terms do not grow with it, so "fits" is monotone and a
// binary search finds the edge. Plan's weight figure is conservative, so the result can land lower than the build
// would allow (docs/code-notes/cuda.md#ctxForSlots).
func ctxForSlots(m *decoder.Model, marginedFree, extraBytes int64, oneSlot, slots, floor int) int {
	stateBytes := m.Plan("cuda", 1<<40, fit.PlanRequest{Ctx: 1}).StateBytes // the plan's own line is one slot's
	fits := func(ctx int) bool {
		more := int64(slots-1) * (cudaExtraSlotKVBytes(m, ctx) + stateBytes) // each further slot, at the load's own KV precision, with its recurrent state
		p := m.Plan("cuda", marginedFree, fit.PlanRequest{Ctx: ctx, CtxPinned: true, ExtraBytes: extraBytes + more})
		return p.Placement != fit.PlacementDecline
	}
	if fits(oneSlot) {
		return oneSlot
	}
	if !fits(floor) {
		logs.Debugf("cuda: resident context %d (the floor), not %d, for the %d requested KV slots (--kv-sessions); "+
			"the build grants as many as fit there, and an explicit --ctx keeps a longer context with fewer slots",
			floor, oneSlot, slots)
		return floor
	}
	lo, hi := floor, oneSlot // fits(lo), !fits(hi)
	for hi-lo > 1 {
		if mid := lo + (hi-lo)/2; fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	logs.Warnf("cuda: resident context %d, not %d, so the %d requested KV slots fit (--kv-sessions); an "+
		"explicit --ctx keeps a longer context with fewer slots", lo, oneSlot, slots)
	return lo
}

// errKVWontFit marks the one setup failure that must NOT degrade quietly to the staged path: an explicitly configured resident
// context whose KV does not fit (fitdecide.ErrKVWontFit). BuildResident turns it into a hard startup error instead of a decline.
var errKVWontFit = fitdecide.ErrKVWontFit

type kvWontFitError = fitdecide.KVWontFitError

// checkKVFits fails the LOAD if the KV the configured cap implies does not fit beside what is already on the device.
// It runs after the weights are uploaded and before the K/V caches are allocated, so `free` is genuinely "what is
// left for KV". The point is the failure MODE, not the arithmetic: without it an over-large cap surfaces as an
// allocation failure part-way through sizing the per-layer caches, or as an OOM mid-decode after the server reported
// ready. Naming the number at startup turns a production incident into a config error: the message states what was
// asked for, what it costs, and what is actually free.
//
// The KV is priced as the card charges the pools (kvChargedBytes), and the margin is the build's (ctxMargin).
//
// It also sets r.kvSlotsN, the resident KV slot count: the request, clamped (kvChargedSlotsFit) to the slots whose pools fit the same
// budget, and logged when clamped. The first slot is the one this check refuses; further ones only shrink.
func (r *cudaResident) checkKVFits() error {
	if recurrentStateAtKVFitForTest != nil {
		if l, ok := r.dev.Leaser().(*deviceLedger); ok {
			recurrentStateAtKVFitForTest(l.Usage().ByLine[mem.LineState])
		}
	}
	free, err := cudaFreeVRAM(r)
	if memReadSeenForTest != nil {
		memReadSeenForTest("kv", int64(free), err == nil)
	}
	d := kvFitDecision(kvFitIn{ctxCap: r.ctxCap, explicit: r.ctxExplicit, slotsReq: r.kvSlotsReq, layers: r.layers,
		extraBytes: r.extraBytes, margin: r.ctxMargin(), slotState: r.recurrentStateCharged()}, free, err == nil)
	for _, l := range d.logs {
		logs.InfoText(l)
	}
	r.ctxCap, r.kvSlotsN = d.ctxCap, d.slots
	return d.err
}

// kvFitIn is what checkKVFits decides from besides the free VRAM: the planned context and whether it was asked for, the KV
// slots requested, the layers the pools are priced over, the companion reserve, the context margin and a further slot's recurrent state.
type kvFitIn struct {
	ctxCap     int
	explicit   bool
	slotsReq   int
	layers     []cudaLayer
	extraBytes int64
	margin     int64
	slotState  int64 // each KV slot past the first's recurrent state (recurrentStateCharged); 0 for a model with none
}

// kvFitOut is checkKVFits' decision: the context the pools are sized for, the slots granted (>= 1), the lines the build logs
// (each ends in a newline) and the refusal, if any.
type kvFitOut struct {
	ctxCap int
	slots  int
	logs   []string
	err    error
}

// kvFitDecision is checkKVFits' decision over the free VRAM it read (fitdecide.KVFitDecision), priced by kvChargedBytes over
// in.layers.
func kvFitDecision(in kvFitIn, free uint64, ok bool) kvFitOut {
	d := fitdecide.KVFitDecision(fitdecide.KVFitIn{CtxCap: in.ctxCap, Explicit: in.explicit, SlotsReq: in.slotsReq,
		Charged: func(cap, slots int) int64 { return kvChargedBytes(cap, slots, in.layers) }, Layers: len(in.layers),
		NoTrim: kvTrimOffForTest, ExtraBytes: in.extraBytes, Margin: in.margin, SlotState: in.slotState}, free, ok)
	return kvFitOut{ctxCap: d.CtxCap, slots: d.Slots, logs: d.Logs, err: d.Err}
}

// weightsMarginPlantSiteForTest, when set (tests only) to a weights-check site ("weightsOpen" or "weights"), prices that check's
// margin a thousandfold: TestBackendMemoryCensus_cuda_plant's plant.
var weightsMarginPlantSiteForTest string

// plantedMargin is margin, or the planted one at weightsMarginPlantSiteForTest's site.
func plantedMargin(site string, margin int64) int64 {
	if site == weightsMarginPlantSiteForTest {
		return margin * 1000
	}
	return margin
}

// kvTrimOffForTest, when set (tests only), turns kvFitDecision's one-slot trim off: TestKVFitDecision_trimPlant's plant.
var kvTrimOffForTest bool

// memReadSeenForTest, when set (tests only), is called with every free-VRAM reading a build's memory decisions take: the site
// ("weightsOpen", "plan", "weights", "kv"), the figure and whether the read succeeded.
var memReadSeenForTest func(site string, free int64, ok bool)

// planState is what the memory plan and the fit checks decided for cudaResident: the context cap and how it was set, the
// margins and companion bytes priced into every free-VRAM check, the dense weights' arena, and the planning flags.
type planState struct {
	// planning is set while BuildResident prices the margin on the planned geometry, before any module loads: the span routing then
	// assumes the row-batched kernels and, under fastAttn, the fused attention for head dims 64 and 128 (fusedLoadedFor).
	planning bool
	// openedForPlan is set when a MoE build opened its device and stream before the plan (openForPlan); the setup job keeps them.
	openedForPlan   bool
	ctxExplicit     bool             // the cap came from configuration (Options.ResidentContext), not the default — decides whether a VRAM miss is a hard error or a decline
	ctxCap          int              // effective resident KV capacity in positions = resolveCtxCap(request, model ctx). Every kc/vc is sized cap*kvDim; checkCap guards against it.
	extraBytes      int64            // decoder.Model.ExtraResidentBytes() at construction — a companion allocation (--drafter) reserved out of every free-VRAM check this resident makes after it (checkWeightsFit, checkKVFits, allocSlots)
	ctxPlanned      int              // ctxCap as the plan chose it, before checkKVFits can trim it against the real free VRAM
	allocSlackBytes int64            // the driver's allocation rounding over the packed weights, priced into the plan (packedAllocSlack)
	wArena          *weightArena     // the dense weights' arena while uploadLayers sub-allocates from it; nil otherwise
	wArenaStats     weightArenaStats // what the arena held when uploadLayers closed it
	ctxMarginBytes  int64            // the context margin at load (ctxMarginFor at ctxMarginCtx), read by checkKVFits, checkWeightsFit and their declines
	ctxMarginCtx    int              // the context the margin is priced at: the one the plan returned
}
