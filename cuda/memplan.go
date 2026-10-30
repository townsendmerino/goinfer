//go:build cuda

package cuda

import (
	"errors"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// memReadings are the free-VRAM figures a resident build's memory decisions read, in the order the build reads them. Each
// figure has an ok flag: false means no reading, which each decision treats as the build does (it passes).
//   - WeightsOpen: what a MoE build's openForPlan checks the weights against, before moe_route's pool is paid. Read only by
//     a MoE build whose context is planned.
//   - Plan: what the context plan reads (resolveCtxCapFitFree), only when the context is planned. A MoE build reads it after
//     the pool is paid.
//   - Weights: what the setup job checks the weights against, against the planned margin.
//   - KV: what checkKVFits reads, after the weights and scratch are on the device.
//   - Ckpt: what the recurrent checkpoint is sized against, after the KV pools.
//   - GrantPre: what an expert-cache load's windowed grant reads, before the cache is sized.
//   - Slots: what the expert cache is sized against.
//   - Grant: what the windowed grant reads last, on a load without an expert cache.
//   - Prefill: what the prefill width is chosen from, once the build is done.
type memReadings struct {
	WeightsOpen, Plan, Weights, KV         int64
	WeightsOpenOK, PlanOK, WeightsOK, KVOK bool
	Ckpt, GrantPre, Slots, Grant, Prefill  int64
	CkptOK, GrantPreOK, SlotsOK, GrantOK   bool
	PrefillOK                              bool
}

// memPlan is a build's memory decisions over a memReadings: the context and margin the plan chose, the weights verdict,
// and the KV fit (the context after its trim, the slots granted, the refusal). DeclineAt names the decision that refused
// ("weightsOpen", "weights" or "kv"); empty when none did. A plan stopped by a weights refusal has no KV fields; one stopped at
// weightsOpen has no planned context either, and its Margin and ExtraBytes are what that check priced (the margin at the floor,
// the companion's weights). PoolPaid is
// true for a MoE build whose context is planned: its plan and later readings are taken after moe_route's pool is paid.
//
// The post-allocation decisions follow the KV fit: the recurrent checkpoint's bytes or the reason none is kept, the windowed slack after
// its grants, the expert-cache slots per layer (a refusal there is DeclineAt "slots"), and the prefill width. The plan assumes every
// allocation its figures admit succeeds; the build can still fail one and keep what it had. Whether batched prefill is ready is the
// build's (its kernels load); the plan gives the width its reading allows.
type memPlan struct {
	PoolPaid     bool
	CtxPlanned   int
	Margin       int64
	ExtraBytes   int64
	Decline      string
	DeclineAt    string
	CtxCap       int
	KVSlots      int
	KVSlotsReq   int
	CkptBytes    int64
	CkptDeclined string
	KVSlack      int
	CacheSlots   int
	PrefillWidth int
	Logs         []string
}

// memPlanRun carries a memory plan through buildResidentMode: the readings in, the record out. With derive set, only in.WeightsOpen
// is given and each later reading is derived from it as the plan goes (cudaBackend.deriveMemory); pool is moe_route's pool, taken
// off the plan's reading when the build pays it first.
type memPlanRun struct {
	in     memReadings
	out    memPlan
	done   bool
	derive bool
	pool   int64
}

// deriveMemory plans m's memory from one figure, p, the free VRAM before the load (what a machine profile records), deriving each
// later reading from p less what the plan says is on the device by then: the pool when the build pays it before its plan, the dense
// weights, the planned slack and the recurrent state before the KV fit; the KV pools and the post-pool allocations before the
// checkpoint; the checkpoint, the grants and the expert slots as each is granted. It returns the decisions and the readings it
// derived. What the plan does not price (the decode scratch before the KV pools, the driver's own pools) is the model's error.
func (b *cudaBackend) deriveMemory(m *decoder.Model, p, pool int64) (memPlan, memReadings, error) {
	mp := &memPlanRun{derive: true, pool: pool}
	mp.in = memReadings{WeightsOpen: p, Plan: p, Weights: p, WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true,
		CkptOK: true, GrantPreOK: true, SlotsOK: true, GrantOK: true, PrefillOK: true}
	_, _, err := b.buildResidentMode(m, false, mp)
	if mp.done {
		return mp.out, mp.in, nil
	}
	if err == nil {
		err = errors.New("cuda: memory plan: the build returned before its memory decisions")
	}
	return memPlan{}, mp.in, err
}

// errMemPlanned ends buildResidentMode once the memory plan's record is written.
var errMemPlanned = errors.New("cuda: memory plan")

// planMemory runs the build's planning on m with the free-VRAM readings in, and returns its memory decisions. It allocates
// nothing, opens no device and starts no executor. An error is a decline the build makes before any memory decision (a
// static one, the same as plan-only's).
func (b *cudaBackend) planMemory(m *decoder.Model, in memReadings) (memPlan, error) {
	mp := &memPlanRun{in: in}
	_, _, err := b.buildResidentMode(m, false, mp)
	if mp.done {
		return mp.out, nil
	}
	if err == nil {
		err = errors.New("cuda: memory plan: the build returned before its memory decisions")
	}
	return memPlan{}, err
}

// weightsCheck is the weights decision at site ("weightsOpen" or "weights") over that site's reading. A refusal ends the
// plan: it is recorded and errMemPlanned returned.
func (mp *memPlanRun) weightsCheck(m *decoder.Model, site string, margin int64) error {
	free, ok := mp.in.Weights, mp.in.WeightsOK
	if site == "weightsOpen" {
		free, ok = mp.in.WeightsOpen, mp.in.WeightsOpenOK
	}
	if e := weightsFitDecision(m.ResidentDenseWeightBytesFor("cuda"), m.ExtraResidentBytes(), free, ok, plantedMargin(site, margin)); e != nil {
		mp.out.Decline, mp.out.DeclineAt, mp.done = e.Error(), site, true
		return errMemPlanned
	}
	return nil
}

// finishMemPlan records the plan's context and margin, then takes the setup job's weights check and checkKVFits' decision over
// the planned layers, windowed as the setup job windows them. The KV pools are priced over the planned geometry; the
// built layers carry the same KV dimensions, precision and sharing (planLayerGeometry).
func (r *cudaResident) finishMemPlan(m *decoder.Model, hls []hlayer, planned []cudaLayer, mp *memPlanRun) error {
	mp.out.CtxPlanned, mp.out.Margin, mp.out.ExtraBytes, mp.out.KVSlotsReq = r.ctxPlanned, r.ctxMarginBytes, r.extraBytes, r.kvSlotsReq
	if e := mp.weightsCheck(m, "weights", r.ctxMargin()); e != nil {
		return e
	}
	r.layers = planned
	r.kvWindowSetup(m)
	if mp.derive {
		mp.in.KV = mp.in.Weights - m.ResidentDenseWeightBytesFor("cuda") - r.allocSlackBytes - r.recurrentStateCharged()
	}
	d := kvFitDecision(kvFitIn{ctxCap: r.ctxCap, explicit: r.ctxExplicit, slotsReq: r.kvSlotsReq, layers: r.layers,
		extraBytes: r.extraBytes, margin: r.ctxMargin()}, uint64(max(mp.in.KV, 0)), mp.in.KVOK)
	mp.out.CtxCap, mp.out.KVSlots, mp.out.Logs = d.ctxCap, d.slots, d.logs
	mp.done = true
	if d.err != nil {
		mp.out.Decline, mp.out.DeclineAt = d.err.Error(), "kv"
		return errMemPlanned
	}
	r.ctxCap, r.kvSlotsN = d.ctxCap, d.slots
	r.postAllocPlan(hls, mp)
	return errMemPlanned
}

// postAllocPlan takes the build's post-allocation decisions, in the build's order, each over its own reading: the recurrent
// checkpoint, an expert-cache load's grant, the expert-cache slots, the last grant, and the prefill width. r.planning stays set, so the
// pass prices assume the fused kernels the plan assumes for the margin.
func (r *cudaResident) postAllocPlan(hls []hlayer, mp *memPlanRun) {
	in, out := &mp.in, &mp.out
	plant := func(site string) bool { return slices.Contains(strings.Split(memPlanPlantForTest, ","), site) }
	r.planning = true
	firstRouted := -1
	if r.cacheExperts {
		r.planSlotGeom, firstRouted = planSlotGeomFor(hls, r.nE)
	}
	// left is the derived free VRAM as the post-allocation sites go: after the KV pools and the post-pool allocations, then less each
	// grant as it is made.
	left := in.KV - kvChargedBytes(r.ctxCap, r.kvSlotsN, r.layers) - r.postPoolBytes()
	derived := func(v *int64) {
		if mp.derive {
			*v = left
		}
	}
	derived(&in.Ckpt)
	if bytes := r.recurrentLiveBytes(); r.recurrent() && bytes > 0 {
		reserve := r.ckptReserve()
		if plant("ckpt") {
			reserve = 0
		}
		if why := ckptDeclineFor(bytes, reserve, uint64(max(in.Ckpt, 0)), unreadErr(in.CkptOK)); why != "" {
			out.CkptDeclined, out.Logs = why, append(out.Logs, ckptDeclinedLine(why))
		} else {
			out.CkptBytes, out.Logs = bytes, append(out.Logs, ckptKeptLine(bytes, 2*r.recurrentLayers()))
			left -= r.recurrentStateCharged()
		}
	}
	grant := func(site string, holdBack, free int64, ok bool) {
		if !r.kvWin || !ok || plant(site) {
			return
		}
		if extra, perPos := r.kvGrantFor(holdBack, uint64(max(free, 0))); extra > 0 {
			r.kvSlack += extra
			out.Logs = append(out.Logs, r.kvGrantLine(extra, perPos, uint64(max(free, 0))))
			left -= int64(extra) * perPos
		}
	}
	if r.cacheExperts {
		derived(&in.GrantPre)
		grant("grantPre", r.expertCacheFloorBytes(), in.GrantPre, in.GrantPreOK)
	}
	derived(&in.Slots)
	if r.cacheExperts && firstRouted >= 0 && r.planSlotGeom.n == 0 {
		// As allocSlots: a routed layer with no per-expert bytes turns the cache off, and the last grant then runs.
		r.cacheExperts = false
		out.Logs = append(out.Logs, zeroExpertBytesLine(firstRouted))
	}
	if n, strides := r.expertSlotGeom(); r.cacheExperts && n > 0 {
		sd := slotsDecision(uint64(max(in.Slots, 0)), in.SlotsOK && !plant("slots"), r.extraBytes, n, strides, r.topK, r.cacheSlots)
		out.Logs = append(out.Logs, sd.logs...)
		if sd.err != nil {
			out.Decline, out.DeclineAt = sd.err.Error(), "slots"
			return
		}
		r.cacheSlots = sd.slots
		left -= slotRequirement(r.cacheSlots, int64(n), strides)
	}
	out.CacheSlots = r.cacheSlots
	if !r.cacheExperts {
		derived(&in.Grant)
		grant("grant", 0, in.Grant, in.GrantOK)
	}
	derived(&in.Prefill)
	out.KVSlack = r.kvSlack
	out.PrefillWidth = prefillDefaultChunk
	if !r.kvWin && in.PrefillOK && !plant("prefill") {
		w, line := r.prefillWidthChoice(uint64(max(in.Prefill, 0)))
		out.PrefillWidth, out.Logs = w, append(out.Logs, line)
	}
}

// memPlanPlantForTest, when set (tests only), plants defects in the memory plan's post-allocation decisions only, a comma-separated
// list of exact site names: "ckpt" (the checkpoint's reserve dropped), "grant" (no windowed grant on a load without an expert cache),
// "grantPre" (none before an expert cache), "slots" (the slot cap ignored), "prefill" (the width not chosen). G-PA3 shows each turns
// the plan against the build. Like the package's other test seams it is a plain variable: tests that set it do not run in parallel.
var memPlanPlantForTest string

// planSlotGeomFor is the expert cache's geometry from the planned stacks: every layer whose experts the build caches (a routed MoE
// layer, or Gemma 4's MoE block), and the per-slot strides of the first (perExpertWords over its gate-up and down stacks, as cacheWQ
// sizes them). As the build's expertSlotGeom, it has no layers when the first one has no per-expert bytes (or there is no expert
// count); first is that layer's index, or -1 when no layer is cached.
func planSlotGeomFor(hls []hlayer, nE int) (g *slotGeom, first int) {
	g, first = &slotGeom{}, -1
	for l := range hls {
		if !hls[l].isMoE && !hls[l].g4moe {
			continue
		}
		if first < 0 {
			first = l
			if nE > 0 {
				guW, guS := perExpertWords(hls[l].expGU.N, hls[l].expGU.K, nE)
				dnW, dnS := perExpertWords(hls[l].expDown.N, hls[l].expDown.K, nE)
				g.strides = []int64{int64(guW) * 4, int64(guS) * 2, int64(dnW) * 4, int64(dnS) * 2}
			}
		}
		g.n++
	}
	var per int64
	for _, s := range g.strides {
		per += s
	}
	if per <= 0 {
		g.n, g.strides = 0, nil
	}
	return g, first
}

// unreadErr is the error a reading's absence stands for.
func unreadErr(ok bool) error {
	if ok {
		return nil
	}
	return errors.New("no reading")
}
