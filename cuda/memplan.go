//go:build cuda

package cuda

import (
	"errors"

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
type memReadings struct {
	WeightsOpen, Plan, Weights, KV         int64
	WeightsOpenOK, PlanOK, WeightsOK, KVOK bool
}

// memPlan is a build's memory decisions over a memReadings: the context and margin the plan chose, the weights verdict,
// and the KV fit (the context after its trim, the slots granted, the refusal). DeclineAt names the decision that refused
// ("weightsOpen", "weights" or "kv"); empty when none did. A plan stopped by a weights refusal has no KV fields; one stopped at
// weightsOpen has no planned context either, and its Margin and ExtraBytes are what that check priced (the margin at the floor,
// the companion's weights). PoolPaid is
// true for a MoE build whose context is planned: its plan and later readings are taken after moe_route's pool is paid.
type memPlan struct {
	PoolPaid   bool
	CtxPlanned int
	Margin     int64
	ExtraBytes int64
	Decline    string
	DeclineAt  string
	CtxCap     int
	KVSlots    int
	KVSlotsReq int
	Logs       []string
}

// memPlanRun carries a memory plan through buildResidentMode: the readings in, the record out.
type memPlanRun struct {
	in   memReadings
	out  memPlan
	done bool
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
	if e := weightsFitDecision(m.ResidentDenseWeightBytesFor("cuda"), m.ExtraResidentBytes(), free, ok, margin); e != nil {
		mp.out.Decline, mp.out.DeclineAt, mp.done = e.Error(), site, true
		return errMemPlanned
	}
	return nil
}

// finishMemPlan records the plan's context and margin, then takes the setup job's weights check and checkKVFits' decision over
// the planned layers, windowed as the setup job windows them. The KV pools are priced over the planned geometry; the
// built layers carry the same KV dimensions, precision and sharing (planLayerGeometry).
func (r *cudaResident) finishMemPlan(m *decoder.Model, planned []cudaLayer, mp *memPlanRun) error {
	mp.out.CtxPlanned, mp.out.Margin, mp.out.ExtraBytes, mp.out.KVSlotsReq = r.ctxPlanned, r.ctxMarginBytes, r.extraBytes, r.kvSlotsReq
	if e := mp.weightsCheck(m, "weights", r.ctxMargin()); e != nil {
		return e
	}
	r.layers = planned
	r.kvWindowSetup(m)
	d := kvFitDecision(kvFitIn{ctxCap: r.ctxCap, explicit: r.ctxExplicit, slotsReq: r.kvSlotsReq, layers: r.layers,
		extraBytes: r.extraBytes, margin: r.ctxMargin()}, uint64(max(mp.in.KV, 0)), mp.in.KVOK)
	mp.out.CtxCap, mp.out.KVSlots, mp.out.Logs = d.ctxCap, d.slots, d.logs
	if d.err != nil {
		mp.out.Decline, mp.out.DeclineAt = d.err.Error(), "kv"
	}
	mp.done = true
	return errMemPlanned
}
