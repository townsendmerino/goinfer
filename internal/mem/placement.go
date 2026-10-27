package mem

import "fmt"

// Placement is where a device plan puts a model's weights.
type Placement int

const (
	Resident            Placement = iota
	ExpertCached                  // the routed experts in a cache of Slots per layer
	HostComputedExperts           // reserved: never chosen
	WeightPaged                   // the CPU pages weights from disk: its fallback for what does not fit
	Declined
)

// DeviceLoad is what a resident build puts on one backend's memory, as the caller priced it from the loaded model.
type DeviceLoad struct {
	Backend string
	Tier    Tier // the Machine tier whose reading is the backend's free memory

	DenseBytes      int64 // the fixed term: every non-expert matrix the backend holds
	ExpertBytesFull int64 // every routed expert, unpaged; 0 for a dense model
	NExperts, TopK  int   // per layer; NExperts 0 for a dense model
	// HostFixed is the host copy a unified-memory backend holds whatever the expert cache does (the dense weights'); HostAll is
	// the host copy with every expert unpaged, asked only when the plan tries it. Both 0 off Metal.
	HostFixed int64
	HostAll   func() int64
	KV        KVCurve // the backend's KV allocation at a context
	// FixedSlack is what the backend's allocator adds over the buffers of the fixed terms (CUDA rounds every buffer of 2 MiB or
	// more up to 2 MiB); it is priced wherever the dense weights are.
	FixedSlack int64

	CtxCeiling int // a fixed context cap past which the backend allocates nothing more (0 for none)
	// CtxStep is the grid a shrunk context stays on (Metal allocates KV in 8-position steps; 1 elsewhere). CtxSearch makes the
	// context search a bisection instead of arithmetic, for a backend whose KV is a step function plus a constant (CUDA).
	CtxStep   int
	CtxSearch bool
}

// PlanRequest is what the caller asked for, every default already resolved.
type PlanRequest struct {
	Ctx        int
	CtxPinned  bool  // refuse rather than shrink
	Slots      int   // an explicit expert-cache slot count; 0 lets the plan choose
	ExtraBytes int64 // whatever shares the device and the model does not know of (a drafter, a tower)
}

// DevicePlan is SolvePlacement's answer, with every number behind it. A field a path does not reach stays zero, as the
// decoder's Plan reports it.
type DevicePlan struct {
	Placement       Placement
	Ctx, Slots      int
	ExpertBytesUsed int64
	KVBytes         int64
	HostCopyBytes   int64
	Reason          string
}

func gbf(b int64) float64 { return float64(b) / (1 << 30) }

// slackNote names the allocator's rounding in a reason when there is any.
func slackNote(b int64) string {
	if b <= 0 {
		return ""
	}
	return fmt.Sprintf(" + allocation rounding %.2f GB", gbf(b))
}

func hostCopyNote(b int64) string {
	if b <= 0 {
		return ""
	}
	return fmt.Sprintf(" + host copy %.2f GB", gbf(b))
}

// SolvePlacement places l on m's free memory for l.Tier, in the planner's order: shrink the context toward the policy's plan
// floor first (it costs no numerics), then cap the routed experts into a cache; the dense weights always stay. The CPU never
// declines: what does not fit pages from disk.
func SolvePlacement(l DeviceLoad, req PlanRequest, m Machine, pol Policy) DevicePlan {
	free := m.Available(l.Tier)
	var p DevicePlan
	decline := func(format string, args ...any) DevicePlan {
		p.Reason = fmt.Sprintf(format, args...)
		if l.Backend == "cpu" {
			p.Placement = WeightPaged
		} else {
			p.Placement = Declined
		}
		return p
	}
	// An MoE whose experts could not be priced (under a byte each) is not placed on unknown bytes, resident or cached.
	if l.NExperts > 0 && l.ExpertBytesFull < int64(l.NExperts) {
		return decline("the routed experts could not be priced (%d bytes over %d experts): nothing is placed on unknown bytes",
			l.ExpertBytesFull, l.NExperts)
	}
	tryCtx := func(ctx int) (kv int64, fits bool) {
		kv = l.KV(ctx)
		fits = l.DenseBytes+l.FixedSlack+l.HostFixed+kv+req.ExtraBytes <= free
		if l.CtxCeiling > 0 && ctx > l.CtxCeiling {
			fits = false
		}
		return kv, fits
	}
	floor := pol.PlanCtxFloor
	chooseCtx := func() (ctx int, kv int64, ok bool) {
		if kv, fits := tryCtx(req.Ctx); fits {
			return req.Ctx, kv, true
		}
		if req.CtxPinned {
			kv, _ := tryCtx(req.Ctx)
			return req.Ctx, kv, false
		}
		if l.CtxSearch {
			// KV is whole blocks beside a constant here, so a rate read off one position misprices: search for the largest
			// context that fits (tryCtx is monotone in it), never past what was asked.
			if _, fits := tryCtx(floor); !fits {
				return req.Ctx, l.KV(req.Ctx), false
			}
			lo, hi := floor, req.Ctx // fits(lo), !fits(hi)
			for hi-lo > 1 {
				if mid := lo + (hi-lo)/2; func() bool { _, ok := tryCtx(mid); return ok }() {
					lo = mid
				} else {
					hi = mid
				}
			}
			return lo, l.KV(lo), true
		}
		step := max(l.CtxStep, 1)
		perPos := l.KV(step) / int64(step)
		if perPos <= 0 {
			return req.Ctx, 0, false
		}
		budget := free - l.DenseBytes - l.FixedSlack - l.HostFixed - req.ExtraBytes
		fitCtx := int(budget/perPos) / step * step
		if l.CtxCeiling > 0 && fitCtx > l.CtxCeiling {
			fitCtx = l.CtxCeiling // a fixed cap, not a byte budget: never grow past it either
		}
		if fitCtx < floor {
			return req.Ctx, l.KV(req.Ctx), false
		}
		if fitCtx > req.Ctx {
			fitCtx = req.Ctx // never grow past what was asked
		}
		return fitCtx, l.KV(fitCtx), true
	}

	ctx, kv, ctxOK := chooseCtx()
	p.Ctx, p.KVBytes = ctx, kv
	if !ctxOK {
		if l.CtxCeiling > 0 && req.Ctx > l.CtxCeiling {
			if l.Backend == "metal" {
				return decline("context %d exceeds metal's %d-position ceiling — pass a smaller -ctx", req.Ctx, l.CtxCeiling)
			}
			return decline("context %d exceeds webgpu's fixed %d-position ceiling at this KV precision (M-32) — pass a smaller -ctx or drop --kv-f16/--kv-i8",
				req.Ctx, l.CtxCeiling)
		}
		return decline("context %d does not fit even at the %d-position floor: dense %.2f GB%s + KV %.2f GB exceeds %.2f GB free",
			req.Ctx, floor, gbf(l.DenseBytes), slackNote(l.FixedSlack)+hostCopyNote(l.HostFixed), gbf(kv), gbf(free))
	}
	if l.NExperts == 0 {
		p.Placement = Resident
		p.HostCopyBytes = l.HostFixed
		p.Reason = fmt.Sprintf("dense %.2f GB%s + KV@%d %.2f GB fits %.2f GB free", gbf(l.DenseBytes), slackNote(l.FixedSlack)+hostCopyNote(l.HostFixed), ctx, gbf(kv), gbf(free))
		return p
	}
	// Every expert resident, at the context just chosen; unpaged experts carry a host copy too on Metal.
	var hostAll int64
	if l.HostAll != nil {
		hostAll = l.HostAll()
	}
	if l.DenseBytes+l.FixedSlack+l.ExpertBytesFull+hostAll+kv+req.ExtraBytes <= free {
		p.Placement = Resident
		p.ExpertBytesUsed = l.ExpertBytesFull
		p.HostCopyBytes = hostAll
		p.Reason = fmt.Sprintf("dense %.2f GB + experts %.2f GB%s + KV@%d %.2f GB fits %.2f GB free",
			gbf(l.DenseBytes), gbf(l.ExpertBytesFull), slackNote(l.FixedSlack)+hostCopyNote(hostAll), ctx, gbf(kv), gbf(free))
		return p
	}
	// The largest slot count whose bytes fit beside the fixed terms. A layer's experts are uniform in shape, so this is arithmetic.
	remaining := free - l.DenseBytes - l.FixedSlack - l.HostFixed - kv - req.ExtraBytes
	if remaining <= 0 {
		return decline("dense %.2f GB%s + KV@%d %.2f GB alone exceeds %.2f GB free — no room left for any expert cache",
			gbf(l.DenseBytes), slackNote(l.FixedSlack)+hostCopyNote(l.HostFixed), ctx, gbf(kv), gbf(free))
	}
	perExpert := l.ExpertBytesFull / int64(l.NExperts) // at least 1: unpriced experts declined above
	slots := int(remaining / perExpert)
	if req.Slots > 0 && req.Slots < slots {
		slots = req.Slots // an explicit, smaller request is honoured even if more would fit
	}
	if slots > l.NExperts {
		slots = l.NExperts
	}
	if slots < l.TopK {
		return decline("expert-cached needs at least top-k=%d slots/layer (%.2f GB) but only %.2f GB is free beside dense+KV",
			l.TopK, gbf(int64(l.TopK)*perExpert), gbf(remaining))
	}
	p.Placement = ExpertCached
	p.Slots = slots
	p.ExpertBytesUsed = int64(slots) * perExpert
	p.HostCopyBytes = l.HostFixed
	p.Reason = fmt.Sprintf("dense %.2f GB%s + %d/%d experts %.2f GB + KV@%d %.2f GB fits %.2f GB free",
		gbf(l.DenseBytes), slackNote(l.FixedSlack)+hostCopyNote(l.HostFixed), slots, l.NExperts, gbf(p.ExpertBytesUsed), ctx, gbf(kv), gbf(free))
	return p
}
