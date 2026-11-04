package mem

import "fmt"

// GIWLoad is what a .giw load puts in host memory beside its file-backed weights: KV and prefill scratch (unless the KV is
// on a device) and a pinned copy of the routed experts where a CUDA expert cache builds one.
type GIWLoad struct {
	Ctx           int     // the context priced: the pin, or the model's window
	Pinned        bool    // the caller asked for Ctx explicitly
	KV            KVCurve // the host KV at a context
	DeviceKV      bool    // the KV and the prefill scratch are on a device: only the pinned experts are host memory
	Scratch       int64   // the prefill attention scratch cap
	PinnedExperts int64   // the page-locked expert copy, 0 for none; the same at every context
}

// GIWFit is SolveGIW's answer and the numbers behind it. Known is false when it could not price (proceed).
type GIWFit struct {
	Known               bool
	Avail, Budget, Need int64 // Need is at Ctx
	PinnedExperts       int64
	Ctx                 int // the context priced: the pin, the model's window, or the capped one
	Pinned              bool
	PinCtx              int // the smaller context an unpinned load is pinned to, 0 for none
	Err                 error
	Note                string // the line printed when it caps
}

// SolveGIW is the .giw guard on l against m under p: refuse, or auto-pin an unpinned load to a smaller context, when its host
// need would exceed available memory less the policy's margin.
func SolveGIW(l GIWLoad, m Machine, p Policy) GIWFit {
	avail := m.Available(HostAnon)
	if avail <= 0 {
		return GIWFit{}
	}
	budget := max(avail-p.GIWMargin, 0)
	what := "KV + scratch"
	if l.PinnedExperts > 0 {
		what += " + the pinned MoE expert copy"
	}
	needAt := func(ctx int) int64 {
		if l.DeviceKV {
			return l.PinnedExperts
		}
		return l.KV(ctx) + l.Scratch + l.PinnedExperts
	}
	margin := float64(p.GIWMargin) / gb
	need := needAt(l.Ctx)
	d := GIWFit{Known: true, Avail: avail, Budget: budget, Need: need, Ctx: l.Ctx, Pinned: l.Pinned, PinnedExperts: l.PinnedExperts}
	if need <= budget {
		return d
	}
	if l.Pinned {
		d.Err = fmt.Errorf(
			"decoder: pinned context %d needs ~%.2f GB (%s) but only %.2f GB of this "+
				"machine's %.2f GB currently-available memory would be left as a %.0f GB safety "+
				"margin — pass a smaller -ctx, or set GOINFER_NO_FIT_GUARD=1 to allow it anyway",
			l.Ctx, float64(need)/gb, what, float64(budget)/gb, float64(avail)/gb, margin)
		return d
	}
	floor := p.CtxFloor
	if floorNeed := needAt(floor); floorNeed > budget {
		d.Ctx, d.Need = floor, floorNeed
		d.Err = fmt.Errorf(
			"decoder: this model needs ~%.2f GB (%s) even at the %d-token floor, but only "+
				"%.2f GB of this machine's %.2f GB currently-available memory would be left as a "+
				"%.0f GB safety margin — set GOINFER_NO_FIT_GUARD=1 to allow it anyway",
			float64(floorNeed)/gb, what, floor, float64(budget)/gb, float64(avail)/gb, margin)
		return d
	}
	lo, hi := floor, l.Ctx
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if needAt(mid) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	d.PinCtx, d.Ctx, d.Need = lo, lo, needAt(lo)
	d.Note = fmt.Sprintf(
		"decoder: .giw context capped at %d (model allows %d) so that KV+scratch fits: needs ~%.2f GB "+
			"of %.2f GB available (%.0f GB margin) — pass -ctx to choose, or free memory to lift\n",
		lo, l.Ctx, float64(needAt(lo))/gb, float64(avail)/gb, margin)
	return d
}
