//go:build darwin

package metal

// expertPool is a bounded per-layer LRU pool of N expert slots for SYNCHRONOUS Metal MoE paging.
// The gemma4-26b full expert set is 11.96 GB (per-expert W4A8 ≈ 3.19 MB × 128 × 30 layers) — it does
// not fit resident, so each MoE layer keeps only N experts on the GPU and stages the routed top-k in
// from host (mmap-backed) bytes on demand, evicting the least-recently-used slot. One pool per MoE
// layer (experts differ per layer). This is the CPU expertPager (decoder/moepaging.go) analogue for
// the Metal backend — the paging half of the synchronous path; the per-layer submit+wait boundary
// that reads the router idx before staging lives in the encode path.
//
// Staging is a host→shared-buffer copy (UMA). Eviction is BOOKKEEPING ONLY: it frees a slot for
// reuse, it does NOT return pages to the OS (Darwin mmap.Advise(_,false) is a documented no-op) — so
// resident footprint must be MEASURED, not inferred from N × per-expert. See
// [[metal-moe-paging-needs-speculation]] and the Step-6 budget probe (paging_budget_test.go).
type expertSlot struct{ guW, guS, dW, dS Buffer }

// stageFn fetches expert e's W4A8 bytes (gate|up words+f16 scales, down words+f16 scales). In
// production it is int4DirectWords over the layer bundle's expert WeightMats (mmap-backed, read on
// demand); in isolation tests it returns synthetic per-expert data. Returned slices are copied into
// the slot immediately, so the callee may reuse/alias its backing.
type stageFn func(e int) (guW []uint32, guS []uint16, dW []uint32, dS []uint16)

type expertPool struct {
	slots      []expertSlot
	slotExpert []int       // slot index → expert id resident there (-1 = free)
	where      map[int]int // expert id → slot index
	lru        []int       // slot indices, most-recently-used at front
	stage      stageFn

	stages     int // miss-stages performed (telemetry + isolation-test oracle)
	hits       int // ensureResident calls that found the expert already resident (same-expert-reuse)
	coldStarts int // stages into a previously-free slot (pool not yet full)
	evictions  int // stages that evicted an occupied slot (pool under pressure)
}

// newExpertPool allocates N slots sized to one expert's W4A8 buffers (nGuW/nGuS gate|up words/scales,
// nDW/nDS down words/scales). All slots start free; the first N distinct experts fill them, then LRU
// eviction begins. N must be >= the router's top-k or a single token can thrash its own pool.
func newExpertPool(d *Device, N, nGuW, nGuS, nDW, nDS int, stage stageFn) *expertPool {
	p := &expertPool{
		slots:      make([]expertSlot, N),
		slotExpert: make([]int, N),
		where:      make(map[int]int, N),
		lru:        make([]int, 0, N),
		stage:      stage,
	}
	for s := 0; s < N; s++ {
		p.slots[s] = expertSlot{
			guW: d.NewBufferUint32s(make([]uint32, nGuW)),
			guS: d.NewBufferU16s(make([]uint16, nGuS)),
			dW:  d.NewBufferUint32s(make([]uint32, nDW)),
			dS:  d.NewBufferU16s(make([]uint16, nDS)),
		}
		p.slotExpert[s] = -1
	}
	return p
}

// ensureResident returns the slot holding expert e, staging it in on a miss (evicting the LRU slot)
// and marking it most-recently-used. Value-DEPENDENT (e comes from the router readback), which is why
// the paged forward must submit+wait at each layer before calling this — it cannot be pre-encoded.
func (p *expertPool) ensureResident(e int) expertSlot {
	if s, ok := p.where[e]; ok {
		p.hits++
		p.touch(s)
		return p.slots[s]
	}
	s := p.pickSlot()
	if old := p.slotExpert[s]; old >= 0 {
		delete(p.where, old)
		p.evictions++
	} else {
		p.coldStarts++
	}
	guW, guS, dW, dS := p.stage(e)
	copy(p.slots[s].guW.U32s(), guW)
	copy(p.slots[s].guS.U16s(), guS)
	copy(p.slots[s].dW.U32s(), dW)
	copy(p.slots[s].dS.U16s(), dS)
	p.slotExpert[s] = e
	p.where[e] = s
	p.touch(s)
	p.stages++
	return p.slots[s]
}

// pickSlot returns a free slot if one exists, else the least-recently-used slot (lru tail).
func (p *expertPool) pickSlot() int {
	for s, e := range p.slotExpert {
		if e < 0 {
			return s
		}
	}
	return p.lru[len(p.lru)-1]
}

// touch moves slot s to the MRU front of the LRU list (fresh slice — an in-place p.lru[:0] filter
// would overwrite lru[0] before the loop reads it).
func (p *expertPool) touch(s int) {
	out := make([]int, 0, len(p.lru)+1)
	out = append(out, s)
	for _, x := range p.lru {
		if x != s {
			out = append(out, x)
		}
	}
	p.lru = out
}
