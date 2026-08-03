//go:build darwin

package metal

import (
	"io"
	"syscall"
	"time"
	"unsafe"
)

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

// stageFn fetches expert e's W4A8 data: gate|up packed nibble BYTES + f16 scales, down nibble bytes
// + f16 scales. In production it is int4DirectBytes over the layer bundle's expert WeightMats — the
// nibble bytes ALIAS the mmap (zero-copy, no reconstruction, no per-stage allocation); in isolation
// tests it returns synthetic per-expert bytes. The word buffers are uint32 on the GPU but staged as
// raw LE bytes (copyBytesToU32Buf) — the mmap span is byte-for-byte the uint32 words on LE, and a
// byte copy needs no source alignment (the mmap offsets are not 4-aligned). Returned slices are
// copied into the slot immediately, so the callee may reuse/alias its backing.
type stageFn func(e int) (guW []byte, guS []uint16, dW []byte, dS []uint16)

// copyBytesToU32Buf memcpys little-endian nibble bytes into a uint32 slot buffer's shared contents.
// The buffer is UMA/page-aligned, so reinterpreting its []uint32 view as []byte is always safe; the
// source is an unaligned mmap span, which a byte copy handles (a *uint32 alias of it would be
// misaligned UB — measured 73% of expert spans are not 4-aligned). On LE this yields the exact words
// bytesToU32 would build, so paged ≡ non-paged byte-identity is preserved.
func copyBytesToU32Buf(dst Buffer, src []byte) {
	d := dst.U32s()
	if len(d) == 0 {
		return
	}
	db := unsafe.Slice((*byte)(unsafe.Pointer(&d[0])), len(d)*4)
	copy(db, src)
}

type expertPool struct {
	slots      []expertSlot
	slotExpert []int       // slot index → expert id resident there (-1 = free)
	where      map[int]int // expert id → slot index
	lru        []int       // slot indices, most-recently-used at front
	stage      stageFn

	stages     int   // miss-stages performed (telemetry + isolation-test oracle)
	hits       int   // ensureResident calls that found the expert already resident (same-expert-reuse)
	coldStarts int   // stages into a previously-free slot (pool not yet full)
	evictions  int   // stages that evicted an occupied slot (pool under pressure)
	stageNanos int64 // total host time spent staging (fetch + copy) — the paging-traffic penalty term
	fetchNanos int64 // time in stage() — mmap-aliased nibble bytes + f32→f16 scales (no reconstruction)
	copyNanos  int64 // time byte-copying the fetched nibbles/scales into the slot's shared Metal buffers

	// prefetch, when set, issues an MADV_WILLNEED readahead over expert e's mmap-backed nibble spans.
	// The synchronous paged forward calls prefetchAll for the whole routed top-k BEFORE touching any
	// of them, converting each expert's ~200 serial 16 KB demand faults into one large sequential read
	// (and giving the SSD queue depth across the k experts instead of k serial stalls). nil = off.
	// MEASURED AND DECLINED (see gemma4_moe.go) — kept only so the env flag stays wired.
	prefetch func(e int)

	// stagePread, when set (GOINFER_MOE_PREAD=1 on a .giw-mmap'd model), REPLACES the mmap byte-copy:
	// it preads expert e's nibbles straight into slot s's unified-memory buffers — one syscall, one
	// large sequential read, zero page faults (cold pread measured 3687 MB/s vs the mmap demand-fault's
	// 375 MB/s, 9.8×). Fetch and copy collapse into the single read. nil ⇒ the mmap byte-copy path.
	stagePread func(e int, s expertSlot)
}

// preadIntoU32Buf preads the destination's worth of nibbles from file offset off DIRECTLY into the
// slot buffer's unified-memory contents (host-writable UMA — the read lands where the GPU reads it,
// no intermediate copy). The []byte view of the []uint32 destination is always page-aligned; buffered
// pread has no source-offset alignment requirement, so the 73%-unaligned expert spans are a non-issue
// here. Loops on short reads.
func preadIntoU32Buf(fd int, dst Buffer, off int64) error {
	d := dst.U32s()
	if len(d) == 0 {
		return nil
	}
	db := unsafe.Slice((*byte)(unsafe.Pointer(&d[0])), len(d)*4)
	for done := 0; done < len(db); {
		n, err := syscall.Pread(fd, db[done:], off+int64(done))
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrUnexpectedEOF
		}
		done += n
	}
	return nil
}

// prefetchAll issues the WILLNEED readahead for every routed expert at once, before staging. No-op
// when prefetch is nil (readahead disabled — the demand-fault baseline arm).
func (p *expertPool) prefetchAll(experts []int) {
	if p.prefetch == nil {
		return
	}
	for _, e := range experts {
		p.prefetch(e)
	}
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
	t0 := time.Now()
	if p.stagePread != nil {
		// pread path: one syscall reads nibbles straight into the slot's UMA words (fetch+copy fused).
		p.stagePread(e, p.slots[s])
		p.fetchNanos += time.Since(t0).Nanoseconds()
	} else {
		guW, guS, dW, dS := p.stage(e) // mmap-aliased nibble bytes + f16 scales (no reconstruction/alloc)
		t1 := time.Now()
		copyBytesToU32Buf(p.slots[s].guW, guW)
		copy(p.slots[s].guS.U16s(), guS)
		copyBytesToU32Buf(p.slots[s].dW, dW)
		copy(p.slots[s].dS.U16s(), dS)
		t2 := time.Now()
		p.fetchNanos += t1.Sub(t0).Nanoseconds()
		p.copyNanos += t2.Sub(t1).Nanoseconds()
	}
	p.stageNanos += time.Since(t0).Nanoseconds() // total paging-traffic cost (penalty decomposition)
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
