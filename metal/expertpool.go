//go:build darwin

package metal

import (
	"fmt"
	"io"
	"sync"
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

	// preads counts miss-stages served by the pread fast path (0 ⇒ the mmap byte-copy path ran).
	// It lets a test PROVE pread engaged: a byte-identity check passes just as well when the offset
	// resolution silently fell back, which would leave the fast path green but unexercised.
	preads int

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
	return preadRangeIntoU32Buf(fd, dst, 0, off, len(d)*4)
}

// preadRangeIntoU32Buf is preadIntoU32Buf over a SUB-RANGE of the destination: it reads n bytes from
// file offset off into the slot buffer's contents starting at byte offset dstOff. The generic-MoE
// paged path needs this because a layer's gate and up projections are SEPARATE tensors (separate
// .giw spans) that the slot's fused gate|up buffer concatenates — gate at dstOff 0, up at dstOff
// len(gateBytes) — whereas gemma4's bundle hands over one already-fused gate|up span that fills the
// whole buffer. Group-32 int4 makes every span a multiple of 16 bytes, so the up half always starts
// word-aligned and the concatenation matches the stacked layout byte-for-byte.
//
// dstOff/n outside the destination is a programming error in the offset resolution, not a runtime
// condition — it would silently stage a truncated expert, so it returns an error rather than
// clamping. Loops on short reads.
func preadRangeIntoU32Buf(fd int, dst Buffer, dstOff int, off int64, n int) error {
	d := dst.U32s()
	if n == 0 {
		return nil
	}
	if dstOff < 0 || n < 0 || dstOff+n > len(d)*4 {
		return fmt.Errorf("pread range [%d,%d) outside %d-byte slot buffer", dstOff, dstOff+n, len(d)*4)
	}
	db := unsafe.Slice((*byte)(unsafe.Pointer(&d[0])), len(d)*4)[dstOff : dstOff+n]
	for done := 0; done < len(db); {
		r, err := syscall.Pread(fd, db[done:], off+int64(done))
		if err != nil {
			return err
		}
		if r == 0 {
			return io.ErrUnexpectedEOF
		}
		done += r
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
	for s := range N {
		p.slots[s] = expertSlot{
			guW: NewBufferUint32s(d, make([]uint32, nGuW)),
			guS: NewBufferU16s(d, make([]uint16, nGuS)),
			dW:  NewBufferUint32s(d, make([]uint32, nDW)),
			dS:  NewBufferU16s(d, make([]uint16, nDS)),
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
		p.preads++
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

// ensureResidentBatch is ensureResident generalized over a whole layer's routed top-k at once
// (M-12, audit-metal-2026-09-12.md): slot selection and eviction bookkeeping run SEQUENTIALLY
// first — the LRU/where/slotExpert state is not safe for concurrent mutation, and two misses in
// the same batch must never be handed the same slot — but the actual staging I/O for every miss
// then runs CONCURRENTLY. Serial staging at queue depth 1 was the measured bottleneck (per-miss
// cost RISING with N, the signature of latency-bound cold reads with one outstanding request);
// pread on a shared fd into disjoint destination buffers is safe (each syscall is positional —
// it never moves a shared file offset), and the mmap byte-copy path's per-expert source/dest
// spans are equally disjoint, so there is nothing here that needs the calls serialized.
//
// A duplicate id within one call (the router selecting the same expert into two of its top-k
// slots — not expected, but not assumed impossible either) resolves to the SAME already-picked
// slot rather than staging it twice or racing two goroutines over one slot's buffers.
//
// Callers must still serialize ACROSS calls exactly as before (ensureResident's own doc comment:
// "the paged forward must submit+wait at each layer before calling this") — this only parallelizes
// the I/O WITHIN one call, on the single host goroutine that already owns this pool exclusively.
func (p *expertPool) ensureResidentBatch(ids []int) []expertSlot {
	out := make([]expertSlot, len(ids))
	type miss struct{ idx, expert, slot int }
	var misses []miss
	seen := make(map[int]int, len(ids))     // expert id -> slot, for a duplicate id within this call
	claimed := make(map[int]bool, len(ids)) // slot -> claimed by a miss earlier in THIS batch
	for i, e := range ids {
		if s, ok := seen[e]; ok {
			// A repeat of an id already resolved earlier IN THIS BATCH (the router selecting the
			// same expert into two top-k slots) — counts as a hit even if the FIRST occurrence was
			// itself a miss: sequentially, ensureResident(e) called twice in a row would see the
			// second call find e already resident. Never double-stage: the slot is correct either
			// way (already staged, or its stage is in flight — the WaitGroup below still waits on
			// it since it's tracked in misses by the first occurrence's index).
			p.hits++
			out[i] = p.slots[s]
			continue
		}
		if s, ok := p.where[e]; ok {
			p.hits++
			p.touch(s)
			out[i] = p.slots[s]
			seen[e] = s
			continue
		}
		s := p.pickSlot()
		if claimed[s] {
			// Out of contract: more DISTINCT misses in one batch than this pool has slots for, so
			// pickSlot's eviction fallback landed on a slot a miss EARLIER in this same batch
			// already claimed — staging both would race two goroutines over one buffer and leave
			// p.where pointing two experts at the same slot. newExpertPool's own doc comment
			// requires N >= the router's top-k specifically to make this unreachable in production
			// (ids is always one layer's routed top-k, length <= N by that build-time guard); a
			// caller that violates it gets a loud panic, not silent corruption.
			panic(fmt.Sprintf("metal expertpool: batch of %d distinct experts exceeds this pool's "+
				"%d slots (id %d re-picked slot %d) — the caller must ensure N >= top-k", len(ids), len(p.slots), e, s))
		}
		claimed[s] = true
		if old := p.slotExpert[s]; old >= 0 {
			delete(p.where, old)
			p.evictions++
		} else {
			p.coldStarts++
		}
		// Claim AND touch the slot now, not after staging: pickSlot's eviction fallback reads
		// p.lru's tail, and deferring touch() until after this whole loop would leave p.lru
		// reflecting the PRE-BATCH ordering while slotExpert already shows every slot claimed so
		// far as occupied — exactly the state that makes a later miss in this same batch either
		// evict a slot another miss in this batch already claimed, or (once every slot has been
		// claimed once) index p.lru out of range. touch() only reorders p.lru by slot INDEX; it
		// does not depend on staged content, so moving it before the I/O below changes nothing
		// observable about eviction order, only that the pick loop stays self-consistent.
		p.slotExpert[s] = e
		p.touch(s)
		seen[e] = s
		misses = append(misses, miss{i, e, s})
	}

	if len(misses) > 0 {
		var wg sync.WaitGroup
		var mu sync.Mutex
		var fetchNanos int64
		wg.Add(len(misses))
		for _, ms := range misses {
			go func(ms miss) {
				defer wg.Done()
				t0 := time.Now()
				if p.stagePread != nil {
					p.stagePread(ms.expert, p.slots[ms.slot])
				} else {
					guW, guS, dW, dS := p.stage(ms.expert)
					copyBytesToU32Buf(p.slots[ms.slot].guW, guW)
					copy(p.slots[ms.slot].guS.U16s(), guS)
					copyBytesToU32Buf(p.slots[ms.slot].dW, dW)
					copy(p.slots[ms.slot].dS.U16s(), dS)
				}
				d := time.Since(t0).Nanoseconds()
				mu.Lock()
				fetchNanos += d
				mu.Unlock()
			}(ms)
		}
		wg.Wait()
		p.fetchNanos += fetchNanos
		// stageNanos is the wall-clock paging-traffic penalty term (ensureResident's own comment);
		// under concurrent staging that wall-clock is the WaitGroup's own span, not a fetchNanos
		// sum, but summing fetchNanos here is a defensible upper bound and keeps this counter's
		// unit consistent with the sequential path without adding a second wall-clock timer.
		p.stageNanos += fetchNanos
		if p.stagePread != nil {
			p.preads += len(misses)
		}
		p.stages += len(misses)
		for _, ms := range misses {
			// where[] is populated only after staging completes — a concurrent reader is not
			// expected here (this pool has exactly one caller, per ensureResident's own doc
			// comment), but this keeps "resident" meaning "actually holds the right bytes" rather
			// than "has been claimed", matching the sequential path's own ordering. touch() already
			// ran in the claiming loop above.
			p.where[ms.expert] = ms.slot
			out[ms.idx] = p.slots[ms.slot]
		}
	}
	return out
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
