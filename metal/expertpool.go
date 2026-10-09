//go:build darwin

package metal

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
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
//
// M-11 (audit-metal-2026-09-12.md): slots live in ONE contiguous Buffer per field (guW/guS/dW/dS),
// not N separately-allocated Buffer objects. A separate object per slot forced the paged phase-2
// encoder to pick a specific Buffer identity at ENCODE time — after staging, which is exactly why
// the paged forward could not be pre-encoded into one command buffer with the rest of the token
// (the blocker M-11 names). A contiguous pool lets phase 2 bind the SAME buffer identity every
// call and read a slot NUMBER at kernel-execution time instead — the same trick the non-paged
// stacked-all-E path already uses via rIdx (see moe.go's gemv_w4a8_moe: "idx[slot]*rowsPerExpert").
// expertSlot is now a lightweight VIEW into that contiguous storage (Buffer.At()-offset), valid for
// GPU-side binding and for gpu.Upload (both respect Buffer's bind offset) but NOT for U32s()/U16s()
// (those always read from the buffer's base, ignoring the offset — see preadIntoPoolSlot's doc
// comment for why the pread path does its own offset arithmetic instead of relying on a view).
type expertSlot struct {
	guW, guS, dW, dS Buffer // .At()-offset views onto the pool's contiguous per-field buffers
	slot             int    // the raw slot index — the GPU-side "which row of the pool" value
}

// stageFn fetches expert e's W4A8 data: gate|up packed nibble BYTES + f16 scales, down nibble bytes
// + f16 scales. In production it is int4DirectBytes over the layer bundle's expert WeightMats — the
// nibble bytes ALIAS the mmap (zero-copy, no reconstruction, no per-stage allocation); in isolation
// tests it returns synthetic per-expert bytes. The word buffers are uint32 on the GPU but staged as
// raw LE bytes (copyBytesToU32Buf) — the mmap span is byte-for-byte the uint32 words on LE, and a
// byte copy needs no source alignment (the mmap offsets are not 4-aligned). Returned slices are
// copied into the slot immediately, so the callee may reuse/alias its backing.
type stageFn func(e int) (guW []byte, guS []uint16, dW []byte, dS []uint16)

// copyBytesToU32Buf memcpys little-endian nibble bytes into a uint32 slot buffer's shared contents,
// via gpu.Upload rather than a hand-rolled unsafe.Slice reinterpret (N-33, audit-metal-2026-09-12.md):
// the source is an unaligned mmap span, which gpu.Upload's byte-slice copy handles just as the old
// reinterpret did (a *uint32 alias of it would be misaligned UB — measured 73% of expert spans are
// not 4-aligned), but gpu.Upload also bounds-checks src against the buffer's actual allocated bytes
// instead of Go's `copy` silently truncating an oversized src with nothing to say so. On LE this
// yields the exact words bytesToU32 would build, so paged ≡ non-paged byte-identity is preserved.
// Panics on error: every call site's dst is sized for exactly this src by construction (newExpertPool
// / the stage() contract), so a failure here is an invariant violation, not a runtime condition to
// recover from.
func copyBytesToU32Buf(dst Buffer, src []byte) {
	if err := gpu.Upload(dst, src); err != nil {
		panic(fmt.Sprintf("metal expertpool: %v", err))
	}
}

// copyU16sToBuf writes f16-bits scales into a (possibly slot-offset) uint16 buffer view via
// gpu.Upload rather than dst.U16s()+copy(): U16s() always reads from the buffer's BASE address and
// ignores Buffer.At()'s bind offset (see expertSlot's doc comment), so a plain copy() into it would
// silently land in slot 0 regardless of which slot's view was passed. gpu.Upload respects the offset.
func copyU16sToBuf(dst Buffer, src []uint16) {
	if err := gpu.Upload(dst, src); err != nil {
		panic(fmt.Sprintf("metal expertpool: %v", err))
	}
}

type expertPool struct {
	// guW/guS/dW/dS are ONE contiguous Buffer per field, N slots wide (M-11) — fixed identity for
	// the pool's whole lifetime, never reallocated. nGuW/nGuS/nDW/nDS are the per-slot element
	// counts (uint32/uint16 words), i.e. the stride: slot s's region starts at element s*nGuW (etc),
	// byte offset s*nGuW*4 (etc, uint16 fields use *2).
	guW, guS, dW, dS     Buffer
	nGuW, nGuS, nDW, nDS int
	slotExpert           []int        // slot index → expert id resident there (-1 = free)
	evictedFrom          map[int]bool // planted defect 1 only (expertPoolDefect): slots whose expert was evicted
	where                map[int]int  // expert id → slot index
	lru                  []int        // slot indices, most-recently-used at front
	stage                stageFn

	// preads counts miss-stages served by the pread fast path (0 ⇒ the mmap byte-copy path ran).
	// It lets a test PROVE pread engaged: a byte-identity check passes just as well when the offset
	// resolution silently fell back, which would leave the fast path green but unexercised.
	preads int

	stages     int // miss-stages performed (telemetry + isolation-test oracle)
	hits       int // ensureResident calls that found the expert already resident (same-expert-reuse)
	coldStarts int // stages into a previously-free slot (pool not yet full)
	evictions  int // stages that evicted an occupied slot (pool under pressure)
	// distinctExperts is every expert id this pool has EVER staged, across its whole lifetime —
	// unlike coldStarts (caps at N once the pool fills) or where (only currently-resident), this
	// answers "how many distinct experts did this layer's routing actually touch" — the floor an
	// expert-major (route-then-group-then-stage-once) prefill could hit, vs. stages, which also
	// counts every re-fetch of a previously-evicted-then-needed-again expert (M-05 investigation,
	// audit-metal-2026-09-12.md). Negligible memory (≤ nE entries); left in production code as
	// always-on telemetry, same as the counters above.
	distinctExperts map[int]bool
	stageNanos      int64 // total host time spent staging (fetch + copy) — the paging-traffic penalty term
	fetchNanos      int64 // time in stage() — mmap-aliased nibble bytes + f32→f16 scales (no reconstruction)
	copyNanos       int64 // time byte-copying the fetched nibbles/scales into the slot's shared Metal buffers

	// stagePread, when set (GOINFER_MOE_PREAD=1 on a .giw-mmap'd model), REPLACES the mmap byte-copy:
	// it preads expert e's nibbles straight into slot s's unified-memory buffers — one syscall, one
	// large sequential read, zero page faults (cold pread measured 3687 MB/s vs the mmap demand-fault's
	// 375 MB/s, 9.8×). Fetch and copy collapse into the single read. nil ⇒ the mmap byte-copy path.
	stagePread  func(e int, s expertSlot)
	scalePreads atomic.Int64 // stages whose f16 scales were pread from the file too (C-P01); atomic: batch stages run concurrently

	// Lever 3 (docs/tasks/task-m26-mac-2026-10.md): experts guessed for this layer's next use, claimed on the caller's
	// thread and read in the background (prefetchAsync); finishPrefetch, which ensureResidentBatch runs first, waits for
	// the reads and only then marks them resident, so the GPU never reads a slot still being written.
	prefetchWG      sync.WaitGroup
	prefetchPending []prefetchSlot
	prefetchLive    map[int]bool // prefetched experts not yet asked for: a later hit on one counts in prefetchHits
	prefetched      int          // experts prefetched
	prefetchHits    int          // ensureResidentBatch hits on an expert that was prefetched for it
}

type prefetchSlot struct{ expert, slot int }

// preadIntoPoolSlot/preadRangeIntoPoolSlot pread a SLOT within a pool's contiguous per-field buffer
// (M-11), addressed by slot NUMBER rather than by a distinct Buffer object. They do NOT use
// Buffer.At() for the destination: Buffer.U32s() (which preadRangeIntoU32Buf calls internally to
// bounds-check and byte-view the destination) always reads from the buffer's base address and
// ignores its bind offset (Buffer.At() only affects GPU-side
// binding and gpu.Upload/Download, not U32s()/U16s()) — an At()-shifted view here would silently
// pread into the POOL'S FIRST slot every time, not the one requested. Computing the byte offset
// against the pool's base buffer directly (slot*strideWords*4 [+ subOff]) sidesteps that mismatch.
func preadIntoPoolSlot(fd int, base Buffer, slot, strideWords int, off int64) error {
	return preadRangeIntoU32Buf(fd, base, slot*strideWords*4, off, strideWords*4)
}

func preadRangeIntoPoolSlot(fd int, base Buffer, slot, strideWords, subOff int, off int64, n int) error {
	return preadRangeIntoU32Buf(fd, base, slot*strideWords*4+subOff, off, n)
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
	return preadFull(fd, db, off)
}

// preadScalesIntoPoolSlot is preadRangeIntoPoolSlot for a pool's f16 scale buffer (guS or dS, uint16 elements): it
// reads n bytes from file offset off into slot's region at byte subOff, addressed against the BASE buffer by slot
// number for the same reason (U16s ignores a Buffer.At view's offset). C-P01 (audit-metal-2026-09-30.md): a v14 metal
// or v15 .giw stores each tensor's f16 scales in the file, so the pread stage reads them as it reads the nibbles,
// instead of copying them out of the mapping and taking a page fault per 16 KB on a cold expert.
func preadScalesIntoPoolSlot(fd int, base Buffer, slot, strideElems, subOff int, off int64, n int) error {
	d := base.U16s()
	if n == 0 {
		return nil
	}
	dstOff := slot*strideElems*2 + subOff
	if dstOff < 0 || n < 0 || dstOff+n > len(d)*2 {
		return fmt.Errorf("pread range [%d,%d) outside %d-byte scale buffer", dstOff, dstOff+n, len(d)*2)
	}
	db := unsafe.Slice((*byte)(unsafe.Pointer(&d[0])), len(d)*2)[dstOff : dstOff+n]
	return preadFull(fd, db, off)
}

// preadFull preads len(db) bytes from file offset off, looping on short reads.
func preadFull(fd int, db []byte, off int64) error {
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

// scaleFileOffset returns where an f16 scale slice lies in m's .giw mapping, if it does (C-P01). It views the slice's
// bytes without copying; MmapByteOffset is pointer arithmetic and touches no page.
func scaleFileOffset(m *decoder.Model, s []uint16) (int64, bool) {
	if len(s) == 0 {
		return 0, false
	}
	return m.MmapByteOffset(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), 2*len(s)))
}

// newExpertPool allocates N slots sized to one expert's W4A8 buffers (nGuW/nGuS gate|up words/scales,
// nDW/nDS down words/scales) as ONE contiguous buffer per field (M-11) — never N separate objects.
// All slots start free; the first N distinct experts fill them, then LRU eviction begins. N must be
// >= the router's top-k or a single token can thrash its own pool.
func newExpertPool(d *Device, N, nGuW, nGuS, nDW, nDS int, stage stageFn) *expertPool {
	p := &expertPool{
		// N-21 (audit-metal-2026-09-12.md): every slot's contents are about to be overwritten by its
		// first stage anyway (cold-started slots stage on first use, never read before that), so
		// there is nothing to gain from zero-filling a transient Go slice just to copy it into the
		// buffer and discard it — gpu.NewBufferLenOf allocates the right-sized, uninitialized device
		// buffer directly.
		guW:  gpu.NewBufferLenOf[uint32](d, N*nGuW),
		guS:  gpu.NewBufferLenOf[uint16](d, N*nGuS),
		dW:   gpu.NewBufferLenOf[uint32](d, N*nDW),
		dS:   gpu.NewBufferLenOf[uint16](d, N*nDS),
		nGuW: nGuW, nGuS: nGuS, nDW: nDW, nDS: nDS,
		slotExpert: make([]int, N),
		where:      make(map[int]int, N),
		lru:        make([]int, 0, N),
		stage:      stage,
	}
	for s := range N {
		p.slotExpert[s] = -1
	}
	return p
}

// slotView returns the .At()-offset views for slot s — the SAME Buffer identity every call (only
// the offset differs), valid for encode-time GPU binding and for gpu.Upload staging (both respect
// Buffer's bind offset; see the expertSlot doc comment for why this is NOT true of U32s()/U16s()).
func (p *expertPool) slotView(s int) expertSlot {
	return expertSlot{
		guW:  p.guW.At(s * p.nGuW * 4),
		guS:  p.guS.At(s * p.nGuS * 2),
		dW:   p.dW.At(s * p.nDW * 4),
		dS:   p.dS.At(s * p.nDS * 2),
		slot: s,
	}
}

// ensureResident returns the slot holding expert e, staging it in on a miss (evicting the LRU slot)
// and marking it most-recently-used. Value-DEPENDENT (e comes from the router readback), which is why
// the paged forward must submit+wait at each layer before calling this — it cannot be pre-encoded.
func (p *expertPool) ensureResident(e int) expertSlot {
	if s, ok := p.where[e]; ok {
		p.hits++
		p.touch(s)
		return p.slotView(s)
	}
	s := p.pickSlot()
	if old := p.slotExpert[s]; old >= 0 {
		delete(p.where, old)
		p.evictions++
	} else {
		p.coldStarts++
	}
	if p.distinctExperts == nil {
		p.distinctExperts = make(map[int]bool)
	}
	p.distinctExperts[e] = true
	sv := p.slotView(s)
	t0 := time.Now()
	if p.stagePread != nil {
		// pread path: one syscall reads nibbles straight into the slot's UMA words (fetch+copy fused).
		p.stagePread(e, sv)
		p.fetchNanos += time.Since(t0).Nanoseconds()
		p.preads++
	} else {
		guW, guS, dW, dS := p.stage(e) // mmap-aliased nibble bytes + f16 scales (no reconstruction/alloc)
		t1 := time.Now()
		copyBytesToU32Buf(sv.guW, guW)
		copyU16sToBuf(sv.guS, guS)
		copyBytesToU32Buf(sv.dW, dW)
		copyU16sToBuf(sv.dS, dS)
		t2 := time.Now()
		p.fetchNanos += t1.Sub(t0).Nanoseconds()
		p.copyNanos += t2.Sub(t1).Nanoseconds()
	}
	p.stageNanos += time.Since(t0).Nanoseconds() // total paging-traffic cost (penalty decomposition)
	p.slotExpert[s] = e
	p.where[e] = s
	p.touch(s)
	p.stages++
	return sv
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
	p.finishPrefetch()
	for _, e := range ids {
		if p.prefetchLive[e] {
			p.prefetchHits++
			delete(p.prefetchLive, e)
		}
	}
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
			out[i] = p.slotView(s)
			continue
		}
		if s, ok := p.where[e]; ok {
			p.hits++
			p.touch(s)
			out[i] = p.slotView(s)
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
				"%d slots (id %d re-picked slot %d) — the caller must ensure N >= top-k", len(ids), len(p.slotExpert), e, s))
		}
		claimed[s] = true
		if old := p.slotExpert[s]; old >= 0 {
			delete(p.where, old)
			p.evictions++
			if expertPoolDefect == 1 {
				p.markEvicted(s)
			}
		} else {
			p.coldStarts++
		}
		if p.distinctExperts == nil {
			p.distinctExperts = make(map[int]bool)
		}
		p.distinctExperts[e] = true
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
				p.stageInto(ms.expert, ms.slot)
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
			out[ms.idx] = p.slotView(ms.slot)
		}
	}
	return out
}

// expertPoolDefect plants G-1's defects (docs/tasks/task-option-path-admission-2026-10.md §4.3, finding 2); 0 in
// production. 1: a slot whose expert is evicted keeps the evicted expert's bytes (the new one is not staged). 2: the
// GPU is told the next slot along. 3: each staged expert takes the next expert's scales.
var expertPoolDefect int

func (p *expertPool) markEvicted(s int) {
	if p.evictedFrom == nil {
		p.evictedFrom = map[int]bool{}
	}
	p.evictedFrom[s] = true
}

// expertPoolSlotForTest is the pool row a paged forward writes into slotIdx: s, or under planted defect 2 the next row.
func expertPoolSlotForTest(s, n int) int {
	if expertPoolDefect == 2 {
		return (s + 1) % n
	}
	return s
}

// stageInto reads expert e's weights into slot s's buffers (pread, or the mmap byte-copy).
func (p *expertPool) stageInto(e, s int) {
	sv := p.slotView(s)
	if expertPoolDefect == 1 && p.evictedFrom[s] {
		return // planted: the evicted expert's bytes stay
	}
	if p.stagePread != nil && expertPoolDefect != 3 {
		p.stagePread(e, sv)
		return
	}
	guW, guS, dW, dS := p.stage(e)
	if expertPoolDefect == 3 {
		_, guS, _, dS = p.stage(e ^ 1) // a neighbour's scales (in range for an even expert count)
	}
	copyBytesToU32Buf(sv.guW, guW)
	copyU16sToBuf(sv.guS, guS)
	copyBytesToU32Buf(sv.dW, dW)
	copyU16sToBuf(sv.dS, dS)
}

// prefetchAsync starts reading the experts of ids this pool does not hold into slots, in the background, on a guess
// that the next ensureResidentBatch asks for them (lever 3, the decode prefetch). It claims the slots now, on the
// caller's thread (the pool's one owner), evicting least-recently-used ones as a miss would, and never one of ids
// itself; at most len(slots) - 1 claims, so a guess can never take the whole pool. The caller must not let the GPU read
// this pool until finishPrefetch (ensureResidentBatch calls it first).
func (p *expertPool) prefetchAsync(ids []int) {
	p.finishPrefetch()
	claimed := map[int]bool{}
	want := map[int]bool{}
	for _, e := range ids {
		want[e] = true
	}
	for _, e := range ids {
		if _, ok := p.where[e]; ok || claimed[e] || len(p.prefetchPending) >= len(p.slotExpert)-1 {
			continue
		}
		s := p.pickSlot()
		if old := p.slotExpert[s]; old >= 0 {
			if want[old] {
				continue // the LRU slot holds another guessed expert: leave it
			}
			delete(p.where, old)
			delete(p.prefetchLive, old)
			p.evictions++
		} else {
			p.coldStarts++
		}
		p.slotExpert[s] = e
		p.touch(s)
		claimed[e] = true
		p.prefetchPending = append(p.prefetchPending, prefetchSlot{e, s})
	}
	for _, ps := range p.prefetchPending {
		p.prefetchWG.Add(1)
		go func(ps prefetchSlot) {
			defer p.prefetchWG.Done()
			p.stageInto(ps.expert, ps.slot)
		}(ps)
	}
}

// finishPrefetch waits for the reads prefetchAsync started and marks those experts resident.
func (p *expertPool) finishPrefetch() {
	if len(p.prefetchPending) == 0 {
		return
	}
	t0 := time.Now()
	p.prefetchWG.Wait()
	p.stageNanos += time.Since(t0).Nanoseconds() // the part of the reads the caller still waited for
	if p.prefetchLive == nil {
		p.prefetchLive = map[int]bool{}
	}
	if p.distinctExperts == nil {
		p.distinctExperts = map[int]bool{}
	}
	for _, ps := range p.prefetchPending {
		p.where[ps.expert] = ps.slot
		p.prefetchLive[ps.expert] = true
		p.distinctExperts[ps.expert] = true
	}
	p.prefetched += len(p.prefetchPending)
	if p.stagePread != nil {
		p.preads += len(p.prefetchPending)
	}
	p.prefetchPending = p.prefetchPending[:0]
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
