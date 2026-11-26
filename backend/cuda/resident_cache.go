//go:build cuda

package cuda

import (
	"cmp"
	"fmt"
	"slices"
	"time"
	"unsafe"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// expertCache is C′ step 2's per-layer LRU slot residency: which cached expert occupies which of the nSlots device
// slots, so a routed expert already resident skips its H2D DMA. nSlots = topK is the step-1 staging degenerate
// case (every token evicts, no reuse); nSlots > topK gives cross-token reuse. LRU because the router signal is a
// stationary skew where recency is a sufficient statistic for frequency.
// expGU and expDown share the slot index (loaded together), so the cache is per-LAYER, not per-projection.
type expertCache struct {
	nSlots       int
	slotOf       []int32  // [nE]     expert → slot, -1 = not resident
	inSlot       []int32  // [nSlots] slot → expert, -1 = empty
	used         []uint64 // [nSlots] last-touch clock (LRU)
	clock        uint64
	hits, misses uint64 // reuse accounting (a miss is one expert's H2D DMA)

	// DEMAND accounting answers a different question from hits/misses: those say how much DMA reuse SAVED, these say
	// how much the caller ASKED FOR per staging event. stages counts loadRoutedExperts calls on this layer; distinct
	// sums the UNIQUE experts each stage requested. distinct/stages separates "speculation blows the slot budget"
	// from "speculation is just slow here": a verify that presented its whole width to the pager at once would push it
	// ABOVE topK and grow with the width, while one that stages position by position pins it AT topK.
	stages   uint64
	distinct uint64
}

func newExpertCache(nE, nSlots int) *expertCache {
	c := &expertCache{nSlots: nSlots, slotOf: make([]int32, nE), inSlot: make([]int32, nSlots), used: make([]uint64, nSlots)}
	for i := range c.slotOf {
		c.slotOf[i] = -1
	}
	for i := range c.inSlot {
		c.inSlot[i] = -1
	}
	return c
}

// admit returns the slot holding expert e, evicting the LRU slot on a miss. hit=false means the
// caller must DMA e's weights into `slot` before the GEMV reads it. An expert admitted earlier in
// the SAME token is the newest, so it is never the victim (nSlots ≥ topK guarantees room).
func (c *expertCache) admit(e uint32) (slot int, hit bool) {
	c.clock++
	if s := c.slotOf[e]; s >= 0 {
		c.used[s] = c.clock
		c.hits++
		return int(s), true
	}
	c.misses++
	victim, oldest := 0, ^uint64(0)
	for s := 0; s < c.nSlots; s++ {
		if c.inSlot[s] < 0 { // an empty slot always wins
			victim = s
			break
		}
		if c.used[s] < oldest {
			oldest = c.used[s]
			victim = s
		}
	}
	if old := c.inSlot[victim]; old >= 0 {
		c.slotOf[old] = -1
	}
	c.inSlot[victim] = int32(e)
	c.slotOf[e] = int32(victim)
	c.used[victim] = c.clock
	return victim, false
}

// unadmit marks slot empty and forgets expert e, so the next admit(e) misses and re-uploads. admit commits the
// slot mapping BEFORE the DMA, which is what lets the caller batch a layer's copies into one UploadBatch; if that
// upload fails the cache would claim e is resident in a slot holding the EVICTED expert's bytes, and the next
// token routing e would hit, skip the DMA and run the wrong expert with no error anywhere. Rolled back to EMPTY
// rather than to the evicted expert: a failed batch may have completed some copies, so the slot's contents are
// unknown.
func (c *expertCache) unadmit(slot int, e uint32) {
	if c.inSlot[slot] == int32(e) {
		c.inSlot[slot] = -1
	}
	if c.slotOf[e] == int32(slot) {
		c.slotOf[e] = -1
	}
}

// appendExpertSlot QUEUES one expert's weight+scales copy from the pinned host source into device slot `slot`. It
// does not upload: loadRoutedExperts submits the layer's whole batch with a single gpu.UploadBatch, so one
// synchronize covers every miss in the layer instead of two per miss (each gpu.Upload ends in a full device
// Synchronize, and a MoE decode token loads ~120 slots). UploadBatch still synchronizes before it returns, so the
// guarantee equals Upload's, including the race that sync exists for (non-blocking streams unordered against the
// null stream); nothing moves into a caller's hands.
func (r *cudaResident) appendExpertSlot(w *cudaWQ, e, slot int) {
	srcW, srcS := w.srcW.Bytes(), w.srcS.Bytes()
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	r.expBatch = append(r.expBatch,
		gpu.HostCopy{Dst: w.W.At(slot * w.perExpertW * 4), Src: srcW[wOff : wOff+wLen]},
		gpu.HostCopy{Dst: w.ws16.At(slot * w.perExpertS * 2), Src: srcS[sOff : sOff+sLen]},
	)
	if r.cacheProf {
		r.profWBytes += uint64(wLen)
		r.profWCalls++
		r.profSBytes += uint64(sLen)
		r.profSCalls++
	}
}

// pendingAdmit is one provisional slot claim made by loadRoutedExperts before its DMA.
type pendingAdmit struct {
	slot   int
	expert uint32
}

// rollbackAdmits undoes every admit made in the current loadRoutedExperts call.
func (r *cudaResident) rollbackAdmits(c *expertCache) {
	for _, p := range r.pendingAdmits {
		c.unadmit(p.slot, p.expert)
	}
	r.pendingAdmits = r.pendingAdmits[:0]
}

// allocSlotStacks allocates the slot stacks of every routed layer in moeLayers, r.cacheSlots deep. record keeps each
// request's size in dbgAllocSizes (the build's allocation; a resize is not recorded). It runs on the executor and panics on
// an allocation failure, as every allocation helper does.
func (r *cudaResident) allocSlotStacks(moeLayers []int, record bool) {
	defer r.onLine(mem.LineExperts)()
	// Issue LARGEST FIRST across all layers, not group by group. The total is identical (capSlots and the granularity
	// form are untouched; only the order moves). This is a packing improvement, not a fix: the real constraint is a
	// driver ALLOCATION FLOOR (~144 MiB that cuMemGetInfo reports free and cuMemAlloc will not hand out at any request
	// size, TestAllocFloor), and the leftover after allocSlots must exceed it, which slotMarginBytes provides. The
	// order is kept on its measured merit (it drains ~27 MiB further before hitting the floor), not on the refuted
	// contiguity theory it was introduced for (docs/code-notes/cuda.md#allocSlots.order).
	//
	// Sort by MEASURED request size, not an assumed stride order: per-layer geometry is not guaranteed uniform (Gemma
	// 4's KV widths differ per layer, and slotBytesPerLayer exists because layer 0 can be dense). A stable sort keeps
	// layer order within each size class, so the sequence stays deterministic.
	type slotReq struct {
		w    *cudaWQ
		n    int // element count for the allocator
		size int // bytes, for ordering
		isW  bool
	}
	reqs := make([]slotReq, 0, 4*len(moeLayers))
	for _, i := range moeLayers {
		L := &r.layers[i]
		for _, w := range []*cudaWQ{&L.expGU, &L.expDown} {
			reqs = append(reqs,
				slotReq{w, r.cacheSlots * w.perExpertW, r.cacheSlots * w.perExpertW * 4, true},
				slotReq{w, r.cacheSlots * w.perExpertS, r.cacheSlots * w.perExpertS * 2, false})
		}
	}
	slices.SortStableFunc(reqs, func(a, b slotReq) int { return cmp.Compare(b.size, a.size) })
	for _, q := range reqs {
		if q.isW {
			q.w.W = r.au32(q.n)
		} else {
			q.w.ws16 = gpu.NewBufferLenOfOn[uint16](r.dev, r.line, q.n)
		}
		// Recording only: the REQUESTED byte size of each allocation, in issue (sorted) order, so predicted consumption
		// can be recomputed with per-buffer 2 MiB rounding instead of a raw sum. Every gate on it reads it
		// order-independently (counts, distinct sizes, occurrences, sum).
		if record {
			r.dbgAllocSizes = append(r.dbgAllocSizes, q.size)
		}
	}
}

// ExpertCacheStats implements decoder.ResidentRuntimeSubsystem: the granted slots per layer and the C′ cache's lookups summed over
// the MoE layers.
func (r *cudaResident) ExpertCacheStats() (decoder.ExpertCacheStats, bool) {
	if !r.cacheExperts {
		return decoder.ExpertCacheStats{}, false
	}
	st := decoder.ExpertCacheStats{Slots: r.cacheSlots}
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			st.Hits += c.hits
			st.Misses += c.misses
		}
	}
	return st, true
}

// CacheProfForTest reports the C′ round-trip decomposition (stall / host / dma) and the call
// count. Zero unless GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) CacheProfForTest() (stall, host, dma time.Duration, calls uint64) {
	return r.profStall, r.profHost, r.profDMA, r.profCalls
}

// profMark closes one decode class: drains the stream, books the time since the previous mark
// into acc, and restarts the clock. No-op unless cacheProf. A drain error is parked in launchErr
// (the sticky accumulator launchToken already returns) rather than dropped.
func (r *cudaResident) profMark(acc *time.Duration) {
	if !r.cacheProf {
		return
	}
	if e := r.stream.Sync(); e != nil && r.launchErr == nil {
		r.launchErr = e
	}
	now := time.Now()
	*acc += now.Sub(r.profT0)
	r.profT0 = now
}

// initOverlap builds the overlap path's resources: the DMA stream, the router event, one event per
// possible miss rank.
func (r *cudaResident) initOverlap() error {
	r.dmaQ = r.dev.NewCommandQueue()
	ev, e := r.dev.NewEvent()
	if e != nil {
		return e
	}
	r.evRoute = ev
	r.evMiss = make([]gpu.Event, r.topK)
	for k := range r.evMiss {
		if r.evMiss[k], e = r.dev.NewEvent(); e != nil {
			return e
		}
	}
	r.missEv = make([]int, r.topK)
	for j := range r.missEv {
		r.missEv[j] = -1
	}
	return nil
}

// recordRouted marks the router's position on r.stream — the ONLY thing loadRoutedExperts waits for
// in overlap mode. Called by every MoE pre-half right after its route kernel.
func (r *cudaResident) recordRouted() error {
	if !r.overlap {
		return nil
	}
	return r.stream.Record(r.evRoute)
}

// waitMiss makes r.stream wait, device-side, for rank j's expert DMA if that rank missed the cache
// this call. Hit ranks — and every rank when overlap is off — pass straight through, so the ranks
// ahead of a miss execute under its DMA in exactly the order they always did.
func (r *cudaResident) waitMiss(j int) error {
	if !r.overlap || r.missEv[j] < 0 {
		return nil
	}
	return r.stream.Wait(r.evMiss[r.missEv[j]])
}

// dmaExpertSlot enqueues one expert's weight+scales copy on dmaQ, stream-ordered and unsynchronized —
// the overlap-mode twin of appendExpertSlot.
func (r *cudaResident) dmaExpertSlot(w *cudaWQ, e, slot int) error {
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	// UploadAsyncAtFrom, not UploadAsyncAt+.Host(): srcW/srcS may be either MappedHostBuffer origin (NewMappedHostBuffer
	// or, under GOINFER_MOE_PIN_REGISTER, RegisterMappedHostBuffer) and .Host() returns nil for the latter, which has no
	// *gpu.HostBuffer to hand UploadAsyncAt, only Bytes(); UploadAsyncAtFrom reads through either origin.
	if err := r.dmaQ.UploadAsyncAtFrom(w.W.At(slot*w.perExpertW*4), w.srcW, wOff, wLen); err != nil {
		return err
	}
	if err := r.dmaQ.UploadAsyncAtFrom(w.ws16.At(slot*w.perExpertS*2), w.srcS, sOff, sLen); err != nil {
		return err
	}
	if r.cacheProf {
		r.profWBytes += uint64(wLen)
		r.profWCalls++
		r.profSBytes += uint64(sLen)
		r.profSCalls++
	}
	return nil
}

// SetOverlapForTest flips the overlap path between tokens for an A/B on one loaded model. Returns
// false (and changes nothing) if the model was loaded without the overlap resources.
func (r *cudaResident) SetOverlapForTest(on bool) bool {
	if r.evMiss == nil {
		return false
	}
	r.overlap = on
	return true
}

// THERE ARE TWO INDEX SPACES HERE AND THEY DIVERGE ONLY WHEN EXPERT CACHING IS ON. Keeping them
// as two named accessors rather than one is the whole guard: with caching off they return the
// same buffer, so a site that binds the wrong one is correct in every configuration anyone has
// run, and wrong — silently, with plausible logits — in the one configuration that needs it.
//
//	expIdx        WHERE the weights live  → slot ids when caching, expert ids otherwise
//	expertBiasIdx WHICH expert is running → ALWAYS expert ids
//
// expIdx is the idx argument the expert GEMVs bind: the slot ids when caching (slot j holds
// routed expert j), else the router's real rIdx (fully-resident path).
func (r *cudaResident) expIdx() Buffer {
	if r.cacheExperts {
		return r.slotIdx
	}
	return r.rIdx
}

// expertBiasIdx is the index for PER-EXPERT TABLES that are uploaded ONCE for all experts and indexed on the
// device (today only gpt-oss's [nExpert][2*I] gate||up bias table). It is always the router's real expert ids,
// NEVER the slot ids: the table is expert-indexed and does not move when an expert is streamed into a slot.
// Binding expIdx here is a silent defect: glu_quant_gptoss does `biasGU + idx[slot]*2*I`, so with caching on it
// selects the bias row by SLOT id, the wrong expert's gate/up biases, with no error and plausible output
// (docs/code-notes/cuda.md#expertBiasIdx).
func (r *cudaResident) expertBiasIdx() Buffer { return r.rIdx }

func u32bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func u16bytes(v []uint16) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*2)
}

// mapBytes stages src as the C′ DMA source: pinned (device-mapped) host memory holding exactly src's bytes. src is
// caller-owned and already fully populated (the merged expert-stack slice packWeight built), and never touched
// again by anyone else, which is what makes the register-in-place arm safe: the MappedHostBuffer keeps src's
// backing array reachable for as long as the pin lives. Register in place is the default
// (GOINFER_MOE_PIN_REGISTER=0 restores allocate-then-copy): it pins the caller's own bytes, so the DMA source is
// byte-for-byte the same either way and no numerics are at stake (docs/code-notes/cuda.md#mapBytes.register).
// Panics on the UVA guard failing: eligibility already asserted a UVA device, so a failure here is a broken
// invariant, not a runtime condition.
func (r *cudaResident) mapBytes(src []byte) *gpu.MappedHostBuffer {
	if r.knobValue("GOINFER_MOE_PIN_REGISTER") != "0" { // register in place, above
		mb, err := r.dev.RegisterMappedHostBufferOn(mem.LinePinnedExperts, src)
		if err != nil {
			panic(fmt.Sprintf("cacheWQ: RegisterMappedHostBuffer(%d): %v", len(src), err))
		}
		return mb
	}
	mb, err := r.dev.NewMappedHostBufferOn(mem.LinePinnedExperts, len(src))
	if err != nil {
		panic(fmt.Sprintf("cacheWQ: NewMappedHostBuffer(%d): %v", len(src), err))
	}
	copy(mb.Bytes(), src)
	return mb
}

// dbgState is cudaResident's measurement instruments: VRAM readings around the slot allocation and the first launches, the
// per-launch probe and the routing hook. Each is zero or off unless a test sets it.
type dbgState struct {
	// Instrument fields, RECORDING ONLY, never read by production logic: free device VRAM immediately before and after
	// allocSlots in the SAME process, so a test measures consumption without a cross-run assumption.
	dbgFreeBefore, dbgFreeAfter uint64
	dbgPredInline               int64  // consumption predicted by the arithmetic production executes
	dbgPredCapSlots             int64  // consumption predicted from capSlots' chosen count
	dbgSlotsInline              int    // slot count the INLINE copy at allocSlots chose
	dbgSlotsCapSlots            int    // slot count capSlots would choose from the same free
	dbgFreeBeforeLaunch         uint64 // free VRAM immediately before the first forward's launches
	dbgFreePreLaunch            uint64 // free VRAM immediately before the MOST RECENT launch
	dbgLaunchTrace              []string
	// dbgProbe enables the per-launch pre-launch probe. A FIELD, not a package var read from the environment: a
	// package var is initialised before t.Setenv can run, so it would silently record nothing; the test sets the field
	// on the resident it already holds. Off by default: the probe costs a MemInfo round-trip and a reflection scan on
	// EVERY launch.
	dbgProbe bool
	// routeRecord, when non-nil, is called once per loadRoutedExperts call with the layer index and that call's routed
	// expert ids (r.hostIdx[:topK]): a measurement hook for how many DISTINCT experts a whole M-row chunk touches per
	// layer. A field for the same reason dbgProbe is.
	routeRecord func(layer int, ids []uint32)
}

// expertStream is cudaResident's C′ expert streaming: the per-layer device slots the routed experts are copied into, the
// overlapped copy stream and its events, the pending admissions, and the routing round trip's profiling counters. Zero unless
// the experts are cached.
type expertStream struct {
	cacheExperts bool // C′: routed experts DMA'd host→VRAM slots per token (device read; correct)
	// unpinnedExperts (MoEExpertsUnpinned) packs each expert stack into an unlinked, read-only file mapping under unpinnedDir instead of pinned
	// host memory, so the slots upload from pageable pages; pageableStacks counts the stacks it built that way.
	unpinnedExperts bool
	unpinnedDir     string
	pageableStacks  int
	// overlap (GOINFER_MOE_DMA_OVERLAP, default on with C′; off under graphs): the C′ round trip no longer drains the
	// stream. The host waits on evRoute (recorded right after the router), the misses are copied on dmaQ, a second
	// stream, while the kernels issued after the router keep running, and segC's rank loop waits device-side only
	// before a rank that missed. Everything keeps its launch order and arithmetic, so the output is bit-identical to
	// the draining path; only the idle time moves. Ceiling and decision rule:
	// docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md.
	overlap     bool
	dmaQ        Queue
	evRoute     gpu.Event
	evMiss      []gpu.Event        // per miss of one call (≤ topK): recorded on dmaQ after that miss's two copies
	missEv      []int              // per rank: index into evMiss, or -1 for a hit
	slotIdxHost *HostBuffer[uint8] // pinned source of the per-token slot-id upload (UploadAsync needs pinned)
	cacheProf   bool               // GOINFER_MOE_CACHE_PROF: time the per-layer routing round trip
	profStall   time.Duration
	profHost    time.Duration
	profDMA     time.Duration
	// The expert DMA is split into the BIG weight copy and the TINY scale copy, with bytes and call counts for each:
	// if a small scale upload costs about as much as a large weight upload, the path is per-call-overhead bound and
	// the fix is batching. Zero cost when cacheProf is off.
	profWBytes, profSBytes uint64
	profWCalls, profSCalls uint64
	// expBatch queues one layer's host-to-device copies, issued together by one gpu.UploadBatch (appendExpertSlot).
	// profWCalls/profSCalls count logical COPIES, so the per-token copy count stays comparable across the batching;
	// profSyncCalls counts the SYNCHRONIZES, which is what batching moves. There is no per-kind timing: the copies
	// are appended here and issued together later, so a "weight upload took Xµs" figure would name something that no
	// longer happens.
	expBatch []gpu.HostCopy
	// pendingAdmits are the slot claims loadRoutedExperts made before its DMA, rolled back if the upload fails so the
	// cache cannot assert residency for bytes never copied.
	pendingAdmits []pendingAdmit
	profBatchTime time.Duration
	profSyncCalls uint64
	profCalls     uint64
	// Per-class decode timing for a g4moe token, sync-bounded (profMark), same cacheProf gate. It
	// exists to size the compute/DMA overlap ceiling: only dense (independent of the routing) and
	// the hit-expert prefix of segC can run under a miss DMA, so those two are the classes that
	// bound any overlap design. Sync-to-sync, so each class includes its own launch latency and
	// the token is slower with the profiler on; the numbers are upper bounds on class cost.
	profT0                                                                 time.Time
	profAttn, profDense, profRouter, profClear, profRT, profSegC, profHead time.Duration

	cacheSlots int      // C′ step 2: device slots per layer (≥ topK; = topK ⇒ step-1 fresh-load, no reuse)
	slotIdx    Buffer   // C′: per-token slot ids for the routed experts, bound as the GEMV's idx
	hostIdx    []uint32 // C′: scratch for the per-layer rIdx device→host readback
	hostSlot   []uint32 // C′: scratch for the per-token slot ids uploaded to slotIdx
	// cacheSlotsAsked is the slot count the load asked for, before allocSlots capped it to free VRAM: the ceiling a resize grows to.
	cacheSlotsAsked int
	cacheSlotsReq   int   // slots REQUESTED (pre-cap), so errors can name both
	dbgAllocSizes   []int // every requested slot-buffer size, recording only
	// planSlotGeom is the expert-cache geometry the memory plan supplies from the planned stacks (expertSlotGeom); nil in a build.
	planSlotGeom *slotGeom
}
