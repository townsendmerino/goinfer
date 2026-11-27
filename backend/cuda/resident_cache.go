//go:build cuda

package cuda

import (
	"cmp"
	"fmt"
	"slices"
	"time"
	"unsafe"

	"github.com/townsendmerino/goinfer/backend/cuda/internal/fitdecide"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/moecache"
)

// expertCache aliases moecache.SlotCache for per-layer LRU expert slot residency.
type expertCache = moecache.SlotCache

type pendingAdmit = moecache.PendingAdmit

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

// rollbackAdmits undoes every admit made in the current loadRoutedExperts call.
func (r *cudaResident) rollbackAdmits(c *expertCache) {
	c.Rollback(r.pendingAdmits)
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
			st.Hits += c.Hits()
			st.Misses += c.Misses()
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

// slotBytesPerLayer is the device VRAM one slot's worth of BOTH expert projections costs (int4 weight + f16
// scales), used to size the cache to free VRAM. It must be measured from a ROUTED layer, not from layer 0: layer 0
// is dense on every family with `first_k_dense_replace` (GLM-4.5/4.6, DeepSeek-V2/V3, Kimi), so its strides are
// zero and the caller's `budget / len(moeLayers) / perLayer` would be an integer divide by zero, a panic on the
// executor goroutine.
func (r *cudaResident) slotBytesPerLayer(layer int) int {
	if layer < 0 || layer >= len(r.layers) {
		return 0
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return (gu.perExpertW+dn.perExpertW)*4 + (gu.perExpertS+dn.perExpertS)*2
}

// slotStrides is the per-slot byte size of each buffer allocated per MoE layer, IN THE ORDER AND
// GROUPING allocSlots allocates them. The sum is slotBytesPerLayer; the split matters because the
// driver charges each allocation its own whole quanta, so the total is not a function of the sum.
func (r *cudaResident) slotStrides(layer int) []int64 {
	if layer < 0 || layer >= len(r.layers) {
		return nil
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return []int64{
		int64(gu.perExpertW) * 4, int64(gu.perExpertS) * 2,
		int64(dn.perExpertW) * 4, int64(dn.perExpertS) * 2,
	}
}

// slotMarginBytes is the launch-time headroom the expert-cache cap leaves free; TestAllocFloor checks it against the measured
// driver allocation floor (docs/code-notes/cuda.md#slotMarginBytes).
const slotMarginBytes = fitdecide.SlotMarginBytes

// expertCacheFloorBytes is the smallest expert cache allocSlots accepts, top-k slots on every routed layer (slotRequirement), or 0 for a
// load with no routed layer or no per-expert stride.
func (r *cudaResident) expertCacheFloorBytes() int64 {
	n, strides := r.expertSlotGeom()
	if n == 0 {
		return 0
	}
	return fitdecide.SlotRequirement(r.topK, int64(n), strides)
}

// expertSlotGeom is the expert cache's routed-layer count and per-slot strides (slotStrides, from the first routed layer); 0 and nil
// for a load with no cached layer or no per-expert stride. The memory plan, which uploads nothing, supplies the same figures from the
// planned stacks (planSlotGeom).
func (r *cudaResident) expertSlotGeom() (int, []int64) {
	if r.planSlotGeom != nil {
		return r.planSlotGeom.n, r.planSlotGeom.strides
	}
	var moeLayers []int
	for i := range r.layers {
		if r.layers[i].expGU.srcW != nil {
			moeLayers = append(moeLayers, i)
		}
	}
	if len(moeLayers) == 0 || r.slotBytesPerLayer(moeLayers[0]) <= 0 {
		return 0, nil
	}
	return len(moeLayers), r.slotStrides(moeLayers[0])
}

// slotGeom is an expert cache's routed-layer count and per-slot strides.
type slotGeom struct {
	n       int
	strides []int64
}

// allocSlots caps r.cacheSlots to the MEASURED free device VRAM (with a safety margin) and then
// allocates each MoE layer's slot buffers + its LRU cache. Called after the resident core + KV +
// scratch are up, so the cap reflects what is actually left — an over-large GOINFER_MOE_CACHE_SLOTS
// is capped-and-logged (the repo's "decline honestly at load" discipline), never OOM'd mid-build.
func (r *cudaResident) allocSlots() error {
	defer r.onLine(mem.LineExperts)()
	var moeLayers []int
	for i := range r.layers {
		if r.layers[i].expGU.srcW != nil {
			moeLayers = append(moeLayers, i)
		}
	}
	if len(moeLayers) == 0 {
		return nil
	}
	// Size from the FIRST ROUTED layer (moeLayers[0]), not layer 0: see slotBytesPerLayer.
	perLayer := r.slotBytesPerLayer(moeLayers[0])
	if perLayer <= 0 {
		// No routed layer reports a per-expert stride (a degenerate blob): nothing to cache and nothing to divide by.
		// Clear cacheExperts so the decode path cannot later read slots/expCache that were never allocated. This does NOT
		// yield a working "hold every expert" path (upExperts left the stacks host-mapped-only, so the expert GEMVs would
		// bind zero-value device buffers); it is unreachable with any real MoE checkpoint and exists only to fail safe. A
		// blob that trips it should be declined to the staged/CPU path upstream.
		r.cacheExperts = false
		logs.WarnText(zeroExpertBytesLine(moeLayers[0]))
		return nil
	}
	strides := r.slotStrides(moeLayers[0])
	r.cacheSlotsAsked = r.cacheSlots
	if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
		r.dbgFreeBefore = f0 // recording only
	}
	free, err := slotsFreeVRAM(r)
	if memReadSeenForTest != nil {
		memReadSeenForTest("slots", int64(free), err == nil)
	}
	d := slotsDecision(free, err == nil, r.extraBytes, len(moeLayers), strides, r.topK, r.cacheSlots)
	for _, l := range d.logs {
		logs.InfoText(l)
	}
	if d.err != nil {
		return d.err
	}
	r.cacheSlots = d.slots
	r.allocSlotStacks(moeLayers, true)
	for _, i := range moeLayers {
		r.layers[i].expCache = moecache.NewSlotCache(r.nE, r.cacheSlots)
	}
	if f1, _, e1 := r.dev.Context().MemInfo(); e1 == nil {
		r.dbgFreeAfter = f1 // recording only
	}
	// The inline and capSlots figures are one value: there is one implementation.
	r.dbgSlotsInline = r.cacheSlots
	r.dbgPredInline = fitdecide.SlotRequirement(r.cacheSlots, int64(len(moeLayers)), strides)
	r.dbgSlotsCapSlots = r.cacheSlots
	r.dbgPredCapSlots = r.dbgPredInline
	return nil
}

// slotsFreeVRAM is allocSlots' free-VRAM probe, a variable so a test can stand in a card with less room.
var slotsFreeVRAM = func(r *cudaResident) (uint64, error) {
	free, _, err := r.dev.Context().MemInfo()
	return free, err
}

// zeroExpertBytesLine is the line allocSlots logs when the first routed layer has no per-expert bytes and the cache is turned off.
func zeroExpertBytesLine(layer int) string {
	return fmt.Sprintf("[cuda] C′ cache: routed layer %d reports zero per-expert bytes — expert cache disabled\n", layer)
}

// slotsOut is allocSlots' decision: the slots per layer, the lines the build logs, and the refusal when not even the minimum fits.
type slotsOut struct {
	slots int
	logs  []string
	err   error
}

// slotsDecision is allocSlots' decision over the free VRAM it read (fitdecide.SlotsDecision).
func slotsDecision(free uint64, ok bool, extra int64, nLayers int, strides []int64, topK, want int) slotsOut {
	d := fitdecide.SlotsDecision(free, ok, extra, nLayers, strides, topK, want)
	return slotsOut{slots: d.Slots, logs: d.Logs, err: d.Err}
}

// loadRoutedExperts reads back the router's idx (device→host — C′'s acknowledged per-layer sync),
// admits each routed expert into the layer's slot cache (DMAing only cache misses), and uploads the
// per-token slot ids the GEMV binds as `idx` (slot j ← the slot now holding routed expert j).
func (r *cudaResident) loadRoutedExperts(L *cudaLayer) error {
	// Timing seam (GOINFER_MOE_CACHE_PROF). This is the only host round trip on the decode path, once per MoE layer
	// per token, and it splits into three costs that need different fixes: stall (the stream drain waiting for the
	// router; fixed by removing the round trip, not by making it faster), host (the LRU bookkeeping; ordinary
	// optimization) and dma (the H2D of missed experts; more slots or fewer bytes). Off by default, one branch when
	// off; it adds no syncs, because the stall it measures is a sync that already exists.
	var t0 time.Time
	if r.cacheProf {
		t0 = time.Now()
	}
	if r.overlap {
		// Wait for the ROUTER, not the stream: everything issued after it keeps executing while the
		// idx is read back and the misses are copied on dmaQ.
		if e := r.evRoute.Sync(); e != nil {
			return e
		}
	} else if e := r.stream.Sync(); e != nil {
		return e
	}
	if r.cacheProf {
		r.profStall += time.Since(t0)
		t0 = time.Now()
	}
	if e := gpu.Download(r.rIdx, r.hostIdx[:r.topK]); e != nil {
		return e
	}
	if r.routeRecord != nil {
		r.routeRecord(L.idx, r.hostIdx[:r.topK])
	}
	c := L.expCache
	c.RecordStageDemand(r.hostIdx[:r.topK])
	r.expBatch = r.expBatch[:0] // reused across layers; the slice keeps its capacity
	// Every admit made in THIS call is provisional until the batch lands. On any error below they are rolled back, so
	// a failed upload cannot leave the cache asserting residency for bytes that were never copied.
	r.pendingAdmits = r.pendingAdmits[:0]
	for j := 0; j < r.topK; j++ {
		e := r.hostIdx[j]
		slot, hit := c.Admit(e)
		r.hostSlot[j] = uint32(slot)
		if r.overlap {
			r.missEv[j] = -1
		}
		if !hit {
			r.pendingAdmits = append(r.pendingAdmits, pendingAdmit{Slot: slot, Expert: e})
			var td time.Time
			if r.cacheProf {
				r.profHost += time.Since(t0)
				td = time.Now()
			}
			if r.overlap {
				// Both copies on dmaQ, then the event this rank's launches will wait on. A later rank
				// routed to the same expert is ordered behind this rank's wait by the stream itself.
				k := len(r.pendingAdmits) - 1
				r.missEv[j] = k
				err := r.dmaExpertSlot(&L.expGU, int(e), slot)
				if err == nil {
					err = r.dmaExpertSlot(&L.expDown, int(e), slot)
				}
				if err == nil {
					err = r.dmaQ.Record(r.evMiss[k])
				}
				if err != nil {
					r.rollbackAdmits(c)
					return err
				}
			} else {
				r.appendExpertSlot(&L.expGU, int(e), slot)
				r.appendExpertSlot(&L.expDown, int(e), slot)
			}
			if r.cacheProf {
				r.profDMA += time.Since(td) // overlap: enqueue time, not transfer time
				t0 = time.Now()
			}
		}
	}
	if r.overlap {
		// The slot-id table rides r.stream, ordered before segC by the stream; its pinned source is
		// next rewritten only after the next layer's evRoute wait, which follows this layer's segC.
		hs := r.slotIdxHost.Slice()
		for j := 0; j < r.topK; j++ {
			v := r.hostSlot[j]
			hs[4*j], hs[4*j+1], hs[4*j+2], hs[4*j+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
		}
		if e := r.stream.UploadAsync(r.slotIdx, r.slotIdxHost); e != nil {
			r.rollbackAdmits(c)
			return e
		}
		if r.cacheProf {
			r.profHost += time.Since(t0)
			r.profCalls++
		}
		return nil
	}
	// One synchronize for the WHOLE layer: every expert-slot miss plus the per-token slot-index upload the GEMV reads
	// this round's routing from, folded into the SAME batch (slotIdx is always present and tiny, so keeping it apart
	// would only add a synchronize). All destinations are slot buffers on the one resident context, so UploadBatch's
	// mixed-context refusal should never fire; if it does, the layer's buffers are not what this code believes.
	r.expBatch = append(r.expBatch, gpu.HostCopy{Dst: r.slotIdx, Src: u32bytes(r.hostSlot[:r.topK])})
	var tb time.Time
	if r.cacheProf {
		r.profHost += time.Since(t0)
		tb = time.Now()
	}
	e := gpu.UploadBatch(r.expBatch)
	if e != nil {
		// The expert bytes and/or the slot-index table may be partially applied: the cheap conservative move is to make
		// the next call re-establish both.
		r.rollbackAdmits(c)
	}
	if r.cacheProf {
		r.profBatchTime += time.Since(tb)
		r.profSyncCalls++
		r.profDMA += time.Since(tb)
		r.profCalls++
	}
	return e
}

// ResizeExpertCache implements decoder.ResidentRuntimeSubsystem: it sizes the C′ cache to what fits in free VRAM with reserve
// bytes left over for an allocation that comes after it, up to the slot count the load asked for. A resize frees and
// reallocates every routed layer's slot stacks, so the cache restarts empty; the hit and miss counts carry over. It runs on
// the executor between forwards. It refuses under CUDA graphs, whose captured segments hold the stacks' addresses. A fit
// below top-k is an error, and the cache is left as it was.
func (r *cudaResident) ResizeExpertCache(reserve int64) (slots int, err error) {
	err = r.do(func() error {
		if !r.cacheExperts {
			return fmt.Errorf("cuda: this resident keeps no expert cache")
		}
		if r.graphs {
			return fmt.Errorf("cuda: the expert cache cannot be resized under CUDA graphs")
		}
		var moeLayers []int
		for i := range r.layers {
			if r.layers[i].expCache != nil {
				moeLayers = append(moeLayers, i)
			}
		}
		if len(moeLayers) == 0 {
			return fmt.Errorf("cuda: no routed layer holds an expert cache")
		}
		strides := r.slotStrides(moeLayers[0])
		nL := int64(len(moeLayers))
		free, _, e := r.dev.Context().MemInfo()
		if e != nil {
			return e
		}
		// What the stacks hold now is free once they are released: the same rounded figure capSlots priced them at.
		budget := fitdecide.ReservedBudget(int64(free)+fitdecide.SlotRequirement(r.cacheSlots, nL, strides), reserve)
		fit, decline := fitdecide.CapSlots(budget, nL, strides, r.topK, r.cacheSlotsAsked)
		if decline {
			return fmt.Errorf("cuda: the expert cache cannot keep %d slots/layer beside %s reserved (%d fit)", r.topK, mem.GiB(reserve), fit)
		}
		if fit == r.cacheSlots {
			return nil
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if r.overlap {
			if e := r.dmaQ.Sync(); e != nil {
				return e
			}
		}
		for _, i := range moeLayers {
			L := &r.layers[i]
			for _, w := range []*cudaWQ{&L.expGU, &L.expDown} {
				r.dev.ReleaseBuf(w.W)
				r.dev.ReleaseBuf(w.ws16)
				w.W, w.ws16 = Buffer{}, Buffer{}
			}
		}
		r.cacheSlots = fit
		r.allocSlotStacks(moeLayers, false)
		for _, i := range moeLayers {
			old := r.layers[i].expCache
			c := moecache.NewSlotCache(r.nE, fit)
			c.CopyStatsFrom(old)
			r.layers[i].expCache = c
		}
		return nil
	})
	return r.cacheSlots, err
}
