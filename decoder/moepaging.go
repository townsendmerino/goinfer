package decoder

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
)

// expertPager bounds the resident RAM of a MoE model's expert weights by paging them on demand out of the read-only
// .giw mapping (docs/ideas-weight-memory.md). The router's top-k selection is the demand signal: a 35B-A3B holds ~32 GB
// of experts but activates only K*L per token. The generic span-residency pager (touch faults in with WILLNEED, eviction
// over budget releases with DONTNEED) lives in aikit/mmap.SpanCache, which this pager runs with the frequency-aware
// EvictLeastRecent policy (newExpertPager). This type holds the MoE-specific half: which experts alias the mapping and
// the touch hook the router calls (moeMLP in mlp.go for [NumExperts]expertWeights families, gemma4MoEFFN for gemma4's
// fused gate||up + down experts). Releasing is lossless: the mapping is read-only and file-backed, so an evicted expert
// re-faults from disk and output stays bit-identical; only the cold-miss fault costs.
//
// Only experts whose quantized weights alias the mapping are managed; heap-backed weights (a GGUF load) and the
// always-on shared expert are left alone.
//
// Two backing modes, chosen once at build time (newExpertPager):
//   - mmap+madvise (the default except on darwin): cache aliases the read-only mapping directly, WILLNEED faults it in,
//     DONTNEED releases it. Zero-copy, but on darwin DONTNEED is a no-op (madvise_darwin.go), so this mode has no real
//     RAM cap on macOS.
//   - owned-buffer pread (the darwin default; Options.MoEPager elsewhere): pool holds a fixed set of owned buffers it
//     fills via pread, giving a firm cap on every platform at the cost of a memcpy per miss and losing .giw zero-copy
//     aliasing.
//
// Guarded by two mutexes (mu and readMu, below): the pager lives on *Model and StreamWeights supports concurrent decode
// streams, so its shared LRU state is locked (SpanCache is not internally locked, and pool needs protection for a
// different reason).
type expertPager struct {
	// mu guards touch()'s own mutation (cache.Touch or pool.ensure), held internally for the duration of that one call in
	// either mode. touch() is therefore safe to call from any goroutine with no outer locking
	// (TestExpertPager_concurrentNoRace). It does not by itself protect a caller's reads of the WeightMat fields after
	// touch() returns: see readMu.
	mu sync.Mutex
	// readMu is pool mode's additional requirement, on top of mu: a slot's owned buffer is
	// mutable storage a competing miss can refill mid-read, unlike an mmap alias (whose bytes
	// never move) — so pool-mode callers must hold readMu across touch() AND every subsequent
	// matmul read of the repointed WeightMat fields, for the duration of one MoE FFN call (see
	// expertBufferPool's doc comment). Lock/Unlock export this. A distinct mutex from mu (never
	// nested on the same object) — mmap-mode callers may take it too for uniformity, but
	// touch()'s own self-locking via mu already makes that mode safe without it.
	readMu   sync.Mutex
	cache    *mmap.SpanCache[unsafe.Pointer] // mmap+madvise mode; nil if pool is set
	pool     *expertBufferPool               // owned-buffer pread mode; nil if cache is set
	nExperts int                             // mapping-backed experts under management (for the banner)
	total    int64                           // total mapped expert bytes (for the banner)

	// minfltBase and majfltBase are the process-wide getrusage(RUSAGE_SELF) minor and major page-fault counts at pager
	// creation; faultDelta reports faults since. They are process-wide on purpose: getrusage cannot attribute a fault to a
	// mapping, so the delta is contaminated by whatever else the process does. That suits an A/B run with nothing else
	// happening, and it must not be read as per-pager. faultsOK false means the platform has no probe
	// (faultcount_other.go); pagerSummary then omits the term.
	minfltBase, majfltBase int64
	faultsOK               bool
}

// Lock/Unlock should wrap touch() and the matmul reads that follow it, for the duration of one
// MoE FFN call (see readMu's doc comment). Required for correctness in pool mode; a no-op-ish
// safety margin in mmap mode, where touch() is already self-contained-safe on its own.
func (p *expertPager) Lock()   { p.readMu.Lock() }
func (p *expertPager) Unlock() { p.readMu.Unlock() }

// MoEPagerDefault is the registered default pager mode: darwin, where MADV_DONTNEED is a no-op and mmap mode therefore
// cannot enforce its budget, gets the owned-buffer pool; every other platform keeps mmap mode, where DONTNEED works and
// the alias is free. It is the one source for the default: serve's --moe-pager default and a library Load with
// Options.MoEPager unset both resolve here.
func MoEPagerDefault(goos string) string {
	if goos == "darwin" {
		return "pool"
	}
	return "mmap"
}

// resolveMoEPagerPool reports whether the CPU expert pager uses the owned-buffer pool: an explicit
// Options.MoEPager wins; otherwise GOINFER_MOE_PREAD_CPU ("1"/"0") when set — preadCPU, read once at Load
// (loadKnob), so Options.Knobs can set it per model; otherwise the platform default.
func resolveMoEPagerPool(opt, preadCPU string) bool {
	switch opt {
	case "pool":
		return true
	case "mmap":
		return false
	}
	switch preadCPU {
	case "1":
		return true
	case "0":
		return false
	}
	return MoEPagerDefault(runtime.GOOS) == "pool"
}

// newExpertPager builds a pager over the experts of an mmap-backed MoE model, or returns nil when paging does not apply
// (not MoE, not mmap-backed, or no expert weights alias the mapping). budget <= 0 selects an automatic budget (about half
// of available RAM); it is clamped to [one expert, total expert bytes]. giwPath is the .giw file the mapping was built
// from (Model.GiwPath), used only when pool mode is requested, to open an independent fd for pread (the mmap's own fd is
// closed right after mapping).
func newExpertPager(w *Weights, mapping []byte, budget int64, giwPath string, pool bool) *expertPager {
	if w.arch.MoE == nil || len(mapping) == 0 {
		return nil
	}
	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))

	type member struct {
		key   unsafe.Pointer
		spans [][]byte
	}
	var members []member
	var poolMembers []poolMember
	var total, maxExpert int64
	// addExpert registers one expert under a stable identity (the address of its primary weight struct, the value the
	// forward touches), collecting only the projections that alias the mapping. MappedSpan returns nil for heap-backed
	// (GGUF) weights and the always-on shared expert, so those are skipped.
	//
	// A kind-4 tensor carries two on-disk representations (canonical and row4, docs/completed/task-w4a8-neon-bandwidth.md),
	// but the M==1 decode kernel (MatmulBTW4A8Into) reads only row4 whenever it is present, and this arch's forward is
	// always M==1. Register only the span that will be read: row4 when present, canonical otherwise, never both.
	// SpanCache.Touch issues MADV_WILLNEED on every span under a key, so registering both prefetches both copies from disk
	// for twice the I/O per miss (docs/completed/task-zeno-compare.md, "At-scale acceptance run").
	addExpert := func(key unsafe.Pointer, wms ...*linalg.WeightMat) {
		var spans [][]byte
		var fields []expertField
		var n int64
		pageable := true
		for _, wm := range wms {
			s := wm.MappedSpanRow4(base, end)
			if len(s) == 0 {
				s = wm.MappedSpan(base, end)
			}
			if len(s) == 0 {
				continue
			}
			spans = append(spans, s)
			n += int64(len(s))
			if f, ok := buildExpertField(wm, base, end); ok {
				fields = append(fields, f)
			} else {
				pageable = false // a field that mapped for madvise purposes didn't resolve for pread — don't offer this member to pool mode
			}
		}
		if n == 0 {
			return // heap-backed — nothing to page
		}
		members = append(members, member{key, spans})
		if pageable && len(fields) == len(wms) {
			poolMembers = append(poolMembers, poolMember{key, fields})
		}
		total += n
		if n > maxExpert {
			maxExpert = n
		}
	}
	for li := range w.Layers {
		// Mixtral-style experts: [NumExperts]expertWeights, gate/up/down separate.
		exps := w.Layers[li].Experts
		for ei := range exps {
			ex := &exps[ei]
			addExpert(unsafe.Pointer(ex), &ex.Gate, &ex.Up, &ex.Down)
		}
		// Gemma 4 experts: fused gate‖up + down, held in the gemma4moe sub-block. Keyed by
		// the gateUp element address — the same identity gemma4MoEFFN touches.
		if gm := w.Layers[li].gemma4moe; gm != nil {
			for ei := range gm.expertsGateUp {
				addExpert(unsafe.Pointer(&gm.expertsGateUp[ei]), &gm.expertsGateUp[ei], &gm.expertsDown[ei])
			}
		}
	}
	if total == 0 {
		return nil // no mapping-backed experts to manage
	}
	if budget <= 0 {
		budget = mmap.AutoBudget()
	}
	if budget > total {
		budget = total // never budget more than exists
	}
	if budget < maxExpert {
		budget = maxExpert // must hold at least one expert
	}
	minfltBase, majfltBase, faultsOK := processFaultCounts()
	if pool && giwPath != "" && len(poolMembers) == len(members) {
		pool, err := newExpertBufferPool(giwPath, poolMembers, budget, w.arch.MoE.TopK)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decoder: MoE pread pool init failed (%v), falling back to mmap paging\n", err)
		} else {
			return &expertPager{pool: pool, nExperts: len(poolMembers), total: total,
				minfltBase: minfltBase, majfltBase: majfltBase, faultsOK: faultsOK}
		}
	}
	// Frequency-aware (classic LRU tail) eviction: the router's demand is skewed frequency, not a scan, so the hot set must
	// stay resident. SpanCache's default is scan-resistant (evict-most-recent), right for a cyclic scan but evicting exactly
	// the hot experts here; EvictLeastRecent restores it (aikit mmap.EvictPolicy).
	cache := mmap.NewSpanCacheWithPolicy[unsafe.Pointer](budget, mmap.EvictLeastRecent)
	for _, m := range members {
		cache.Add(m.key, m.spans)
	}
	return &expertPager{cache: cache, nExperts: len(members), total: total,
		minfltBase: minfltBase, majfltBase: majfltBase, faultsOK: faultsOK}
}

// touch records that the expert at key is needed now: it becomes most-recently-used and, if it was not resident, is
// faulted in (mmap mode, evicting the LRU tail to stay within budget) or refilled into an owned pread slot (pool mode).
// A no-op for experts the pager does not manage. Safe to call directly and concurrently (mu is held internally), but in
// pool mode the caller must also hold Lock() across this call and its subsequent reads of the repointed WeightMat
// fields, or a concurrent touch() can repoint them mid-read (see readMu).
func (p *expertPager) touch(key unsafe.Pointer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pool != nil {
		if err := p.pool.ensure(key); err != nil {
			// No error return here (matching the mmap path, which cannot fail either) --
			// a pread failure mid-forward isn't recoverable at this call depth. Leaving the
			// WeightMat fields at their last-good contents and logging is the least-bad
			// option; this should only happen on real I/O failure or on-disk corruption.
			fmt.Fprintf(os.Stderr, "decoder: MoE pread refill failed for expert: %v\n", err)
		}
		return
	}
	p.cache.Touch(key)
}

// stats returns cumulative (hits, misses, evictions) over all touch calls. A
// non-zero eviction count means the budget was actually enforced (the LRU tail was
// released), as opposed to mere cold-start misses.
func (p *expertPager) stats() (hits, misses, evictions int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pool != nil {
		return p.pool.stats()
	}
	return p.cache.Stats()
}

// advisedBytes returns cumulative bytes fetched from disk over every miss: WILLNEED-hinted bytes in mmap mode, pread'd
// bytes in pool mode. It is a contamination-proof I/O check, independent of whatever else the machine's disk is doing: a
// member registering redundant spans (see addExpert) shows up directly as bytes-per-miss above the expected per-expert
// working set.
func (p *expertPager) advisedBytes() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pool != nil {
		return p.pool.bytesRead
	}
	return p.cache.AdvisedBytes()
}

// faultDelta reports this process's minor and major page faults since the pager was created, for comparing mmap mode
// (WILLNEED-triggered page faults) with pool mode (pread, which should add near none). The count is process-wide, not
// pager-attributed: see minfltBase.
func (p *expertPager) faultDelta() (minflt, majflt int64, ok bool) {
	if !p.faultsOK {
		return 0, 0, false
	}
	nowMin, nowMaj, ok := processFaultCounts()
	if !ok {
		return 0, 0, false
	}
	return nowMin - p.minfltBase, nowMaj - p.majfltBase, true
}

// close releases the pager's resources — currently only meaningful in pool mode (the
// independent pread fd; mmap mode owns nothing beyond the mapping itself, which the caller
// unmaps separately).
func (p *expertPager) close() error {
	if p.pool != nil {
		return p.pool.close()
	}
	return nil
}

// budget is the resident-bytes cap this pager was built with, in either mode; moeworkingset.go's working-set prediction
// asks it too, so the mode dispatch lives in one place.
func (p *expertPager) budget() int64 {
	if p.pool != nil {
		return p.pool.budget()
	}
	return p.cache.Budget()
}

// pagerSummary is a one-line description of a built pager for the load banner.
func pagerSummary(p *expertPager) string {
	if p == nil {
		return ""
	}
	mode := "mmap"
	if p.pool != nil {
		mode = "pread"
	}
	return fmt.Sprintf("expert paging (%s): %d experts, %.1f GB total, %.1f GB budget",
		mode, p.nExperts, float64(p.total)/1e9, float64(p.budget())/1e9)
}
