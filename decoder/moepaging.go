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

// expertPager bounds the resident RAM of a MoE model's expert weights by paging
// them on demand out of the read-only .giw mapping (idea #2,
// docs/ideas-weight-memory.md). A 35B-A3B holds ~32 GB of experts but activates
// only K·L per token; the router's top-k selection is the demand signal. The
// generic span-residency pager (touch → fault-in WILLNEED, evict over budget
// DONTNEED) now lives in aikit/mmap.SpanCache; this pager runs it with the
// frequency-aware EvictLeastRecent policy (newExpertPager). This type holds only the
// MoE-specific half: which experts alias the mapping and the touch hook the
// router calls: moeMLP (mlp.go) for [NumExperts]expertWeights families, and gemma4MoEFFN
// (forward_gemma4_moe.go) for gemma4's fused gate‖up + down experts. Releasing is lossless — the mapping is read-only
// and file-backed, so an evicted expert merely re-faults from disk (output stays
// bit-identical; the only cost is the cold-miss fault, ~+24 ms/token at a 16 GB
// budget on the measured 35B-A3B — moepaging_spike_test.go).
//
// Only experts whose quantized weights actually alias the mapping are managed;
// heap-backed weights (a GGUF load) and the always-on shared expert are left alone.
//
// Two backing modes, chosen once at build time (newExpertPager):
//   - mmap+madvise (default): cache aliases the read-only mapping directly, WILLNEED faults it
//     in, DONTNEED releases it. Zero-copy, but on darwin DONTNEED is a no-op (madvise_darwin.go)
//     -- there is no real RAM cap on macOS with this mode.
//   - owned-buffer pread (opt-in, GOINFER_MOE_PREAD_CPU=1; Lever 1b, task-moe-streaming.md):
//     pool holds a fixed set of owned buffers it fills via pread, giving a firm cap on every
//     platform at the cost of a memcpy per miss and losing .giw zero-copy aliasing.
//
// Guarded by two internal mutexes (audit C-30 plus Lever 1b's own cross-call requirement — see
// their doc comments below): the pager lives on *Model and StreamWeights supports concurrent
// decode streams, so its shared LRU state is locked (SpanCache is not internally locked, and
// pool needs its own protection for a different reason).
type expertPager struct {
	// mu guards touch()'s own mutation (cache.Touch or pool.ensure) — held internally, for the
	// duration of that one call, regardless of mode. This is the ORIGINAL C-30 contract
	// (TestExpertPager_concurrentNoRace calls touch() directly, concurrently, with no outer
	// locking of its own, and must remain safe doing so): touch() is always self-contained-safe
	// to call from any goroutine. It does NOT by itself protect a caller's reads of the
	// WeightMat fields AFTER touch() returns — see readMu.
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

	// S5 (task-never-swap-2026-09.md): process-wide getrusage(RUSAGE_SELF) minor/major
	// page-fault counts at pager creation — faultDelta() reports how many faults have
	// happened SINCE, the comparison this brief's own A/B (mmap mode's real WILLNEED faults
	// vs pool mode's pread, which should show near-zero additional faults) needs. Process-wide
	// on purpose, not pager-scoped: getrusage has no way to attribute a fault to a specific
	// mapping, so this is contaminated by whatever else the process does between the baseline
	// and the read — acceptable for an A/B run with nothing else happening, stated rather than
	// hidden as a per-fault-attributed number it is not. faultsOK false means the platform has
	// no probe (faultcount_other.go) — every unknown proceeds; pagerSummary omits the term.
	minfltBase, majfltBase int64
	faultsOK               bool
}

// Lock/Unlock should wrap touch() and the matmul reads that follow it, for the duration of one
// MoE FFN call (see readMu's doc comment). Required for correctness in pool mode; a no-op-ish
// safety margin in mmap mode, where touch() is already self-contained-safe on its own.
func (p *expertPager) Lock()   { p.readMu.Lock() }
func (p *expertPager) Unlock() { p.readMu.Unlock() }

// newExpertPager builds a pager over the experts of an mmap-backed MoE model, or
// returns nil when paging doesn't apply (not MoE, not mmap-backed, or no expert
// weights alias the mapping). budget ≤ 0 selects an automatic budget (~half of
// available RAM); it is clamped to [one expert, total expert bytes]. giwPath is the
// .giw file the mapping was built from (Model.GiwPath) -- only consulted when the
// owned-buffer pread mode is requested (GOINFER_MOE_PREAD_CPU=1), to open an
// independent fd for pread (the mmap's own fd is closed right after mapping).
// MoEPagerDefault is S5's registered default (task-never-swap-2026-09.md): darwin, where
// MADV_DONTNEED is a no-op and mmap mode therefore cannot enforce its budget, gets the owned-buffer
// pool; every other platform keeps mmap mode, where DONTNEED works and the alias is free. The ONE
// source for it: serve's --moe-pager default and a library Load with Options.MoEPager unset both
// resolve here (they used to disagree on darwin — serve set an env var, a library caller got mmap).
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
	// addExpert registers one expert under a stable identity (the address of its primary
	// weight struct — the same value the forward touches), collecting only the projections
	// that actually alias the mapping. MappedSpan returns nil for heap-backed (GGUF)
	// weights and the always-on shared expert, so those are silently skipped.
	//
	// A kind-4 tensor carries TWO on-disk representations (canonical + row4,
	// docs/completed/task-w4a8-neon-bandwidth.md's "Format follow-on"), but the M==1 decode kernel
	// (MatmulBTW4A8Into) reads ONLY row4 whenever it's present — this arch's forward is
	// always M==1, decode and prefill alike (confirmed by "prefill path: sequential" on
	// every load). Registering both spans under one cache key was a real, measured bug
	// (docs/completed/task-zeno-compare.md's "At-scale acceptance run"): SpanCache.Touch issues
	// MADV_WILLNEED on EVERY span under a key, unconditionally, so a cold kind-4 touch
	// prefetched both copies from disk though only one was ever read — a fixed ~2x I/O
	// tax per miss that produced a ~25-30% throughput regression instead of the row4
	// kernel's proven gain. Fix: register only the span that will actually be read —
	// row4 when present, canonical otherwise. Never both.
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
	// Frequency-aware (classic LRU tail) eviction: the router's demand signal is
	// skewed FREQUENCY, not a scan — the hottest ~10% of experts absorb ~72% of the
	// top-k picks — so the hot set must stay resident. SpanCache's default is
	// scan-resistant (evict-most-recent), which is right for the ANN cyclic scan but
	// evicts exactly the hot experts here (measured −51 pp hit rate at a 4 GB budget on
	// a real 35B-A3B). EvictLeastRecent restores it. See aikit mmap.EvictPolicy.
	cache := mmap.NewSpanCacheWithPolicy[unsafe.Pointer](budget, mmap.EvictLeastRecent)
	for _, m := range members {
		cache.Add(m.key, m.spans)
	}
	return &expertPager{cache: cache, nExperts: len(members), total: total,
		minfltBase: minfltBase, majfltBase: majfltBase, faultsOK: faultsOK}
}

// touch records that ex is needed now: it becomes most-recently-used and, if it
// wasn't resident, is faulted in (mmap mode, evicting the LRU tail to stay within
// budget) or refilled into an owned pread slot (pool mode). A no-op for experts the
// pager doesn't manage. Always self-contained-safe to call directly and concurrently
// (mu, held internally, for the ORIGINAL C-30 guarantee — see expertPager's doc
// comment) — but in pool mode, the caller must ADDITIONALLY hold Lock() across this
// call and its subsequent reads of the repointed WeightMat fields, or a concurrent
// touch() can repoint them mid-read (readMu's doc comment).
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

// advisedBytes returns cumulative bytes fetched from disk over every miss — WILLNEED-hinted
// bytes in mmap mode, pread'd bytes in pool mode — independent of whatever else the machine's
// disk is doing. A durable, contamination-proof I/O check: a member registering redundant spans
// (the kind-4 double-WILLNEED bug this exists to catch, docs/completed/task-zeno-compare.md's
// "At-scale acceptance run") shows up here directly as bytes-per-miss exceeding the expected
// per-expert working set, immune to whatever an external tool like iostat would also be
// counting on a shared machine.
func (p *expertPager) advisedBytes() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pool != nil {
		return p.pool.bytesRead
	}
	return p.cache.AdvisedBytes()
}

// faultDelta reports this PROCESS's minor/major page faults since the pager was created — the
// mmap-vs-pool A/B this brief exists to run (mmap mode's WILLNEED-triggered real page faults vs
// pool mode's pread, which should show near-zero additional faults past the baseline). See the
// struct field's own doc comment for why this is process-wide, not pager-attributed.
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

// budget is the resident-bytes cap this pager was built with, in either mode — split out of
// pagerSummary so S4 item 5's working-set prediction (moeworkingset.go) can ask the same question
// pagerSummary already answers, without a second copy of the mode dispatch.
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
