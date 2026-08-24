package decoder

import (
	"fmt"
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
// Guarded by an internal mutex (audit C-30): the pager lives on *Model and StreamWeights supports
// concurrent decode streams, so its shared LRU cache is locked (SpanCache is not internally locked).
type expertPager struct {
	// mu guards cache, which mmap.SpanCache does NOT lock internally (audit C-30). The pager lives on
	// *Model, shared across every Generate, and Touch mutates the LRU list — so two concurrent streams
	// on a StreamWeights MoE model race without this. All cache mutation/read goes through touch/stats.
	mu       sync.Mutex
	cache    *mmap.SpanCache[unsafe.Pointer]
	nExperts int   // mapping-backed experts under management (for the banner)
	total    int64 // total mapped expert bytes (for the banner)
}

// newExpertPager builds a pager over the experts of an mmap-backed MoE model, or
// returns nil when paging doesn't apply (not MoE, not mmap-backed, or no expert
// weights alias the mapping). budget ≤ 0 selects an automatic budget (~half of
// available RAM); it is clamped to [one expert, total expert bytes].
func newExpertPager(w *Weights, mapping []byte, budget int64) *expertPager {
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
	var total, maxExpert int64
	// addExpert registers one expert under a stable identity (the address of its primary
	// weight struct — the same value the forward touches), collecting only the projections
	// that actually alias the mapping. MappedSpan returns nil for heap-backed (GGUF)
	// weights and the always-on shared expert, so those are silently skipped.
	//
	// MappedSpanRow4 collects the SAME weight's on-disk row4 layout (weightMat kind 4,
	// docs/task-w4a8-neon-bandwidth.md's "Format follow-on") as a SEPARATE span — it is
	// a different byte range in the same file, not the canonical bytes twice. Without
	// this, a kind-4 expert's row4 half would never be registered with the pager: its
	// bytes would go uncounted against the budget (silently under-sizing it) and never
	// be evicted under pressure (paging the wrong half of the tensor, pinning exactly
	// the bytes the M=1 decode kernel actually reads). nil for a kind-3 tensor (no row4
	// layout) or on a non-arm64/non-DotProd build (linalg.WrapInt4Row4 never populates
	// q4Row4 there) — the same "returns nil, silently skipped" shape as MappedSpan.
	addExpert := func(key unsafe.Pointer, wms ...*linalg.WeightMat) {
		var spans [][]byte
		var n int64
		for _, wm := range wms {
			for _, s := range [2][]byte{wm.MappedSpan(base, end), wm.MappedSpanRow4(base, end)} {
				if len(s) == 0 {
					continue
				}
				spans = append(spans, s)
				n += int64(len(s))
			}
		}
		if n == 0 {
			return // heap-backed — nothing to page
		}
		members = append(members, member{key, spans})
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
	return &expertPager{cache: cache, nExperts: len(members), total: total}
}

// touch records that ex is needed now: it becomes most-recently-used and, if it
// wasn't resident, is faulted in (and the LRU tail released to stay within budget).
// A no-op for experts the pager doesn't manage.
func (p *expertPager) touch(key unsafe.Pointer) {
	p.mu.Lock()
	p.cache.Touch(key)
	p.mu.Unlock()
}

// stats returns cumulative (hits, misses, evictions) over all touch calls. A
// non-zero eviction count means the budget was actually enforced (the LRU tail was
// released), as opposed to mere cold-start misses.
func (p *expertPager) stats() (hits, misses, evictions int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.cache.Stats()
}

// pagerSummary is a one-line description of a built pager for the load banner.
func pagerSummary(p *expertPager) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("expert paging: %d experts, %.1f GB total, %.1f GB budget",
		p.nExperts, float64(p.total)/1e9, float64(p.cache.Budget())/1e9)
}
