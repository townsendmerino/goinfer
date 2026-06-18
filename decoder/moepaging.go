package decoder

import (
	"fmt"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
)

// expertPager bounds the resident RAM of a MoE model's expert weights by paging
// them on demand out of the read-only .giw mapping (idea #2,
// docs/ideas-weight-memory.md). A 35B-A3B holds ~32 GB of experts but activates
// only K·L per token; the router's top-k selection is the demand signal. The
// generic span-residency LRU (touch → fault-in WILLNEED, release the budget tail
// DONTNEED) now lives in aikit/mmap.SpanCache; this type holds only the
// MoE-specific half: which experts alias the mapping and the touch hook the
// router calls (moeMLP, mlp.go). Releasing is lossless — the mapping is read-only
// and file-backed, so an evicted expert merely re-faults from disk (output stays
// bit-identical; the only cost is the cold-miss fault, ~+24 ms/token at a 16 GB
// budget on the measured 35B-A3B — moepaging_spike_test.go).
//
// Only experts whose quantized weights actually alias the mapping are managed;
// heap-backed weights (a GGUF load) and the always-on shared expert are left alone.
// Not goroutine-safe — one decode stream touches it at a time, like the KV cache.
type expertPager struct {
	cache    *mmap.SpanCache[*expertWeights]
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
		ex    *expertWeights
		spans [][]byte
	}
	var members []member
	var total, maxExpert int64
	for li := range w.Layers {
		exps := w.Layers[li].Experts
		for ei := range exps {
			ex := &exps[ei]
			var spans [][]byte
			var n int64
			// Only the quantized projections alias the mapping; MappedSpan returns
			// nil for heap-backed (GGUF) weights and the always-on shared expert.
			for _, wm := range [...]*linalg.WeightMat{&ex.Gate, &ex.Up, &ex.Down} {
				s := wm.MappedSpan(base, end)
				if len(s) == 0 {
					continue
				}
				spans = append(spans, s)
				n += int64(len(s))
			}
			if n == 0 {
				continue // heap-backed — nothing to page
			}
			members = append(members, member{ex, spans})
			total += n
			if n > maxExpert {
				maxExpert = n
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
	cache := mmap.NewSpanCache[*expertWeights](budget)
	for _, m := range members {
		cache.Add(m.ex, m.spans)
	}
	return &expertPager{cache: cache, nExperts: len(members), total: total}
}

// touch records that ex is needed now: it becomes most-recently-used and, if it
// wasn't resident, is faulted in (and the LRU tail released to stay within budget).
// A no-op for experts the pager doesn't manage.
func (p *expertPager) touch(ex *expertWeights) { p.cache.Touch(ex) }

// stats returns cumulative (hits, misses, evictions) over all touch calls. A
// non-zero eviction count means the budget was actually enforced (the LRU tail was
// released), as opposed to mere cold-start misses.
func (p *expertPager) stats() (hits, misses, evictions int64) { return p.cache.Stats() }

// pagerSummary is a one-line description of a built pager for the load banner.
func pagerSummary(p *expertPager) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("expert paging: %d experts, %.1f GB total, %.1f GB budget",
		p.nExperts, float64(p.total)/1e9, float64(p.cache.Budget())/1e9)
}
