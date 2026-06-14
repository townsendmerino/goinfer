package decoder

import (
	"container/list"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// expertPager bounds the resident RAM of a MoE model's expert weights by paging
// them on demand out of the read-only .giw mapping (idea #2,
// docs/ideas-weight-memory.md). A 35B-A3B holds ~32 GB of experts but activates
// only K·L per token; the router's top-k selection is the demand signal. The pager
// keeps an LRU of touched experts up to a byte budget and releases the rest with
// MADV_DONTNEED — safe at any time because the mapping is read-only and file-backed,
// so a released expert merely re-faults from disk (output stays bit-identical to the
// fully-resident run; the only cost is the cold-miss fault, ~+24 ms/token at a 16 GB
// budget on the measured 35B-A3B — see the spike in moepaging_spike_test.go).
//
// Only experts whose quantized weights actually alias the mapping are managed;
// heap-backed weights (a GGUF load) and the always-on shared expert are left alone.
// Not goroutine-safe — one decode stream touches it at a time, like the KV cache.
type expertPager struct {
	budget   int64                            // resident-bytes cap for paged experts
	resident int64                            // bytes currently held resident (our accounting)
	ranges   map[*expertWeights][][]byte      // page-aligned spans of the mapping, per expert
	bytes    map[*expertWeights]int64         // resident bytes per expert (Σ aligned span lens)
	lru      *list.List                       // *expertWeights, most-recently-used at front
	pos      map[*expertWeights]*list.Element // membership + O(1) promotion
	advise   func([]byte, bool) error         // residency hint (real madvise; overridable in tests)

	hits, misses int64
}

// newExpertPager builds a pager over the experts of an mmap-backed MoE model, or
// returns nil when paging doesn't apply (not MoE, not mmap-backed, or no expert
// weights alias the mapping). budget ≤ 0 selects an automatic budget (~half of
// available RAM); it is clamped to [one expert, total expert bytes].
func newExpertPager(w *Weights, mmap []byte, budget int64) *expertPager {
	if w.arch.MoE == nil || len(mmap) == 0 {
		return nil
	}
	base := uintptr(unsafe.Pointer(&mmap[0]))
	end := base + uintptr(len(mmap))
	page := os.Getpagesize()

	p := &expertPager{
		ranges: map[*expertWeights][][]byte{},
		bytes:  map[*expertWeights]int64{},
		lru:    list.New(),
		pos:    map[*expertWeights]*list.Element{},
		advise: madviseBytes,
	}
	var total int64
	maxExpert := int64(0)
	for li := range w.Layers {
		exps := w.Layers[li].Experts
		for ei := range exps {
			ex := &exps[ei]
			var spans [][]byte
			var n int64
			for _, wm := range [...]*linalg.WeightMat{&ex.Gate, &ex.Up, &ex.Down} {
				s := alignedMappedSpan(wm, base, end, page)
				if len(s) == 0 {
					continue
				}
				spans = append(spans, s)
				n += int64(len(s))
			}
			if n == 0 {
				continue // heap-backed (e.g. GGUF) — nothing to page
			}
			p.ranges[ex] = spans
			p.bytes[ex] = n
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
		budget = autoWeightBudget()
	}
	if budget > total {
		budget = total // never budget more than exists
	}
	if budget < maxExpert {
		budget = maxExpert // must hold at least one expert
	}
	p.budget = budget
	return p
}

// touch records that ex is needed now: it becomes most-recently-used and, if it
// wasn't resident, is faulted in (and the LRU tail released to stay within budget).
// A no-op for experts the pager doesn't manage.
func (p *expertPager) touch(ex *expertWeights) {
	spans, managed := p.ranges[ex]
	if !managed {
		return
	}
	if el, ok := p.pos[ex]; ok {
		p.lru.MoveToFront(el)
		p.hits++
		return
	}
	p.misses++
	for _, s := range spans {
		_ = p.advise(s, true) // WILLNEED: hint the fault we're about to take
	}
	p.pos[ex] = p.lru.PushFront(ex)
	p.resident += p.bytes[ex]
	for p.resident > p.budget && p.lru.Len() > 1 {
		back := p.lru.Back()
		victim := back.Value.(*expertWeights)
		p.lru.Remove(back)
		delete(p.pos, victim)
		p.resident -= p.bytes[victim]
		for _, s := range p.ranges[victim] {
			_ = p.advise(s, false) // DONTNEED: release the victim's pages
		}
	}
}

// stats returns cumulative (hits, misses) over all touch calls.
func (p *expertPager) stats() (hits, misses int64) { return p.hits, p.misses }

// alignedMappedSpan returns the page-aligned interior of wm's quantized backing
// bytes, but only if they lie inside the [base,end) mapping (so heap-backed f32 /
// scales are skipped). The aligned span is what MADV_DONTNEED can release without
// touching a neighbor's page; the few boundary pages it omits are negligible
// against a multi-MB expert.
func alignedMappedSpan(wm *linalg.WeightMat, base, end uintptr, page int) []byte {
	var raw []byte
	if q8, _, _, ok := wm.Int8(); ok && len(q8) > 0 {
		raw = unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8))
	} else if q4, _, _, ok := wm.Int4(); ok && len(q4) > 0 {
		raw = q4
	}
	if len(raw) == 0 {
		return nil
	}
	start := uintptr(unsafe.Pointer(&raw[0]))
	if start < base || start+uintptr(len(raw)) > end {
		return nil // heap-backed, not part of the mapping
	}
	pg := uintptr(page)
	as := (start + pg - 1) &^ (pg - 1) // round up to page
	ae := (start + uintptr(len(raw))) &^ (pg - 1)
	if ae <= as {
		return nil
	}
	off := int(as - start)
	return raw[off : off+int(ae-as)]
}

// autoWeightBudget picks a default expert-cache budget: ~half of available RAM,
// falling back to 8 GiB when that can't be read (non-Linux, or no MemAvailable).
func autoWeightBudget() int64 {
	const fallback = 8 << 30
	if avail := availableRAMBytes(); avail > 0 {
		return avail / 2
	}
	return fallback
}

// availableRAMBytes reads MemAvailable from /proc/meminfo (Linux); 0 elsewhere.
func availableRAMBytes() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		f := strings.Fields(line) // "MemAvailable:  12345678 kB"
		if len(f) >= 2 {
			if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
				return kb * 1024
			}
		}
	}
	return 0
}

// pagerSummary is a one-line description of a built pager for the load banner.
func pagerSummary(p *expertPager) string {
	if p == nil {
		return ""
	}
	var total int64
	for _, b := range p.bytes {
		total += b
	}
	return fmt.Sprintf("expert paging: %d experts, %.1f GB total, %.1f GB budget",
		len(p.bytes), float64(total)/1e9, float64(p.budget)/1e9)
}
