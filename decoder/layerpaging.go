package decoder

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/mmap"
)

// layerPager streams a dense model's per-layer weights out of the read-only .giw mapping (docs/ideas-weight-memory.md,
// idea #4). The layer loop is sequential and known in advance, so the pager prefetches the upcoming layer (Advise
// WILLNEED) while the current one computes and releases (Advise DONTNEED) the layer that slides out the back of a
// window. It is a windowed prefetch, not an LRU, so it does not use aikit/mmap.SpanCache; it borrows the span
// alignment (WeightMat.MappedSpan), the residency hint (mmap.Advise) and the RAM budget (mmap.AutoBudget). Resident
// weight RAM is bounded to about window layers, so a dense model too big for RAM still runs; the floor is NVMe
// bandwidth, since a model that does not fit is re-read about once per token. Bit-exact: the mapping is read-only and
// file-backed, so a released layer re-faults with identical bytes (aikit's TestMadvise_dontneedRefaultsIntact).
//
// The resident floor is not zero: only the 7 per-layer projections stream. The token embedding, final norm and LM head
// stay resident, plus the live window, so "bigger than RAM" means bounded to floor + window.
//
// Built only for mmap-backed dense .giw models; nil for MoE (expertPager), heap-backed, or a model small enough to fit
// the budget whole. An internal mutex guards the paging state: the pager lives on *Model and StreamWeights supports
// concurrent decode streams.
type layerPager struct {
	spans  [][][]byte // [layer][weight] page-aligned spans within the mapping
	window int        // resident layer cap (the layer `window` behind is released)
	ahead  int        // prefetch distance (layers ahead to Advise WILLNEED)

	// mu guards the mutable paging state below. The pager lives on *Model, shared across every Generate, and StreamWeights
	// supports concurrent streams that each drive the layer loop, so two streams would race on state[]/counters. The
	// mapping and its WILLNEED/DONTNEED hints are model-level, so one guarded view is correct; the hints are advisory, so
	// a stream may re-fault a page another released, and only the state writes need the lock.
	mu    sync.Mutex
	state []bool // per-layer: currently hinted resident

	prefetches, evictions int64
}

// newLayerPager builds a streaming pager over a dense mmap-backed model, or returns
// nil when paging doesn't apply (MoE, not mmap-backed, no mapped layer weights, or
// the budget already holds every layer). budget ≤ 0 selects ~half of available RAM.
func newLayerPager(w *Weights, mapping []byte, budget int64) *layerPager {
	if w.arch.MoE != nil || len(mapping) == 0 {
		return nil
	}
	// Own-forward families run their own layer loop, which never calls enterLayer, so a pager would print a RAM-bound
	// banner it cannot deliver. Only the generic dense forward pages. Derived from the dispatch table (ownForward).
	if _, own := w.arch.ownForward(); own {
		return nil
	}
	base := uintptr(unsafe.Pointer(&mapping[0]))
	end := base + uintptr(len(mapping))

	n := len(w.Layers)
	spans := make([][][]byte, n)
	var maxLayer int64
	for l := range w.Layers {
		lw := &w.Layers[l]
		var ss [][]byte
		var b int64
		for _, wm := range [...]*linalg.WeightMat{
			&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj,
			&lw.GateProj, &lw.UpProj, &lw.DownProj,
		} {
			// Register only the span the M=1 decode kernel will read: row4 when present, canonical otherwise, never both.
			// SpanCache.Touch WILLNEEDs every span under a key unconditionally, so registering both prefetches the unread copy too,
			// a fixed ~2x I/O tax per miss (moepaging.go's addExpert follows the same rule).
			s := wm.MappedSpanRow4(base, end)
			if len(s) == 0 {
				s = wm.MappedSpan(base, end)
			}
			if len(s) > 0 {
				ss = append(ss, s)
				b += int64(len(s))
			}
		}
		spans[l] = ss
		if b > maxLayer {
			maxLayer = b
		}
	}
	if maxLayer == 0 {
		return nil // heap-backed (e.g. GGUF) — nothing mapped to page
	}
	if budget <= 0 {
		budget = mmap.AutoBudget()
	}
	const ahead = 1
	// enterLayer prefetches l+ahead before releasing l-window, so the resident set at steady state is window+ahead layers,
	// not window: subtract ahead here so window+ahead layers' bytes fit the budget.
	window := max(int(budget/maxLayer)-ahead,
		// never evict a layer we just prefetched
		ahead+2)
	if window >= n {
		return nil // the whole model fits the budget — no streaming needed
	}
	return &layerPager{spans: spans, window: window, ahead: ahead, state: make([]bool, n)}
}

// enterLayer is called at the top of the layer loop before layer l is read. It
// prefetches l and the layer `ahead` of it (so the next fault overlaps this layer's
// compute) and releases the layer `window` behind (keeping resident RAM bounded).
func (p *layerPager) enterLayer(l int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, t := range [2]int{l, l + p.ahead} {
		if t >= 0 && t < len(p.spans) && !p.state[t] && len(p.spans[t]) > 0 {
			for _, s := range p.spans[t] {
				_ = mmap.Advise(s, true) // WILLNEED
			}
			p.state[t] = true
			p.prefetches++
		}
	}
	if e := l - p.window; e >= 0 && p.state[e] {
		for _, s := range p.spans[e] {
			_ = mmap.Advise(s, false) // DONTNEED
		}
		p.state[e] = false
		p.evictions++
	}
}

// finishLayers releases every layer still hinted resident at the end of a forward,
// so resident RAM returns to ~zero between tokens and stays bounded by `window+ahead`
// during the loop (the cross-token re-read of one window is negligible for a model
// that doesn't fit anyway). Idempotent.
func (p *layerPager) finishLayers() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for l := range p.state {
		if p.state[l] {
			for _, s := range p.spans[l] {
				_ = mmap.Advise(s, false)
			}
			p.state[l] = false
			p.evictions++
		}
	}
}

// stats returns cumulative (prefetches, evictions) — both non-zero means streaming
// actually ran (prefetched ahead and released behind).
func (p *layerPager) stats() (prefetches, evictions int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.prefetches, p.evictions
}

// layerPagerSummary describes a built pager for the load banner.
func layerPagerSummary(p *layerPager) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("dense weight streaming: %d layers, window %d resident", len(p.spans), p.window)
}
