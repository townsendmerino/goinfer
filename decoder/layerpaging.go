package decoder

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// layerPager streams a DENSE model's per-layer weights out of the read-only .giw
// mapping (idea #4, docs/ideas-weight-memory.md). Unlike MoE expert paging, the
// transformer layer loop is sequential and fully known in advance, so the pager
// PREFETCHES the upcoming layer (MADV_WILLNEED) while the current one computes —
// overlapping the fault with compute — and RELEASES (MADV_DONTNEED) the layer that
// slides out the back of a window. Resident weight RAM is bounded to ~window layers
// instead of all of them, so a dense model too big for RAM still runs (the floor is
// NVMe bandwidth: a model that doesn't fit is re-read ~once per token). Bit-exact —
// the mapping is read-only and file-backed, so a released layer re-faults from disk
// with identical bytes (TestMadvise_dontneedRefaultsIntact proves the property).
//
// Built only for mmap-backed dense .giw models; nil when the model is MoE (that's
// idea #2's expertPager), heap-backed, or small enough to fit the budget whole.
// Not goroutine-safe — one decode stream drives it, like the KV cache.
type layerPager struct {
	spans  [][][]byte               // [layer][weight] page-aligned spans within the mapping
	window int                      // resident layer cap (the layer `window` behind is released)
	ahead  int                      // prefetch distance (layers ahead to MADV_WILLNEED)
	state  []bool                   // per-layer: currently hinted resident
	advise func([]byte, bool) error // residency hint (real madvise; overridable in tests)

	prefetches, evictions int64
}

// newLayerPager builds a streaming pager over a dense mmap-backed model, or returns
// nil when paging doesn't apply (MoE, not mmap-backed, no mapped layer weights, or
// the budget already holds every layer). budget ≤ 0 selects ~half of available RAM.
func newLayerPager(w *Weights, mmap []byte, budget int64) *layerPager {
	if w.arch.MoE != nil || len(mmap) == 0 {
		return nil
	}
	base := uintptr(unsafe.Pointer(&mmap[0]))
	end := base + uintptr(len(mmap))
	page := os.Getpagesize()

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
			s := alignedMappedSpan(wm, base, end, page)
			if len(s) == 0 {
				continue
			}
			ss = append(ss, s)
			b += int64(len(s))
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
		budget = autoWeightBudget()
	}
	const ahead = 1
	window := int(budget / maxLayer)
	if window < ahead+2 {
		window = ahead + 2 // never evict a layer we just prefetched
	}
	if window >= n {
		return nil // the whole model fits the budget — no streaming needed
	}
	return &layerPager{spans: spans, window: window, ahead: ahead, state: make([]bool, n), advise: madviseBytes}
}

// enterLayer is called at the top of the layer loop before layer l is read. It
// prefetches l and the layer `ahead` of it (so the next fault overlaps this layer's
// compute) and releases the layer `window` behind (keeping resident RAM bounded).
func (p *layerPager) enterLayer(l int) {
	for _, t := range [2]int{l, l + p.ahead} {
		if t >= 0 && t < len(p.spans) && !p.state[t] && len(p.spans[t]) > 0 {
			for _, s := range p.spans[t] {
				_ = p.advise(s, true) // WILLNEED
			}
			p.state[t] = true
			p.prefetches++
		}
	}
	if e := l - p.window; e >= 0 && p.state[e] {
		for _, s := range p.spans[e] {
			_ = p.advise(s, false) // DONTNEED
		}
		p.state[e] = false
		p.evictions++
	}
}

// finishLayers releases every layer still hinted resident at the end of a forward,
// so resident RAM returns to ~zero between tokens and stays bounded by `window`
// during the loop (the cross-token re-read of one window is negligible for a model
// that doesn't fit anyway). Idempotent.
func (p *layerPager) finishLayers() {
	for l := range p.state {
		if p.state[l] {
			for _, s := range p.spans[l] {
				_ = p.advise(s, false)
			}
			p.state[l] = false
			p.evictions++
		}
	}
}

// stats returns cumulative (prefetches, evictions) — both non-zero means streaming
// actually ran (prefetched ahead and released behind).
func (p *layerPager) stats() (prefetches, evictions int64) {
	return p.prefetches, p.evictions
}

// layerPagerSummary describes a built pager for the load banner.
func layerPagerSummary(p *layerPager) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("dense weight streaming: %d layers, window %d resident", len(p.spans), p.window)
}
