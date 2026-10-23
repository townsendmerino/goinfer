package decoder

import "context"

// ResidentPrefillRows is the optional capability of a Prefiller that reads a suffix's embeddings a range of rows at a time, so the
// host holds a chunk of them instead of the whole suffix (MEM6, docs/tasks/task-memory-campaign-2026-10.md). PrefillLastRows is
// PrefillLast for n rows at startPos: rows(lo, hi) returns rows [lo, hi) of the suffix, valid only until the next call, and the
// resident may ask for a range more than once (a declined route re-reads it).
type ResidentPrefillRows interface {
	PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error)
}

// embedRowsObserver, when set (tests only, SetEmbedRowsObserverForTest), is told how many rows each embedding of a prefill's
// suffix holds at once.
var embedRowsObserver func(rows int)

// suffixRows is the rows callback for ids: each call embeds rows [lo, hi) into one slab reused across calls, grown to the
// largest range asked for, with one per-layer-input scratch for every row.
func (m *Model) suffixRows(ids []int) func(lo, hi int) [][]float32 {
	n := m.ResidentEmbedLen()
	var slab, pleScratch []float32
	var view [][]float32
	if g4 := m.w.arch.gemma4; g4 != nil && g4.HiddenSizePerLayerInput > 0 {
		pleScratch = make([]float32, m.w.arch.NumLayers*g4.HiddenSizePerLayerInput)
	}
	return func(lo, hi int) [][]float32 {
		k := hi - lo
		if embedRowsObserver != nil {
			embedRowsObserver(k)
		}
		if k > len(view) {
			slab, view = make([]float32, k*n), make([][]float32, k)
			for i := range view {
				view[i] = slab[i*n : (i+1)*n : (i+1)*n]
			}
		}
		for i, id := range ids[lo:hi] {
			m.embedResidentScratch(id, view[i], pleScratch)
		}
		return view[:k]
	}
}
