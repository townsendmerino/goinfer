package decoder

// embedRowsObserver, when set (tests only, SetEmbedRowsObserverForTest), is told how many rows each embedding of a prefill's
// suffix holds at once.
var embedRowsObserver func(rows int)

// suffixRows is the rows callback for ids: each call embeds rows [lo, hi) into one slab reused across calls, grown to the
// largest range asked for, with one per-layer-input scratch for every row.
func (m *Model) suffixRows(ids []int) func(lo, hi int) [][]float32 {
	n := m.BackendView().EmbedLen()
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
