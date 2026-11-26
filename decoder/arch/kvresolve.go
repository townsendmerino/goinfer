package arch

// KeepsNoKV reports whether no layer of the model keeps attention K/V (a Mamba-only family).
func (a *Architecture) KeepsNoKV() bool {
	for i := 0; i < a.NumLayers; i++ {
		if !a.HasNoAttentionKVAt(i) {
			return false
		}
	}
	return a.NumLayers > 0
}
