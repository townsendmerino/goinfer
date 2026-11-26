package arch

// HasExperts reports whether layer l carries MoE experts (Gemma 4's 26B carries them in its own block).
func (w *Weights) HasExperts(l int) bool {
	return l < len(w.Layers) && (w.Layers[l].Experts != nil || w.Layers[l].Gemma4MoE != nil)
}

// NormWeights returns the weight and bias an opNorm applies: the bias only when the operation says so.
func (lw *LayerWeights) NormWeights(p normParams) (w, b []float32) {
	switch p.Slot {
	case slotPreAttn:
		w, b = lw.PreAttnNorm, lw.PreAttnNormBias
	case slotPreMLP:
		w, b = lw.PreMLPNorm, lw.PreMLPNormBias
	case slotPostAttn:
		w = lw.PostAttnNorm
	case slotPostMLP:
		w = lw.PostMLPNorm
	}
	if !p.Bias {
		b = nil
	}
	return w, b
}
