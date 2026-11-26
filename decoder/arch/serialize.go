package arch

// HasPopulatedLayers reports whether w's body matmul weights actually hold data, as opposed to a freshly make()'d Layers
// slice whose elements are still zero-valued (the streaming GGUF transcode's state at header-write time, before any layer
// has streamed in). Check it before trusting quantLabel(): its "nothing matched" case returns "native", a real quant mode,
// not an empty string, so an unpopulated w would silently produce a false "native" label.
func (w *Weights) HasPopulatedLayers() bool {
	for _, m := range w.BodyMatmulWeights() {
		if m.Rows() > 0 {
			return true
		}
	}
	return false
}

// QuantLabel names the precision of the resident matmul weights for display and the KV-snapshot fingerprint, accounting
// for mixed bundles. It scans the body matmuls (the per-layer attention/FFN projections, experts and routers: exactly
// what `-quant int4|int4mix|int8int8` selects and what the batched-prefill gate inspects) and returns "int4mix" only
// when int4 coexists with an int8 body weight, which is what `-quant int4mix` produces. Pure bundles collapse to int4 /
// int8int8 / int8 / native.
//
// A float32 body weight does not make a mix. No quant mode chooses one: a MoE's routers stay float32 at every quant, so
// counting them would label every int4 MoE "int4mix" and refuse an explicit `-quant int4` against its own sidecar.
//
// The token embedding and LM head are excluded. int4 mode pins them to int8 by default (logit-critical; the EmbedInt4
// knob relaxes them), so their precision is orthogonal to the int4-vs-int4mix distinction: a plain `-quant int4` bundle
// keeps an int8 head. Including them would mislabel such a bundle "int4mix" even though every projection is int4, and the
// label would contradict the path the prefill gate takes. The .giw header's quant field derives from the first weight
// (the int8 embed), so it is not a substitute.
func (w *Weights) QuantLabel() string {
	var hasInt4, hasInt8I8, hasInt8, hasQ4K bool
	// bodyMatmulWeights excludes the int8-pinned logit tables — the single definition of that
	// exclusion (weights.go), so the label and the .giw resolved-quant field can never disagree.
	for _, m := range w.BodyMatmulWeights() {
		if m.Rows() == 0 {
			continue
		}
		switch m.Kind() {
		case "q4k", "q5k", "q6k":
			hasQ4K = true
		case "int4":
			hasInt4 = true
		case "int8":
			if _, _, w8a8, _ := m.Int8(); w8a8 {
				hasInt8I8 = true
			} else {
				hasInt8 = true
			}
		}
	}
	switch {
	case hasQ4K:
		return "q4k" // its other tensors are the file's Q5_K/Q6_K blocks or int8 W8A8 by construction (matmulQuant, q4kNative56)
	case hasInt4 && (hasInt8I8 || hasInt8):
		return "int4mix"
	case hasInt4:
		return "int4"
	case hasInt8I8:
		return "int8int8"
	case hasInt8:
		return "int8"
	default:
		return "native"
	}
}
