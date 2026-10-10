package decoder

// The hidden-state capture seam, extracted so the families with their own runLayers can offer it without each
// reimplementing the copy-on-match loop. It is one definition, called at the tail of each loop body, because separate
// copies drift apart: the generic path (runLayersFromEmbed) copies AFTER the MLP add, and a copy placed a few lines
// earlier would capture a residual that is off by one sublayer while still looking plausible in every test that only
// checks shape. The block drafters (P10) need these residuals from families that never route through the generic path.
//
// THE CONTRACT, stated once because the drafters depend on it: captureResidual(l, h) is called
// with the residual stream AFTER layer l is complete — the same tensor the generic path copies,
// and the same convention HF's `output_hidden_states` uses shifted by one (HF's index 0 is the
// embedding; ours is layer 0's OUTPUT). `layers` indices are 0-based into [0, NumLayers).
// The copy is a copy, never a reference: h is mutated by every later layer.
//
// This does NOT make capture free of the arch guard in ForwardCapture — a family is wired only
// once its runLayers actually calls this AND the guard stops rejecting it. The two are separate
// on purpose, so a half-wired family fails loudly at the guard rather than silently returning
// nil rows.
func (c *KVCache) captureResidual(l int, h []float32) {
	if c == nil || c.captureLayers == nil {
		return
	}
	for i, cl := range c.captureLayers {
		if cl == l {
			c.captured[i] = append(c.captured[i][:0], h...)
		}
	}
}
