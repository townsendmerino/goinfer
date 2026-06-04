package decoder

// KVCache holds the per-layer key/value history for one generation
// sequence. The cache, not a growing per-call buffer, is the decoder's
// memory model: each decode step appends one position's K and V per layer
// and attends over everything stored so far (bounded by the sliding window
// on local layers).
//
// Layout per layer: keys/values are [pos, NumKVHeads*HeadDim] row-major,
// appended in position order. GQA means KV heads (NumKVHeads) are fewer
// than query heads; attention.go broadcasts each KV head across its group.
//
// Not goroutine-safe: one cache belongs to one in-flight sequence.
type KVCache struct {
	numLayers int
	kvDim     int // NumKVHeads * HeadDim
	window    int // sliding-window cap for local layers; 0 = unbounded

	keys [][]float32 // per layer, appended [pos*kvDim]
	vals [][]float32
	pos  int // number of positions stored (the next position index)
}

// NewKVCache allocates an empty cache for a model with the given geometry.
// capHint pre-sizes the per-layer slices to avoid reallocation during a
// known-length generation; 0 is fine (grow on demand).
func NewKVCache(numLayers, numKVHeads, headDim, window, capHint int) *KVCache {
	kvDim := numKVHeads * headDim
	c := &KVCache{
		numLayers: numLayers,
		kvDim:     kvDim,
		window:    window,
		keys:      make([][]float32, numLayers),
		vals:      make([][]float32, numLayers),
	}
	for l := range numLayers {
		c.keys[l] = make([]float32, 0, capHint*kvDim)
		c.vals[l] = make([]float32, 0, capHint*kvDim)
	}
	return c
}

// Append stores one position's K and V for the given layer. k and v must
// each be kvDim long. Returns the position index just written.
func (c *KVCache) Append(layer int, k, v []float32) int {
	c.keys[layer] = append(c.keys[layer], k...)
	c.vals[layer] = append(c.vals[layer], v...)
	// pos advances once per layer-0 append so all layers stay in lockstep.
	if layer == c.numLayers-1 {
		c.pos++
	}
	return c.pos
}

// Keys / Vals return the stored history for a layer as [storedPos, kvDim].
func (c *KVCache) Keys(layer int) []float32 { return c.keys[layer] }
func (c *KVCache) Vals(layer int) []float32 { return c.vals[layer] }

// Pos is the number of positions stored so far.
func (c *KVCache) Pos() int { return c.pos }

// WindowStart returns the first key index a query at absolute position pos
// may attend to on a local (sliding-window) layer. Gemma's window of size W
// admits keys j with pos−W < j ≤ pos, i.e. the W most recent keys
// [pos−W+1, pos] — matching HF's sliding mask (key j attends iff pos−j < W).
// Global layers (and an unset window) attend from 0.
//
// Takes pos explicitly rather than reading c.pos: within one forward, c.pos
// only advances on the last layer's Append, so reading it post-Append would
// shift the window by one on that layer alone. The query position is stable.
func (c *KVCache) WindowStart(pos int, global bool) int {
	if global || c.window <= 0 {
		return 0
	}
	start := max(pos-c.window+1, 0)
	return start
}
