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
	kvDim     int      // NumKVHeads * HeadDim
	window    int      // sliding-window cap for local layers; 0 = unbounded
	imgBlocks [][2]int // multimodal: position ranges [start,end) that attend bidirectionally (image blocks); nil = all-causal

	keys [][]float32 // per layer, appended [pos*kvDim]
	vals [][]float32
	pos  int // number of positions stored (the next position index)

	// manualPos decouples pos from Append's last-layer trigger. Gemma 4's last
	// layer is KV-shared (never appends), so the caller advances pos explicitly
	// via Advance() after each token's full layer sweep. qwen3_5_moe sets it too
	// (its linear layers don't Append, so the last-layer trigger is unreliable).
	manualPos bool

	// delta holds the Gated DeltaNet recurrent state for qwen3_5_moe's linear
	// layers (nil entry on softmax layers, nil slice on every other family). This
	// is the recurrent half of the hybrid cache — see docs/qwen3_5_moe.md.
	delta []*deltaState

	scr *decodeScratch // per-stream reusable forward buffers (Model.NewCache sets it)
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
	// pos advances once per last-layer append so all layers stay in lockstep —
	// unless manualPos (gemma4), where the caller calls Advance() after the sweep.
	if !c.manualPos && layer == c.numLayers-1 {
		c.pos++
	}
	return c.pos
}

// Advance bumps the stored-position count by one — the gemma4 forward's explicit
// pos step (its last layer is KV-shared, so Append's auto-advance never fires).
func (c *KVCache) Advance() { c.pos++ }

// Keys / Vals return the stored history for a layer as [storedPos, kvDim].
func (c *KVCache) Keys(layer int) []float32 { return c.keys[layer] }
func (c *KVCache) Vals(layer int) []float32 { return c.vals[layer] }

// Pos is the number of positions stored so far.
func (c *KVCache) Pos() int { return c.pos }

// TruncateTo drops every stored position at index ≥ pos in all layers and resets
// Pos to pos — the rollback speculative decoding needs after a partial accept
// (rejected draft positions were appended but aren't real), and the seam prefix
// reuse rides on to rewind to a shared prompt prefix (see Session). Cheap: a
// reslice that keeps the backing arrays, so re-appending doesn't reallocate. pos
// must be in [0, Pos()].
//
// Per-position stride is derived per layer from what the layer actually holds
// (len/Pos), not the cache's nominal kvDim, because two Gemma 4 facts break a
// uniform stride: per-layer head_dim / KV-head counts make widths differ between
// layers, and KV-shared tail layers Append nothing (length 0). Deriving the
// stride handles both — a shared layer's stride is simply 0, so it stays empty.
func (c *KVCache) TruncateTo(pos int) {
	if pos < 0 || pos > c.pos {
		return
	}
	for l := range c.numLayers {
		perPos := 0
		if c.pos > 0 {
			perPos = len(c.keys[l]) / c.pos // == this layer's width (0 for KV-shared)
		}
		n := pos * perPos
		c.keys[l] = c.keys[l][:n]
		c.vals[l] = c.vals[l][:n]
	}
	c.pos = pos
}

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

// SetImageBlocks marks position ranges [start,end) (e.g. an image's token block)
// as bidirectional for batched-prefill attention: a query inside a block sees the
// whole block, not just the causal prefix. nil/empty ⇒ fully causal (the
// text-only default). Multimodal prefill sets this; see docs/multimodal.md §5a.
func (c *KVCache) SetImageBlocks(blocks [][2]int) { c.imgBlocks = blocks }

// attendHi returns the inclusive upper key bound for a query at pos: pos for a
// causal (text) position, or end-1 if pos lies in a bidirectional image block
// [start,end) — so [start,pos] ∪ (pos,end) = [start,end) becomes visible. With no
// image blocks it always returns pos, so the seam is inert for text-only decode.
func (c *KVCache) attendHi(pos int) int {
	for _, b := range c.imgBlocks {
		if pos >= b[0] && pos < b[1] {
			return b[1] - 1
		}
	}
	return pos
}
