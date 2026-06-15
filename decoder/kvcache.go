package decoder

import "github.com/townsendmerino/aikit/linalg"

// kvQuant selects the CPU KV-cache storage precision. kvF32 (default) is
// bit-exact; kvI8 stores K and V as per-(position,KV-head) symmetric int8 + a
// f32 scale each — 4× smaller, and decode reads them with the SDOT integer dot
// (DotI8) instead of scalar f64. Opt-in (lossy); see docs/task-cpu-kv-quant.md.
type kvQuant uint8

const (
	kvqF32 kvQuant = iota
	kvI8
)

// quantizeHeads writes src (nKV contiguous rows of headDim) as int8 into q with
// one symmetric maxabs/127 scale per head into scales[:nKV]. Per-head (not
// per-kvDim-row) scales bound RoPE'd-key outlier channels — the cheap KIVI-style
// win. headDim is 64/128, a clean multiple of the SDOT/AVX2 width.
func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {
	for h := 0; h < nKV; h++ {
		o := h * headDim
		scales[h] = linalg.QuantizeRowInt8(src[o:o+headDim], q[o:o+headDim])
	}
}

// dequantHeads reconstructs nKV int8 head-rows (each headDim wide, scale per
// head) back to f32 in dst — the inverse of quantizeHeads, for the batched
// prefill's dequant-into-scratch (the f32 matmul kernels stay unchanged).
func dequantHeads(q []int8, scales []float32, nKV, headDim int, dst []float32) {
	for h := 0; h < nKV; h++ {
		o, s := h*headDim, scales[h]
		for c := 0; c < headDim; c++ {
			dst[o+c] = float32(q[o+c]) * s
		}
	}
}

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
	mropePos  [][3]int // Qwen2.5-VL m-RoPE: per-sequence-position (t,h,w) rotary positions; nil = scalar RoPE. Indexed by absolute sequence position. (P5)

	keys [][]float32 // per layer, appended [pos*kvDim] (global / append-forever layers)
	vals [][]float32
	pos  int // number of positions stored (the next position index)

	// rings holds a fixed-W ring buffer for each sliding-window (local) layer;
	// nil entry = a global layer that append-forevers into keys/vals above. Set by
	// enableRings (from NewCache) for the families that take the attendQuery /
	// attendBatchedHeads paths — full-attention families (no local layers),
	// gemma4, and qwen3_5_moe keep append-forever (rings all nil). A local layer
	// stores only the W most recent positions (the only ones any future query can
	// read), so its KV is O(W) not O(context). See docs/task-kv-ring-eviction.md.
	rings    []*ring
	localAny bool // any ring layer present (gates prefill's assembly scratch)

	// int8 KV (quant == kvI8): global (append-forever) layers store K/V as int8
	// here, parallel to keys/vals, with one symmetric scale per (position,KV-head).
	// Ring layers carry their own int8 fields. kvqF32 leaves all of this nil.
	quant              kvQuant
	headDim            int         // per-head width, for int8 scale striding
	keysQ, valsQ       [][]int8    // [pos*kvDim] (global int8 layers)
	keyScale, valScale [][]float32 // [pos*nKV]   (per-(position,head) scale)

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

	// lora is the active compute-time LoRA adapter for this stream (#7), nil for
	// the merged/base default. Set per-stream (Session.UseAdapter); the forward
	// adds each layer's low-rank delta after the base projection. A non-nil lora
	// forces the sequential prefill path (the batched forwardN is not wired).
	lora *loraRuntime
}

// NewKVCache allocates an empty cache for a model with the given geometry.
// capHint pre-sizes the per-layer slices to avoid reallocation during a
// known-length generation; 0 is fine (grow on demand).
func NewKVCache(numLayers, numKVHeads, headDim, window, capHint int) *KVCache {
	kvDim := numKVHeads * headDim
	c := &KVCache{
		numLayers: numLayers,
		kvDim:     kvDim,
		headDim:   headDim,
		window:    window,
		keys:      make([][]float32, numLayers),
		vals:      make([][]float32, numLayers),
	}
	for l := range numLayers {
		c.keys[l] = make([]float32, 0, capHint*kvDim)
		c.vals[l] = make([]float32, 0, capHint*kvDim)
	}
	c.rings = make([]*ring, numLayers) // all nil: append-forever until enableRings
	return c
}

// ring is the sliding-window KV store for one local layer: it physically holds
// at most w positions, the slot for absolute position p being p%w. The per-layer
// stride (KV width = NumKVHeads*HeadDim for this layer) is learned on the first
// write, so per-layer-width families (Gemma 4) need no special-casing. count is
// the logical length — total positions ever written; the live window (everything
// a future query can read) is [max(0,count-w), count).
type ring struct {
	k, v     []float32 // f32 mode: [w*stride] each; position p at offset (p%w)*stride
	kq, vq   []int8    // int8 mode: [w*stride]
	ksc, vsc []float32 // int8 mode: per-(slot,head) scale, [w*nKV]
	w        int       // window size (slot count)
	stride   int       // per-layer KV width; 0 until the first write sizes it
	count    int       // logical positions written (the absolute next position)
	quant    kvQuant   // kvF32 | kvI8 (set by enableRings)
	headDim  int       // for int8 scale striding (nKV = stride/headDim)
}

// write stores position p's k/v into slot p%w and advances count past p. O(stride),
// no growth or realloc after the first write. Quantizes per head when kvI8.
func (r *ring) write(p int, k, v []float32) {
	if r.stride == 0 {
		r.stride = len(k)
		if r.quant == kvI8 {
			nKV := r.stride / r.headDim
			r.kq = make([]int8, r.w*r.stride)
			r.vq = make([]int8, r.w*r.stride)
			r.ksc = make([]float32, r.w*nKV)
			r.vsc = make([]float32, r.w*nKV)
		} else {
			r.k = make([]float32, r.w*r.stride)
			r.v = make([]float32, r.w*r.stride)
		}
	}
	slot := p % r.w
	if r.quant == kvI8 {
		nKV := r.stride / r.headDim
		o, so := slot*r.stride, slot*nKV
		quantizeHeads(k, r.kq[o:o+r.stride], r.ksc[so:so+nKV], nKV, r.headDim)
		quantizeHeads(v, r.vq[o:o+r.stride], r.vsc[so:so+nKV], nKV, r.headDim)
	} else {
		o := slot * r.stride
		copy(r.k[o:o+r.stride], k)
		copy(r.v[o:o+r.stride], v)
	}
	if p+1 > r.count {
		r.count = p + 1
	}
}

// truncate drops logical positions ≥ p. Returns true iff the result is exact —
// i.e. every position the post-truncation window [max(0,p-w), p) is still
// physically resident. That holds whenever the ring never wrapped (count ≤ w);
// a deeper rewind on a wrapped ring would need positions already evicted, so it
// returns false and the caller must cold-prefill (the rewind rule, wired into
// sessions in Increment 2). Spec-decode draft depths ≪ w on short contexts never
// wrap, so they stay exact.
func (r *ring) truncate(p int) bool {
	if p >= r.count {
		return true
	}
	exact := r.count <= r.w || p >= r.count-1
	r.count = p
	return exact
}

// enableRings switches the local (sliding-window) layers to ring storage: layer l
// is local iff window>0 and it is not a global layer. Called by NewCache for the
// families whose forward uses attendQuery/attendBatchedHeads; other families pass
// a nil predicate or window 0 and keep append-forever. Idempotent-safe to call once.
func (c *KVCache) enableRings(window int, isGlobal func(int) bool) {
	if window <= 0 || isGlobal == nil {
		return
	}
	for l := range c.numLayers {
		if !isGlobal(l) {
			c.rings[l] = &ring{w: window, quant: c.quant, headDim: c.headDim}
			c.localAny = true
		}
	}
}

// setQuant switches the cache to int8 KV storage (kvI8). Global (append-forever)
// layers get parallel int8 + scale arrays; ring layers inherit the mode via
// enableRings (call setQuant FIRST). f32 default leaves everything bit-exact.
func (c *KVCache) setQuant(q kvQuant, capHint int) {
	c.quant = q
	if q == kvI8 {
		nKV := c.kvDim / c.headDim
		c.keysQ = make([][]int8, c.numLayers)
		c.valsQ = make([][]int8, c.numLayers)
		c.keyScale = make([][]float32, c.numLayers)
		c.valScale = make([][]float32, c.numLayers)
		// Pre-size the global int8 arrays so the per-token Append grows in place
		// (reslice, no alloc) instead of churning a fresh make() each step.
		for l := range c.numLayers {
			c.keysQ[l] = make([]int8, 0, capHint*c.kvDim)
			c.valsQ[l] = make([]int8, 0, capHint*c.kvDim)
			c.keyScale[l] = make([]float32, 0, capHint*nKV)
			c.valScale[l] = make([]float32, 0, capHint*nKV)
		}
	}
}

// Append stores one position's K and V for the given layer. k and v must
// each be kvDim long. Returns the position index just written.
func (c *KVCache) Append(layer int, k, v []float32) int {
	// Decode (single query) appends this token's K/V at the current position. A
	// local layer rings it (slot pos%W); a global layer append-forevers. Batched
	// prefill does NOT route local layers here — it defers the ring write past the
	// read (commitBatch), since writing all K first would evict in-batch history.
	if r := c.rings[layer]; r != nil {
		r.write(c.pos, k, v) // ring quantizes internally when kvI8
	} else if c.quant == kvI8 {
		c.appendQuant(layer, k, v)
	} else {
		c.keys[layer] = append(c.keys[layer], k...)
		c.vals[layer] = append(c.vals[layer], v...)
	}
	// pos advances once per last-layer append so all layers stay in lockstep —
	// unless manualPos (gemma4), where the caller calls Advance() after the sweep.
	if !c.manualPos && layer == c.numLayers-1 {
		c.pos++
	}
	return c.pos
}

// appendQuant quantizes one position's K/V per head and appends the int8 rows +
// per-head scales to a global (append-forever) int8 layer.
func (c *KVCache) appendQuant(layer int, k, v []float32) {
	nKV := c.kvDim / c.headDim
	// Grow each array by one position (reslice into the pre-reserved cap; append
	// only reallocs past it) and quantize directly into the new tail — no per-call
	// scratch allocation, which dominated the int8 prefill TTFT overhead.
	ko := len(c.keysQ[layer])
	c.keysQ[layer] = growI8(c.keysQ[layer], c.kvDim)
	c.valsQ[layer] = growI8(c.valsQ[layer], c.kvDim)
	so := len(c.keyScale[layer])
	c.keyScale[layer] = growF32(c.keyScale[layer], nKV)
	c.valScale[layer] = growF32(c.valScale[layer], nKV)
	quantizeHeads(k, c.keysQ[layer][ko:], c.keyScale[layer][so:], nKV, c.headDim)
	quantizeHeads(v, c.valsQ[layer][ko:], c.valScale[layer][so:], nKV, c.headDim)
}

// growI8/growF32 extend a slice by n, reslicing within capacity when possible
// (zero-alloc) and falling back to append (amortized) past it.
func growI8(s []int8, n int) []int8 {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	return append(s, make([]int8, n)...)
}

func growF32(s []float32, n int) []float32 {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	return append(s, make([]float32, n)...)
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
		if r := c.rings[l]; r != nil {
			r.truncate(pos) // exactness flag consumed by the rewind rule (Inc 2)
			continue
		}
		if c.quant == kvI8 {
			// int8 global: reslice the int8 rows + the per-(position,head) scales,
			// each strided per-layer (len/pos), so per-layer widths stay correct.
			perPos, perScale := 0, 0
			if c.pos > 0 {
				perPos = len(c.keysQ[l]) / c.pos
				perScale = len(c.keyScale[l]) / c.pos
			}
			c.keysQ[l] = c.keysQ[l][:pos*perPos]
			c.valsQ[l] = c.valsQ[l][:pos*perPos]
			c.keyScale[l] = c.keyScale[l][:pos*perScale]
			c.valScale[l] = c.valScale[l][:pos*perScale]
			continue
		}
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

// advanceTo sets the stored-position count directly — the batched prefill's
// explicit pos step, since it defers local-layer Appends past attention (so the
// last-layer Append auto-advance can't be relied on when that layer is local).
func (c *KVCache) advanceTo(pos int) { c.pos = pos }

// batchReadLocal assembles the contiguous [base, startPos+K) read buffer a local
// layer's batched prefill needs: the resident window [base, startPos) unwrapped
// from the ring in absolute order, followed by the K new K/V rows (already
// contiguous in newK/newV from the post-RoPE batch buffers). base = the absolute
// position of physical row 0 = max(0, startPos-W+1). dstK/dstV are caller scratch
// (≥ (startPos-base+K)*stride). Returns base and the physical row count. Does NOT
// write the ring — commitBatch does that after attention reads the history.
func (c *KVCache) batchReadLocal(layer, startPos, K int, newK, newV, dstK, dstV []float32) (base, nRows int) {
	r := c.rings[layer]
	stride := r.stride
	if stride == 0 {
		stride = len(newK) / K // first prefill: ring not yet sized
	}
	base = max(startPos-r.w+1, 0)
	hist := startPos - base // resident history rows, ≤ W-1 and ≤ r.count
	if r.quant == kvI8 {
		// int8 ring: dequant the resident window from int8, and round-trip the K
		// new rows (quantize→dequant) so prefill attends the SAME dequantized K/V
		// decode will (commitBatch writes the same int8). The f32 matmul is unchanged.
		nKV := stride / r.headDim
		for p := base; p < startPos; p++ {
			o, so, d := (p%r.w)*stride, (p%r.w)*nKV, (p-base)*stride
			dequantHeads(r.kq[o:o+stride], r.ksc[so:so+nKV], nKV, r.headDim, dstK[d:d+stride])
			dequantHeads(r.vq[o:o+stride], r.vsc[so:so+nKV], nKV, r.headDim, dstV[d:d+stride])
		}
		tq, tsc := make([]int8, stride), make([]float32, nKV)
		for i := 0; i < K; i++ {
			d := (hist + i) * stride
			quantizeHeads(newK[i*stride:(i+1)*stride], tq, tsc, nKV, r.headDim)
			dequantHeads(tq, tsc, nKV, r.headDim, dstK[d:d+stride])
			quantizeHeads(newV[i*stride:(i+1)*stride], tq, tsc, nKV, r.headDim)
			dequantHeads(tq, tsc, nKV, r.headDim, dstV[d:d+stride])
		}
		return base, hist + K
	}
	for p := base; p < startPos; p++ {
		o := (p % r.w) * stride
		d := (p - base) * stride
		copy(dstK[d:d+stride], r.k[o:o+stride])
		copy(dstV[d:d+stride], r.v[o:o+stride])
	}
	copy(dstK[hist*stride:(hist+K)*stride], newK[:K*stride])
	copy(dstV[hist*stride:(hist+K)*stride], newV[:K*stride])
	return base, hist + K
}

// dequantGlobalLayer dequantizes a global (append-forever) int8 layer's stored
// K/V into the caller's f32 scratch dstK/dstV (≥ storedRows*kvDim) for the
// batched prefill's f32 attention. Returns the stored row count. The new keys
// were already quantized into the layer by the prefill's Append loop, so this
// dequant covers history + new uniformly (matching decode's dequant reads).
func (c *KVCache) dequantGlobalLayer(layer, kvDim int, dstK, dstV []float32) int {
	nKV := kvDim / c.headDim
	n := len(c.keysQ[layer]) / kvDim
	for p := 0; p < n; p++ {
		o, so := p*kvDim, p*nKV
		dequantHeads(c.keysQ[layer][o:o+kvDim], c.keyScale[layer][so:so+nKV], nKV, c.headDim, dstK[o:o+kvDim])
		dequantHeads(c.valsQ[layer][o:o+kvDim], c.valScale[layer][so:so+nKV], nKV, c.headDim, dstV[o:o+kvDim])
	}
	return n
}

// commitBatch writes a batched prefill's K new positions into the ring, after
// attention has read the (now safe to overwrite) history. The ring keeps the last
// W of [startPos, startPos+K).
func (c *KVCache) commitBatch(layer, startPos, K int, newK, newV []float32) {
	r := c.rings[layer]
	stride := len(newK) / K
	for i := range K {
		o := i * stride
		r.write(startPos+i, newK[o:o+stride], newV[o:o+stride])
	}
}

// isLocal reports whether layer is a ring (sliding-window) layer.
func (c *KVCache) isLocal(layer int) bool { return c.rings[layer] != nil }

// storedRows is the logical number of positions held for a layer: the ring's
// count (it physically keeps only W) for a local layer, else the append-forever
// length / width.
func (c *KVCache) storedRows(layer, kvDim int) int {
	if r := c.rings[layer]; r != nil {
		return r.count
	}
	if c.quant == kvI8 {
		return len(c.keysQ[layer]) / kvDim
	}
	return len(c.keys[layer]) / kvDim
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
