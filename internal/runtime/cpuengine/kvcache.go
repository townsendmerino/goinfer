package cpuengine

import (
	"github.com/townsendmerino/goinfer/internal/runtime/inkling2"
	"slices"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// KVQuant selects the CPU KV-cache storage precision. kvF32 (default) is
// bit-exact; KVI8 stores K and V as per-(position,KV-head) symmetric int8 + a
// f32 scale each — 4× smaller, and decode reads them with the SDOT integer dot
// (DotI8) instead of scalar f64. Opt-in (lossy);
type KVQuant uint8

const (
	KVQF32 KVQuant = iota
	KVI8
	KVQF16 // f16 (IEEE binary16, linalg.F32ToF16's rounding): half the bytes; read by widening into scratch, as int8 is
)

// widened reports whether stored rows are read by widening them into f32 scratch (int8, f16) rather than in place (f32).
func (q KVQuant) widened() bool { return q != KVQF32 }

// MustUnpadded panics on a padded cache: what names an accessor that hands out the store without its stride.
func (c *Cache) MustUnpadded(what string) {
	if c.RowPad != 0 {
		panic("decoder: " + what + " cannot read a row-padded KV cache; read layerRows and its stride")
	}
}

// SetRowPad pads the cache at Architecture.kvRowPad's answer for its precision (int8 is never padded): each layer's rows then sit
// kvDim+rowPad elements apart in keys/vals (or keysH/valsH), re-reserved here for capHint rows. A cache holding a ring, an MLA latent
// or hybrid state on an architecture kvRowPad accepts means the rule and the cache disagree, which panics.
func (c *Cache) SetRowPad(a *darch.Architecture, capHint int) {
	if c.Quant == KVI8 {
		return
	}
	eb := 4
	if c.Quant == KVQF16 {
		eb = 2
	}
	pad := a.KVRowPad(eb)
	if pad == 0 {
		return
	}
	if c.LocalAny || c.MLALatent != nil || c.ManualPos || a.NumKVHeads*a.HeadDim != c.KVDim {
		panic("decoder: Architecture.kvRowPad accepted a cache that holds a ring, an MLA latent or hybrid state")
	}
	c.RowPad = pad
	rs := c.KVDim + pad
	for l := range c.NumLayers {
		lc := c.AppendHint(l, capHint)
		if c.Quant == KVQF16 {
			c.KeysH[l], c.ValsH[l] = make([]uint16, 0, lc*rs), make([]uint16, 0, lc*rs)
		} else {
			c.KeysBuf[l], c.ValsBuf[l] = make([]float32, 0, lc*rs), make([]float32, 0, lc*rs)
		}
	}
}

// RowStride is the elements between one position's row and the next in a global layer's store: kvDim, plus the pad on a padded cache.
func (c *Cache) RowStride(kvDim int) int { return kvDim + c.RowPad }

// LayerRows is global layer l's f32 store and its row stride: a reader indexes position s at s*stride.
func (c *Cache) LayerRows(l, kvDim int) (k, v []float32, stride int) {
	return c.KeysBuf[l], c.ValsBuf[l], c.RowStride(kvDim)
}

// CompactRows returns n rows of a strided store with the pad dropped (the store itself when it is unpadded).
func CompactRows[T any](src []T, n, width, stride int) []T {
	if stride == width {
		return src[:n*width]
	}
	out := make([]T, n*width)
	for p := range n {
		copy(out[p*width:(p+1)*width], src[p*stride:p*stride+width])
	}
	return out
}

// PadRows lays n compact rows of width elements out at stride, the pad zeroed (the slice itself when stride is width).
func PadRows[T any](src []T, n, width, stride int) []T {
	if stride == width {
		return src
	}
	out := make([]T, n*stride)
	for p := range n {
		copy(out[p*stride:p*stride+width], src[p*width:(p+1)*width])
	}
	return out
}

// RowPadBytesHeld is the K/V bytes a cache's global f32 and f16 stores hold, pad included.
func (c *Cache) RowPadBytesHeld() int64 {
	var n int64
	for l := range c.KeysBuf {
		n += 4 * int64(len(c.KeysBuf[l])+len(c.ValsBuf[l]))
	}
	for l := range c.KeysH {
		n += 2 * int64(len(c.KeysH[l])+len(c.ValsH[l]))
	}
	return n
}

// roundF16InPlace rounds x to f16 and back, the f16 cache's storage rounding.
func roundF16InPlace(x []float32) {
	for i, f := range x {
		x[i] = linalg.F16ToF32(linalg.F32ToF16(f))
	}
}

// narrowF16Into stores src as f16 into dst; widenF16Into is its exact inverse. The f16 cache's one rounding.
func narrowF16Into(dst []uint16, src []float32) { linalg.F32ToF16Slice(dst, src) }
func widenF16Into(dst []float32, src []uint16)  { linalg.F16ToF32Slice(dst, src) }

// quantizeHeads writes src (nKV contiguous rows of headDim) as int8 into q with
// one symmetric maxabs/127 scale per head into scales[:nKV]. Per-head (not
// per-kvDim-row) scales bound RoPE'd-key outlier channels — the cheap KIVI-style
// win. headDim is 64/128, a clean multiple of the SDOT/AVX2 width.
func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {
	for h := range nKV {
		o := h * headDim
		scales[h] = linalg.QuantizeRowInt8(src[o:o+headDim], q[o:o+headDim])
	}
}

// DequantHeads expands one position's int8 KV (nKV head-rows of headDim, one scale per KV head) into f32 dst: the inverse of
// quantizeHeads, for the batched prefill's dequant-into-scratch (the f32 Matmul kernels stay unchanged). The arithmetic lives in
// aikit (linalg.DequantizeRowsInt8Into); this is the argument order goinfer's call sites use.
func DequantHeads(q []int8, scales []float32, nKV, headDim int, dst []float32) {
	linalg.DequantizeRowsInt8Into(dst, q, scales, nKV, headDim)
}

// Cache holds the per-layer key/value history for one generation
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
type Cache struct {
	// knobs is the owning model's operator-knob snapshot (knobs.go), which the forward reads through the cache it is given;
	// nil = read the live environment (a cache built by NewCache directly, in a test).
	Knobs KnobSource

	// cost, when set, counts this stream's operations into the phase costPhase (cost_cpu.go); nil is off.
	Cost      *cost.Counter
	CostPhase cost.Phase

	NumLayers int
	// planRows is the number of positions the current request expects the cache to end with (planRequest); rowCeiling is
	// the context size it may never exceed (0 = unknown). regrownBytes is the old capacity a growth dropped during the
	// position being appended, released to the OS at its last layer (kvRegrowReleaseBytes).
	PlanRows, RowCeiling int
	RegrownBytes         int
	KVDim                int      // NumKVHeads * HeadDim
	Window               int      // sliding-window cap for local layers; 0 = unbounded
	ImgBlocks            [][2]int // multimodal: position ranges [start,end) that attend bidirectionally (image blocks); nil = all-causal
	MRopePos             [][3]int // Qwen2.5-VL m-RoPE: per-sequence-position (t,h,w) rotary positions; nil = scalar RoPE. Indexed by absolute sequence position.
	MRopeDelta           int      // m-RoPE position delta: decode positions past the prefill are scalar seqPos+rope.MRoPEDelta (the image compresses positions, so delta is usually negative).
	// deepstack is Qwen3-VL's DeepStack for the prefill that carries it: after decoder layer l, for l < len(rows), rows[l]
	// ([n*hidden], one row per image position) is added to the hidden state at the image positions [start, start+n), as HF's
	// _deepstack_process does. nil for every other prefill; a decode step never touches it.
	Deepstack *DeepstackRows

	KeysBuf [][]float32 // per layer, appended [pos*kvDim] (global / append-forever layers)
	ValsBuf [][]float32
	// stride is each global layer's KV width, learned on its first append: the authoritative per-layer row width. TruncateTo uses it
	// instead of len/pos, which mis-derives the width when a mid-sweep forward error leaves earlier layers with one extra (ragged)
	// row. 0 for a KV-shared layer that never appends.
	Stride []int
	PosN   int // number of positions stored (the next position index)

	// rings holds a fixed-W ring buffer for each sliding-window (local) layer; a nil entry is a global layer that append-forevers
	// into keys/vals above. Set by enableRings (from NewCache) for the families on the AttendQuery/AttendBatchedHeads paths. A local
	// layer stores only the W most recent positions (the only ones any future query can read), so its KV is O(W), not O(context).
	//
	Rings    []*Ring
	LocalAny bool // any ring layer present (gates prefill's assembly scratch)
	// skipAppend reports the layers that reserve no append-forever capacity (NewCache's skipLayer; nil: none).
	SkipAppend func(l int) bool

	// int8 KV (quant == KVI8): global (append-forever) layers store K/V as int8
	// here, parallel to keys/vals, with one symmetric scale per (position,KV-head).
	// Ring layers carry their own int8 fields. KVQF32 leaves all of this nil.
	Quant              KVQuant
	HeadDim            int         // per-head width, for int8 scale striding
	KeysQ, ValsQ       [][]int8    // [pos*kvDim] (global int8 layers)
	KeyScale, ValScale [][]float32 // [pos*nKV]   (per-(position,head) scale)
	// f16 KV (quant == KVQF16): global layers store K/V as f16 here, parallel to keys/vals (which stay empty).
	KeysH, ValsH [][]uint16 // [pos*kvDim]

	// rowPad > 0 pads every global layer's rows (setRowPad): position s of layer l starts at s*(kvDim+rowPad) in keys/vals or
	// keysH/valsH, and stride[l] records that. Readers index by rowStride; Keys and Vals refuse a padded cache.
	RowPad int
	// f16InPlaceReads counts decode attention calls that read this cache's f16 rows in place (AttendBatchedHeadsKV).
	F16InPlaceReads int64

	// manualPos decouples pos from Append's last-layer trigger, for families whose last layer never appends (Gemma 4's is KV-shared;
	// a hybrid's linear layers do not Append): the caller advances pos explicitly via Advance() after each token's full layer sweep.
	ManualPos bool

	// recurrentResets counts resetRecurrent calls: a diagnostic the test that proves a clean session does not zero its state twice
	// reads. Nothing branches on it.
	RecurrentResets int

	// delta holds the Gated DeltaNet recurrent state for qwen3_5_moe's linear
	// layers (nil entry on softmax layers, nil slice on every other family). This
	// is the recurrent half of the hybrid cache —
	Delta []*DeltaState

	// conv holds the LFM2 gated short-convolution rolling window for the conv layers
	// (nil entry on attention layers, nil slice on every other family). The LFM2
	// analogue of delta/mamba below, and the smallest of the three: a window of
	// K-1 vectors per layer and no recurrent matrix at all.
	Conv []*ShortConvState

	// mamba holds the Mamba-2 recurrent state ({conv window, SSM state}) for
	// Granite-4.0-H's mamba layers (nil entry on attention layers, nil slice on
	// every other family). The Granite analogue of delta.
	Mamba []*Mamba2State

	// mlaLatent holds DeepSeek MLA's compressed-KV latent per layer, appended [pos*latentDim], latentDim = kv_lora_rank +
	// qk_rope_head_dim (the per-position stride, learned on the first append). The cache holds the low-rank latent, not a
	// reconstructed full K+V; forward_deepseek rebuilds per-head K/V from it each step. nil on every other family; the standard
	// keys/vals stay empty.
	MLALatent [][]float32
	LatentDim int

	// kda holds Bailing Hybrid's (Ling 3.0) KDA recurrent state (three conv windows + the
	// [headDim,headDim]-per-head matrix state) for the linear layers (nil entry on the MLA
	// layers, which use mlaLatent above instead; nil slice on every other family).
	KDA []*KDAState

	// Ink is each Inkling 2 visit's four short-convolution windows; nil for other families.
	Ink []*inkling2.ConvState

	Scr *DecodeScratch // per-stream reusable forward buffers (Model.NewCache sets it)

	// captureLayers, when non-nil, requests that runLayersFromEmbed copy the residual stream (the layer output) after each listed
	// layer index into captured[i]: the read-only hidden-state seam a draft head reads (block drafters). nil = no capture, zero
	// overhead. The copies never feed back into the forward, so the token output is byte-identical.
	CaptureLayers []int
	Captured      [][]float32

	// layerInputs, when non-nil, replaces the residual entering layer l with layerInputs[l] wherever that row is non-nil, in the
	// single-token walk (walkToken) only: a resident diff runs each CPU layer on the GPU's own input, so a layer's difference is
	// its own rounding and the K/V it appends are computed from the same input. Set through SetLayerInputsForTest; nil in
	// production, where it costs one nil check per layer.
	LayerInputs [][]float32

	// subCapture, when true, records the per-SUBLAYER contribution added to the residual —
	// scr.sub after attention (post sandwich-norm) and after the MLP — for every layer, into
	// subAttn[l]/subMLP[l]. This is finer than captureLayers (which sees only the layer output
	// residual): it separates the attention contribution from the MLP contribution, which is
	// what localizing a channel's sign flip to a specific sublayer needs. Diagnostic seam, same
	// contract as captureLayers: copies never feed back, so the token output is byte-identical.
	SubCapture bool
	SubAttn    [][]float32 // [layer][hidden] attention contribution (o-proj out, post sandwich-norm)
	SubMLP     [][]float32 // [layer][hidden] MLP contribution (down out, post sandwich-norm)
	SubMLPPre  [][]float32 // [layer][hidden] MLP down output BEFORE the post-MLP sandwich norm
	SubCtx     [][]float32 // [layer][qDim] attention CONTEXT, pre-o-proj (AttendBatchedHeads out)
	SubPLE     [][]float32 // [layer][hidden] Gemma 4's PLE-branch contribution (post-PLE norm); nil unless the layer capture asked for it

	// treeRowPos / treeMask, when non-nil, switch the batched verify (forwardN) from a linear causal chain to tree attention: row i
	// takes its RoPE position from treeRowPos[i] (its depth) instead of startPos+i, and among the K new batch keys attends only to
	// columns j with treeMask[i][j] true (its ancestor path, including itself), plus the whole committed prefix. nil = the ordinary
	// linear behaviour, byte-identical, zero overhead. Nothing in production sets them; they are tested plumbing.
	TreeRowPos []int
	TreeMask   [][]bool

	// lora is the active compute-time LoRA adapter for this stream (#7), nil for
	// the merged/base default. Set per-stream (Session.UseAdapter); the forward
	// adds each layer's low-rank delta after the base projection. A non-nil lora
	// forces the sequential prefill path (the batched forwardN is not wired).
	LoRALayers []LoRALayerDelta
}

// NewCache allocates an empty cache for a model with the given geometry. capHint pre-sizes the per-layer slices to avoid
// reallocation during a known-length generation; 0 is fine (grow on demand).
//
// skipLayer (nil-safe: nil means none skipped) reports whether layer l never writes its append-forever K/V at all (a linear, mamba
// or conv mixer layer, one of Nemotron's non-attention block kinds, or a layer enableRings will make a ring): its capHint
// reservation would be wasted capacity, large per stream at realistic context lengths and multiplied by every session the LRU
// keeps warm. setQuant honours it too. It is independent of
// the caller's own decision to pass capHint=0 for an MLA family: an MLA layer does hold attention-shaped KV, in c.MLALatent,
// which skipLayer knows nothing about.
func NewCache(numLayers, numKVHeads, headDim, window, capHint int, skipLayer func(l int) bool) *Cache {
	if capHint < 0 { // defensive: a negative hint (e.g. from a negative max_tokens) would
		capHint = 0 // panic makeslice with "cap out of range"; the HTTP surface rejects it
	} // upstream; this guards every other caller too.
	kvDim := numKVHeads * headDim
	c := &Cache{
		NumLayers:  numLayers,
		KVDim:      kvDim,
		HeadDim:    headDim,
		Window:     window,
		KeysBuf:    make([][]float32, numLayers),
		ValsBuf:    make([][]float32, numLayers),
		Stride:     make([]int, numLayers),
		SkipAppend: skipLayer,
	}
	for l := range numLayers {
		lc := c.AppendHint(l, capHint)
		c.KeysBuf[l] = make([]float32, 0, lc*kvDim)
		c.ValsBuf[l] = make([]float32, 0, lc*kvDim)
	}
	c.Rings = make([]*Ring, numLayers) // all nil: append-forever until enableRings
	return c
}

// Ring is the sliding-window KV store for one local layer: it physically holds
// at most w positions, the slot for absolute position p being p%w. The per-layer
// stride (KV width = NumKVHeads*HeadDim for this layer) is learned on the first
// write, so per-layer-width families (Gemma 4) need no special-casing. count is
// the logical length — total positions ever written; the live window (everything
// a future query can read) is [max(0,count-w), count).
type Ring struct {
	K, V     []float32 // f32 mode: [w*stride] each; position p at offset (p%w)*stride
	KH, VH   []uint16  // f16 mode: [w*stride]
	KQ, VQ   []int8    // int8 mode: [w*stride]
	KSC, VSC []float32 // int8 mode: per-(slot,head) scale, [w*nKV]
	W        int       // window size (slot count)
	Stride   int       // per-layer KV width; 0 until the first write sizes it
	Count    int       // logical positions written (the absolute next position)
	Quant    KVQuant   // kvF32 | KVI8 (set by enableRings)
	HeadDim  int       // for int8 scale striding (nKV = stride/headDim)
	tq       []int8    // int8 mode scratch (stride)
	tsc      []float32 // int8 mode scale scratch (nKV)
	th       []uint16  // f16 mode scratch (stride)
}

// Write stores position p's k/v into slot p%w and advances count past p. O(stride),
// no growth or realloc after the first write. Quantizes per head when KVI8.
func (r *Ring) Write(p int, k, v []float32) {
	if r.Stride == 0 {
		r.Stride = len(k)
		r.alloc()
	}
	slot := p % r.W
	if r.Quant == KVQF16 {
		o := slot * r.Stride
		narrowF16Into(r.KH[o:o+r.Stride], k)
		narrowF16Into(r.VH[o:o+r.Stride], v)
	} else if r.Quant == KVI8 {
		nKV := r.Stride / r.HeadDim
		o, so := slot*r.Stride, slot*nKV
		quantizeHeads(k, r.KQ[o:o+r.Stride], r.KSC[so:so+nKV], nKV, r.HeadDim)
		quantizeHeads(v, r.VQ[o:o+r.Stride], r.VSC[so:so+nKV], nKV, r.HeadDim)
	} else {
		o := slot * r.Stride
		copy(r.K[o:o+r.Stride], k)
		copy(r.V[o:o+r.Stride], v)
		if r.Mirrored() {
			m := o + r.W*r.Stride
			copy(r.K[m:m+r.Stride], k)
			copy(r.V[m:m+r.Stride], v)
		}
	}
	if p+1 > r.Count {
		r.Count = p + 1
	}
}

// alloc sizes the ring's storage for its mode once its stride is known.
func (r *Ring) alloc() {
	switch r.Quant {
	case KVI8:
		nKV := r.Stride / r.HeadDim
		r.KQ = make([]int8, r.W*r.Stride)
		r.VQ = make([]int8, r.W*r.Stride)
		r.KSC = make([]float32, r.W*nKV)
		r.VSC = make([]float32, r.W*nKV)
	case KVQF16:
		r.KH = make([]uint16, r.W*r.Stride)
		r.VH = make([]uint16, r.W*r.Stride)
	default:
		r.K = make([]float32, r.W*r.Stride)
		r.V = make([]float32, r.W*r.Stride)
	}
}

// RingDirectDecode makes K=1 decode on an f32 ring read its window in place instead of copying every resident row into scratch
// each token. It is an A/B handle and a test seam, never read from the environment: tests flip it to compare the two paths bit
// for bit.
var RingDirectDecode = true

// Mirrored reports whether the ring holds its second copy: slots [w, 2w) equal slots [0, w), so ANY run of up to w consecutive positions is one contiguous
// slice of k/v whatever the wrap point. f32 rings only; the int8 ring keeps the copy path (it round-trips the new row through the quantizer).
func (r *Ring) Mirrored() bool {
	return r.Quant == KVQF32 && r.Stride > 0 && len(r.K) == 2*r.W*r.Stride
}

// ensureMirror allocates the second copy the first time a window wraps, so a layer that never wraps (a short context) costs no extra memory and no copy, and a
// restored snapshot (which rebuilds k/v at w slots) simply loses the mirror and rebuilds it here. Every writer below (write, commitBatch) keeps it coherent
// from then on; TestRingMirror_* asserts that invariant after every kind of mutation.
func (r *Ring) ensureMirror() {
	if r.Mirrored() {
		return
	}
	ws := r.W * r.Stride
	nk, nv := make([]float32, 2*ws), make([]float32, 2*ws)
	copy(nk, r.K[:ws])
	copy(nk[ws:], r.K[:ws])
	copy(nv, r.V[:ws])
	copy(nv[ws:], r.V[:ws])
	r.K, r.V = nk, nv
}

// Window returns the K and V rows of absolute positions [base, base+n) as one contiguous slice each, with no copy: the run starts at slot base%w and, when it
// crosses the end of the first copy, continues into the mirror. n ≤ w. Requires the rows to be resident (they are the live window) and, if the run wraps, the
// mirror (ensureMirror).
func (r *Ring) Window(base, n int) (k, v []float32) {
	slot0 := base % r.W
	if slot0+n > r.W {
		r.ensureMirror()
	}
	o, e := slot0*r.Stride, (slot0+n)*r.Stride
	return r.K[o:e], r.V[o:e]
}

// Truncate drops logical positions >= p. It returns true iff the result is exact, i.e. every position of the post-truncation
// window [max(0,p-w), p) is still physically resident. That holds whenever the ring never wrapped (count <= w); a deeper rewind
// on a wrapped ring would need evicted positions, so it returns false and the caller must cold-prefill. Spec-decode draft depths
// far below w on short contexts never wrap, so they stay exact.
func (r *Ring) Truncate(p int) bool {
	if p >= r.Count {
		return true
	}
	exact := kvmgr.RingTruncateExact(r.Count, r.W, p)
	r.Count = p
	return exact
}

// EnableRings switches the local (sliding-window) layers to ring storage: layer l
// is local iff window>0 and it is not a global layer. Called by NewCache for the
// families whose forward uses AttendQuery/AttendBatchedHeads; other families pass
// a nil predicate or window 0 and keep append-forever. Idempotent-safe to call once.
func (c *Cache) EnableRings(window int, isGlobal func(int) bool) {
	if window <= 0 || isGlobal == nil {
		return
	}
	for l := range c.NumLayers {
		if !isGlobal(l) {
			c.Rings[l] = &Ring{W: window, Quant: c.Quant, HeadDim: c.HeadDim}
			c.LocalAny = true
		}
	}
}

// SetQuant switches the cache to int8 (KVI8) or f16 (KVQF16) KV storage. Global (append-forever) layers get parallel arrays
// and give up any f32 capacity (left empty and non-nil, as a restored KV-shared layer is); ring layers inherit the mode via
// enableRings (call setQuant FIRST). f32 default leaves everything bit-exact.
func (c *Cache) SetQuant(q KVQuant, capHint int) {
	c.Quant = q
	if q != KVQF32 {
		for l := range c.NumLayers {
			if cap(c.KeysBuf[l]) > 0 || cap(c.ValsBuf[l]) > 0 {
				c.KeysBuf[l], c.ValsBuf[l] = []float32{}, []float32{}
			}
		}
	}
	if q == KVQF16 {
		c.KeysH = make([][]uint16, c.NumLayers)
		c.ValsH = make([][]uint16, c.NumLayers)
		for l := range c.NumLayers {
			lc := c.AppendHint(l, capHint)
			c.KeysH[l] = make([]uint16, 0, lc*c.KVDim)
			c.ValsH[l] = make([]uint16, 0, lc*c.KVDim)
		}
	}
	if q == KVI8 {
		nKV := c.KVDim / c.HeadDim
		c.KeysQ = make([][]int8, c.NumLayers)
		c.ValsQ = make([][]int8, c.NumLayers)
		c.KeyScale = make([][]float32, c.NumLayers)
		c.ValScale = make([][]float32, c.NumLayers)
		// Pre-size the global int8 arrays so the per-token Append grows in place
		// (reslice, no alloc) instead of churning a fresh make() each step.
		for l := range c.NumLayers {
			lc := c.AppendHint(l, capHint)
			c.KeysQ[l] = make([]int8, 0, lc*c.KVDim)
			c.ValsQ[l] = make([]int8, 0, lc*c.KVDim)
			c.KeyScale[l] = make([]float32, 0, lc*nKV)
			c.ValScale[l] = make([]float32, 0, lc*nKV)
		}
	}
}

// ReserveWidths reserves each append-forever f32 layer at width(l) where that is not the cache's kvDim, for a hint of capHint rows. Only f32 needs it:
// a family with layers of different widths keeps f32 (Architecture.kvF16OK, kvInt8OK; TestNewCache_reservesEachLayersWidth).
func (c *Cache) ReserveWidths(capHint int, width func(l int) int) {
	for l := range c.NumLayers {
		if lc := c.AppendHint(l, capHint); lc > 0 && width(l) != c.KVDim {
			c.KeysBuf[l] = make([]float32, 0, lc*width(l))
			c.ValsBuf[l] = make([]float32, 0, lc*width(l))
		}
	}
}

// AppendHint is the rows of append-forever capacity layer l reserves for a hint of capHint: none for a layer skipAppend names.
func (c *Cache) AppendHint(l, capHint int) int {
	if c.SkipAppend != nil && c.SkipAppend(l) {
		return 0
	}
	return capHint
}

// Append stores one position's K and V for the given layer. k and v must
// each be kvDim long. Returns the position index just written.
func (c *Cache) Append(layer int, k, v []float32) int {
	// Decode (single query) appends this token's K/V at the current position. A
	// local layer rings it (slot pos%W); a global layer append-forevers. Batched
	// prefill does NOT route local layers here — it defers the ring write past the
	// read (commitBatch), since writing all K first would evict in-batch history.
	if r := c.Rings[layer]; r != nil {
		r.Write(c.PosN, k, v) // ring quantizes internally when KVI8
	} else {
		if c.Stride[layer] == 0 {
			c.Stride[layer] = len(k) + c.RowPad // the per-layer row stride, learned on first append (the pad: setRowPad)
		}
		st := c.Stride[layer]
		if c.Quant == KVQF16 {
			if len(c.KeysH[layer])+st > cap(c.KeysH[layer]) {
				c.regrowLayer(layer)
			}
			o := len(c.KeysH[layer])
			c.KeysH[layer] = growU16(c.KeysH[layer], st)
			c.ValsH[layer] = growU16(c.ValsH[layer], st)
			narrowF16Into(c.KeysH[layer][o:o+len(k)], k)
			narrowF16Into(c.ValsH[layer][o:o+len(v)], v)
		} else if c.Quant == KVI8 {
			if len(c.KeysQ[layer])+len(k) > cap(c.KeysQ[layer]) {
				c.regrowLayer(layer)
			}
			c.appendQuant(layer, k, v)
		} else {
			if len(c.KeysBuf[layer])+st > cap(c.KeysBuf[layer]) {
				c.regrowLayer(layer)
			}
			c.KeysBuf[layer] = append(c.KeysBuf[layer], k...)
			c.ValsBuf[layer] = append(c.ValsBuf[layer], v...)
			if c.RowPad > 0 {
				c.KeysBuf[layer] = growF32(c.KeysBuf[layer], c.RowPad)
				c.ValsBuf[layer] = growF32(c.ValsBuf[layer], c.RowPad)
			}
		}
	}
	if layer == c.NumLayers-1 && c.RegrownBytes >= kvRegrowReleaseBytes {
		c.RegrownBytes = 0
		if fn := activeHooks().ReleaseHostMemory; fn != nil {
			fn()
		}
	}
	// pos advances once per last-layer append so all layers stay in lockstep —
	// unless manualPos (gemma4), where the caller calls Advance() after the sweep.
	if !c.ManualPos && layer == c.NumLayers-1 {
		c.PosN++
	}
	return c.PosN
}

// kvRegrowReleaseBytes is the old KV capacity, summed over a position's layers, above which a growth returns freed memory
// to the OS at once: below it the copy is left to the runtime's scavenger.
const kvRegrowReleaseBytes = 64 << 20

// KVRegrows counts KV layer growths (regrowLayer), for gates that assert a request within its plan never grows the cache.
var KVRegrows atomic.Int64

// PlanRequest records that the current request expects the cache to end with promptRows+budget positions, at most
// ceiling (the context size; 0 = unknown). The prefill reserves to it, and a growth stops at the ceiling. A ceiling the
// prompt already exceeds is not being enforced on this cache, so it is ignored rather than allowed to shrink the plan.
func (c *Cache) PlanRequest(promptRows, budget, ceiling int) {
	if ceiling < promptRows {
		ceiling = 0
	}
	rows := promptRows + max(budget, 0)
	if ceiling > 0 {
		rows = min(rows, ceiling)
	}
	c.PlanRows, c.RowCeiling = rows, ceiling
}

// regrowLayer grows one append-forever layer that has no room for another position, by the cache's growth policy: to a
// quarter more rows than it holds, at least 256 more, and never past the context size while that leaves a row free. The
// old arrays are dropped here, and their capacity is counted toward the release at the position's last layer.
func (c *Cache) regrowLayer(l int) {
	st := c.Stride[l]
	held := len(c.KeysBuf[l]) / st
	if c.Quant == KVI8 {
		held = len(c.KeysQ[l]) / st
	} else if c.Quant == KVQF16 {
		held = len(c.KeysH[l]) / st
	}
	rows := held + max(held/4, 256)
	if c.RowCeiling > held {
		rows = min(rows, c.RowCeiling)
	}
	grow := (rows - held) * st
	KVRegrows.Add(1)
	if c.Quant == KVQF16 {
		c.RegrownBytes += 2 * (cap(c.KeysH[l]) + cap(c.ValsH[l]))
		c.KeysH[l] = slices.Grow(c.KeysH[l], grow)
		c.ValsH[l] = slices.Grow(c.ValsH[l], grow)
		return
	}
	if c.Quant == KVI8 {
		nKV := st / c.HeadDim
		c.RegrownBytes += cap(c.KeysQ[l]) + cap(c.ValsQ[l]) + 4*(cap(c.KeyScale[l])+cap(c.ValScale[l]))
		c.KeysQ[l] = slices.Grow(c.KeysQ[l], grow)
		c.ValsQ[l] = slices.Grow(c.ValsQ[l], grow)
		c.KeyScale[l] = slices.Grow(c.KeyScale[l], (rows-held)*nKV)
		c.ValScale[l] = slices.Grow(c.ValScale[l], (rows-held)*nKV)
		return
	}
	c.RegrownBytes += 4 * (cap(c.KeysBuf[l]) + cap(c.ValsBuf[l]))
	c.KeysBuf[l] = slices.Grow(c.KeysBuf[l], grow)
	c.ValsBuf[l] = slices.Grow(c.ValsBuf[l], grow)
}

// appendQuant quantizes one position's K/V per head and appends the int8 rows +
// per-head scales to a global (append-forever) int8 layer.
func (c *Cache) appendQuant(layer int, k, v []float32) {
	nKV := c.KVDim / c.HeadDim
	// Grow each array by one position (reslice into the pre-reserved cap; append
	// only reallocs past it) and quantize directly into the new tail — no per-call
	// scratch allocation, which dominated the int8 prefill TTFT overhead.
	ko := len(c.KeysQ[layer])
	c.KeysQ[layer] = growI8(c.KeysQ[layer], c.KVDim)
	c.ValsQ[layer] = growI8(c.ValsQ[layer], c.KVDim)
	so := len(c.KeyScale[layer])
	c.KeyScale[layer] = growF32(c.KeyScale[layer], nKV)
	c.ValScale[layer] = growF32(c.ValScale[layer], nKV)
	quantizeHeads(k, c.KeysQ[layer][ko:], c.KeyScale[layer][so:], nKV, c.HeadDim)
	quantizeHeads(v, c.ValsQ[layer][ko:], c.ValScale[layer][so:], nKV, c.HeadDim)
}

// growI8/growF32 extend a slice by n, reslicing within capacity when possible
// (zero-alloc) and falling back to append (amortized) past it.
func growI8(s []int8, n int) []int8 {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	newCap := max(2*cap(s), len(s)+n)
	grown := make([]int8, len(s)+n, newCap)
	copy(grown, s)
	return grown
}

func growU16(s []uint16, n int) []uint16 {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	grown := make([]uint16, len(s)+n, max(2*cap(s), len(s)+n))
	copy(grown, s)
	return grown
}

func growF32(s []float32, n int) []float32 {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	newCap := max(2*cap(s), len(s)+n)
	grown := make([]float32, len(s)+n, newCap)
	copy(grown, s)
	return grown
}

// Advance bumps the stored-position count by one — the gemma4 forward's explicit
// pos step (its last layer is KV-shared, so Append's auto-advance never fires).
func (c *Cache) Advance() { c.PosN++ }

// Keys / Vals return the stored history for a layer as [storedPos, kvDim].
func (c *Cache) Keys(layer int) []float32 { c.MustUnpadded("Cache.Keys"); return c.KeysBuf[layer] }
func (c *Cache) Vals(layer int) []float32 { c.MustUnpadded("Cache.Vals"); return c.ValsBuf[layer] }

// LayerKV returns a layer's stored K and V as f32 in absolute position order — the live
// sliding-window for a ring (local) layer, the full history for an append-forever global —
// dequantized when the cache is int8. base is the absolute position of row 0. Keys/Vals above
// only see the global append-forever store, so they read empty for a windowed layer (Gemma's
// local layers); this is the ring-aware read a resident-KV upload bridge needs to place a CPU
// prefill's K/V at the GPU cache's correct absolute positions (decoder/generate_vl_resident.go).
func (c *Cache) LayerKV(layer int) (k, v []float32, base int) {
	if r := c.Rings[layer]; r != nil {
		lo := kvmgr.WindowHorizon(r.Count, r.W, 0)
		nLive, st := r.Count-lo, r.Stride
		if st == 0 || nLive == 0 {
			return nil, nil, lo
		}
		k, v = make([]float32, nLive*st), make([]float32, nLive*st)
		nKV := st / r.HeadDim
		for i, p := 0, lo; p < r.Count; i, p = i+1, p+1 {
			so, do := (p%r.W)*st, i*st
			if r.Quant == KVQF16 {
				widenF16Into(k[do:do+st], r.KH[so:so+st])
				widenF16Into(v[do:do+st], r.VH[so:so+st])
			} else if r.Quant == KVI8 {
				sso := (p % r.W) * nKV
				DequantHeads(r.KQ[so:so+st], r.KSC[sso:sso+nKV], nKV, r.HeadDim, k[do:do+st])
				DequantHeads(r.VQ[so:so+st], r.VSC[sso:sso+nKV], nKV, r.HeadDim, v[do:do+st])
			} else {
				copy(k[do:do+st], r.K[so:so+st])
				copy(v[do:do+st], r.V[so:so+st])
			}
		}
		return k, v, lo
	}
	if c.Quant == KVQF16 {
		n := c.StoredRows(layer, c.KVDim)
		kh, vh := CompactRows(c.KeysH[layer], n, c.KVDim, c.RowStride(c.KVDim)), CompactRows(c.ValsH[layer], n, c.KVDim, c.RowStride(c.KVDim))
		k, v = make([]float32, len(kh)), make([]float32, len(vh))
		if n > 0 {
			widenF16Into(k, kh)
			widenF16Into(v, vh)
		}
		return k, v, 0
	}
	if c.Quant == KVI8 {
		st := c.KVDim
		nKV := st / c.HeadDim
		n := len(c.KeysQ[layer]) / st
		k, v = make([]float32, n*st), make([]float32, n*st)
		for p := range n {
			o, so := p*st, p*nKV
			DequantHeads(c.KeysQ[layer][o:o+st], c.KeyScale[layer][so:so+nKV], nKV, c.HeadDim, k[o:o+st])
			DequantHeads(c.ValsQ[layer][o:o+st], c.ValScale[layer][so:so+nKV], nKV, c.HeadDim, v[o:o+st])
		}
		return k, v, 0
	}
	if c.RowPad > 0 { // the bridge reads compact rows
		n := c.StoredRows(layer, c.KVDim)
		return CompactRows(c.KeysBuf[layer], n, c.KVDim, c.RowStride(c.KVDim)), CompactRows(c.ValsBuf[layer], n, c.KVDim, c.RowStride(c.KVDim)), 0
	}
	return c.KeysBuf[layer], c.ValsBuf[layer], 0
}

// AppendLatent stores one position's MLA latent (the compressed KV ‖ rope-key vector)
// for a layer, growing the per-layer store. Append-forever (no ring/quant yet — the
// latent is already the compressed state). The caller advances pos once per token via
// Advance() (manualPos), since the standard Append path is bypassed.
func (c *Cache) AppendLatent(layer int, latent []float32) {
	if c.LatentDim == 0 {
		c.LatentDim = len(latent)
	}
	c.MLALatent[layer] = append(c.MLALatent[layer], latent...)
}

// Latent returns the stored MLA latent history for a layer as [storedPos, latentDim].
func (c *Cache) Latent(layer int) []float32 { return c.MLALatent[layer] }

// prefillChunkable reports whether a prefill into this cache may run in several passes: nothing on the cache keeps per-pass
// state that a pass boundary would split (a hidden-state capture, a speculative tree, a caller-driven position). Image blocks,
// m-RoPE positions and DeepStack rows are addressed by absolute position, so they carry across passes; prefillCut keeps each
// image block in one pass.
func (c *Cache) prefillChunkable() bool {
	return c.CaptureLayers == nil && !c.SubCapture && c.TreeRowPos == nil && c.TreeMask == nil && !c.ManualPos
}

// prefillCut returns where a pass of a prefill starting at position start should end, given its rows [lo, hi): hi, unless that
// falls inside an image block, which a pass must hold whole (its rows attend to each other). The cut then moves back to the
// block's start, or forward to its end when the block starts at or before lo.
func (c *Cache) prefillCut(start, lo, hi int) int {
	return CutOutsideBlocks(c.ImgBlocks, start, lo, hi)
}

// CutOutsideBlocks is prefillCut over the bidirectional blocks given ([start, end), absolute positions).
func CutOutsideBlocks(blocks [][2]int, start, lo, hi int) int {
	p := start + hi
	for _, b := range blocks {
		if b[0] < p && p < b[1] {
			if b[0] > start+lo {
				return b[0] - start
			}
			return b[1] - start
		}
	}
	return hi
}

// ReserveRows grows each append-forever attention layer's arrays so that rows more positions append without reallocating.
// Capacity only: nothing stored changes. A ring layer, and a layer whose width is not yet known (nothing written), is left
// alone.
func (c *Cache) ReserveRows(rows int) {
	if rows <= 0 {
		return
	}
	for l := range c.NumLayers {
		if c.Rings[l] != nil {
			continue
		}
		if c.MLALatent != nil {
			if c.LatentDim > 0 {
				c.MLALatent[l] = slices.Grow(c.MLALatent[l], rows*c.LatentDim)
			}
			continue
		}
		st := c.Stride[l]
		if st == 0 {
			continue
		}
		if c.Quant == KVQF16 {
			c.KeysH[l] = slices.Grow(c.KeysH[l], rows*st)
			c.ValsH[l] = slices.Grow(c.ValsH[l], rows*st)
			continue
		}
		if c.Quant == KVI8 {
			nKV := st / c.HeadDim
			c.KeysQ[l] = slices.Grow(c.KeysQ[l], rows*st)
			c.ValsQ[l] = slices.Grow(c.ValsQ[l], rows*st)
			c.KeyScale[l] = slices.Grow(c.KeyScale[l], rows*nKV)
			c.ValScale[l] = slices.Grow(c.ValScale[l], rows*nKV)
			continue
		}
		c.KeysBuf[l] = slices.Grow(c.KeysBuf[l], rows*st)
		c.ValsBuf[l] = slices.Grow(c.ValsBuf[l], rows*st)
	}
}

// Pos is the number of positions stored so far.
func (c *Cache) Pos() int { return c.PosN }

// HasRecurrentState reports whether this cache carries state mutated in place per token with no per-position history: TruncateTo
// cannot rewind it, Snapshot cannot persist it, and a rolled-back session must go cold rather than warm-reuse it. It is one
// predicate over every kind (c.Mamba, c.Delta, c.Conv, c.KDA, c.Ink), read at every site: a kind missed at one site leaks one
// conversation's window into the next (TruncateTo(0) leaving it intact) and reports a partial rewind exact, so a rewound prefix
// is warm-reused over stale windows. recurrent_census_test.go asks the struct, so a new kind cannot be missed.
func (c *Cache) HasRecurrentState() bool {
	return c.Mamba != nil || c.Delta != nil || c.Conv != nil || c.KDA != nil || c.Ink != nil
}

// ResetRecurrent re-zeroes the rolling state of every recurrent kind (Mamba-2, Gated DeltaNet, LFM2 conv, KDA: conv windows and
// SSM / linear-attention state) so a reused cache does not leak the prior sequence's recurrence into a fresh one. No-op on
// non-recurrent families (nil slices).
func (c *Cache) ResetRecurrent() {
	c.RecurrentResets++
	// A conv window left behind would bleed the previous sequence's last K-1 tokens into the first K-1 of the next: a small, fluent,
	// entirely wrong prefix.
	for _, st := range c.Conv {
		if st != nil {
			st.ConvWin = nil
		}
	}
	for _, st := range c.Mamba {
		if st != nil {
			st.Reset()
		}
	}
	for _, st := range c.Delta {
		if st != nil {
			st.ConvWin = nil
			for i := range st.S {
				st.S[i] = 0
			}
		}
	}
	// KDA: three per-stream conv windows plus the per-head matrix state, all mutated in place per token, the same leak as above.
	for _, st := range c.KDA {
		if st != nil {
			st.ConvWinQ, st.ConvWinK, st.ConvWinV = nil, nil, nil
			for i := range st.S {
				st.S[i] = 0
			}
		}
	}
	for _, st := range c.Ink {
		if st != nil {
			st.Reset()
		}
	}
}

// ResetMultimodal clears the per-sequence multimodal state (image attention blocks, Qwen2.5-VL m-RoPE positions and delta) so a
// reused cache does not leak a prior sequence's image blocks or offsets into a fresh one. Like recurrent state it has no
// per-position rewind, so it is cleared only on a full reset. No-op on text-only caches. Not a live leak today (every VL
// generation uses a fresh cache and warm-KV sessions are text-only); it keeps the reset complete before VL reaches the session
// path.
func (c *Cache) ResetMultimodal() {
	c.ImgBlocks = nil
	c.MRopePos = nil
	c.MRopeDelta = 0
}

// TruncateTo rewinds the cache to hold exactly pos positions (pos in [0, Pos()]) and returns whether the rewind was exact. It is
// the rollback speculative decoding needs after a partial accept, and the seam Session prefix reuse rewinds to a shared prefix
// through. Cheap: a reslice that keeps the backing arrays. Callers reusing a rewound prefix MUST cold-prefill when exact is
// false, or produce silently wrong output. A wrapped sliding-window ring (count>w) rewound by more than one position cannot
// restore the dropped positions (its slots still physically hold them and attention reads them as history), and recurrent and
// multimodal state have no per-position rewind; global, int8 and MLA layers store every position and always rewind exactly. A
// rewind to 0 also resets the recurrent and multimodal state.
//
// The per-position stride is the per-layer width recorded at first append, not the nominal kvDim: Gemma 4's per-layer head_dim
// and KV-head counts make widths differ, and its KV-shared tail layers append nothing (stride 0, so they stay empty).
func (c *Cache) TruncateTo(pos int) (exact bool) {
	exact = true
	if pos < 0 || pos > c.PosN {
		return exact // out-of-range no-op is exact
	}
	// Recurrent state is a single rolling state with no per-position history, so it cannot be exactly rewound. Reset it on a full
	// clear (Session.Reset → TruncateTo(0), and sessionLRU.fresh); a partial rewind is reported inexact below, so rewindForReuse
	// cold-prefills rather than decoding a new sequence from the previous one's leaked state.
	if c.HasRecurrentState() && pos == 0 {
		c.ResetRecurrent()
	}
	// A partial rewind is inexact for every kind the cache-state grid marks so (cachestate.go): the
	// four recurrent kinds, and the multimodal image blocks and m-RoPE positions, which are set by an
	// image prefill and have no per-position rewind either. Read from the grid rather than listed
	// here, so a new kind declared inexact is honoured at this site the day it is declared.
	if pos > 0 && pos < c.PosN && (func() bool {
		if fn := inexactRewindCheck; fn != nil {
			return fn(c)
		}
		return c.HasRecurrentState() || len(c.ImgBlocks) > 0 || len(c.MRopePos) > 0
	}()) {
		exact = false
	}
	// Multimodal state (image blocks + m-RoPE) also has no per-position rewind, and unlike recurrent state can live on a
	// non-recurrent (VL) family, so clear it on any full reset, outside the recurrent guard.
	if pos == 0 {
		c.ResetMultimodal()
	}
	for l := range c.NumLayers {
		if c.MLALatent != nil { // DeepSeek MLA: reslice the per-layer latent store
			c.MLALatent[l] = c.MLALatent[l][:min(pos*c.LatentDim, len(c.MLALatent[l]))]
			continue
		}
		if r := c.Rings[l]; r != nil {
			if !r.Truncate(pos) {
				exact = false // wrapped ring rewound >1 position: window holds dropped K/V
			}
			continue
		}
		// Global layers: truncate by the per-layer width recorded at first append, not len/pos, which mis-slices a ragged layer left by
		// a mid-sweep forward error. min guards a KV-shared (stride 0) or shorter-than-pos layer.
		st := c.Stride[l]
		if c.Quant == KVQF16 {
			c.KeysH[l] = c.KeysH[l][:min(pos*st, len(c.KeysH[l]))]
			c.ValsH[l] = c.ValsH[l][:min(pos*st, len(c.ValsH[l]))]
			continue
		}
		if c.Quant == KVI8 {
			nKV := 0
			if c.HeadDim > 0 {
				nKV = st / c.HeadDim
			}
			c.KeysQ[l] = c.KeysQ[l][:min(pos*st, len(c.KeysQ[l]))]
			c.ValsQ[l] = c.ValsQ[l][:min(pos*st, len(c.ValsQ[l]))]
			c.KeyScale[l] = c.KeyScale[l][:min(pos*nKV, len(c.KeyScale[l]))]
			c.ValScale[l] = c.ValScale[l][:min(pos*nKV, len(c.ValScale[l]))]
			continue
		}
		c.KeysBuf[l] = c.KeysBuf[l][:min(pos*st, len(c.KeysBuf[l]))]
		c.ValsBuf[l] = c.ValsBuf[l][:min(pos*st, len(c.ValsBuf[l]))]
	}
	c.PosN = pos
	return exact
}

// AdvanceTo sets the stored-position count directly — the batched prefill's
// explicit pos step, since it defers local-layer Appends past attention (so the
// last-layer Append auto-advance can't be relied on when that layer is local).
func (c *Cache) AdvanceTo(pos int) { c.PosN = pos }

// BatchReadLocal assembles the contiguous [base, startPos+K) read buffer a local
// layer's batched prefill needs: the resident window [base, startPos) unwrapped
// from the ring in absolute order, followed by the K new K/V rows (already
// contiguous in newK/newV from the post-RoPE batch buffers). base = the absolute
// position of physical row 0 = max(0, startPos-W+1). dstK/dstV are caller scratch
// (≥ (startPos-base+K)*stride). Returns base and the physical row count. Does NOT
// write the ring — commitBatch does that after attention reads the history.
func (c *Cache) BatchReadLocal(layer, startPos, K int, newK, newV, dstK, dstV []float32) (base, nRows int) {
	r := c.Rings[layer]
	stride := r.Stride
	if stride == 0 {
		stride = len(newK) / K // first prefill: ring not yet sized
	}
	base = kvmgr.WindowNeedStart(startPos, r.W)
	hist := startPos - base // resident history rows, ≤ W-1 and ≤ r.Count
	slot0 := base % r.W
	seg1 := min(hist, r.W-slot0)
	if r.Quant == KVQF16 {
		// f16 ring: widen the resident window, and round-trip the K new rows through f16 so prefill attends the values decode
		// will read back (commitBatch stores the same f16).
		if activeHooks().KVF16RingOffset {
			slot0 = (slot0 + 1) % r.W
			seg1 = min(hist, r.W-slot0)
		}
		if seg1 > 0 {
			len1, src1 := seg1*stride, slot0*stride
			widenF16Into(dstK[:len1], r.KH[src1:src1+len1])
			widenF16Into(dstV[:len1], r.VH[src1:src1+len1])
		}
		if seg2 := hist - seg1; seg2 > 0 {
			len1, len2 := seg1*stride, seg2*stride
			widenF16Into(dstK[len1:len1+len2], r.KH[:len2])
			widenF16Into(dstV[len1:len1+len2], r.VH[:len2])
		}
		if cap(r.th) < stride {
			r.th = make([]uint16, stride)
		}
		th := r.th[:stride]
		for i := range K {
			d := (hist + i) * stride
			if activeHooks().KVF16NewRowsF32 {
				copy(dstK[d:d+stride], newK[i*stride:(i+1)*stride])
				copy(dstV[d:d+stride], newV[i*stride:(i+1)*stride])
				continue
			}
			narrowF16Into(th, newK[i*stride:(i+1)*stride])
			widenF16Into(dstK[d:d+stride], th)
			narrowF16Into(th, newV[i*stride:(i+1)*stride])
			widenF16Into(dstV[d:d+stride], th)
		}
		return base, hist + K
	}
	if r.Quant == KVI8 {
		// int8 ring: dequant the resident window from int8, and round-trip the K
		// new rows (quantize→dequant) so prefill attends the SAME dequantized K/V
		// decode will (commitBatch writes the same int8). The f32 Matmul is unchanged.
		nKV := stride / r.HeadDim
		if seg1 > 0 {
			len1 := seg1 * stride
			src1 := slot0 * stride
			sc1 := slot0 * nKV
			rows1 := seg1 * nKV
			DequantKVRows(dstK[:len1], dstV[:len1], r.KQ[src1:src1+len1], r.VQ[src1:src1+len1], r.KSC[sc1:sc1+rows1], r.VSC[sc1:sc1+rows1], rows1, r.HeadDim)
		}
		if seg2 := hist - seg1; seg2 > 0 {
			len1 := seg1 * stride
			len2 := seg2 * stride
			rows2 := seg2 * nKV
			DequantKVRows(dstK[len1:len1+len2], dstV[len1:len1+len2], r.KQ[:len2], r.VQ[:len2], r.KSC[:rows2], r.VSC[:rows2], rows2, r.HeadDim)
		}
		if cap(r.tq) < stride {
			r.tq = make([]int8, stride)
		}
		if cap(r.tsc) < nKV {
			r.tsc = make([]float32, nKV)
		}
		tq := r.tq[:stride]
		tsc := r.tsc[:nKV]
		for i := range K {
			d := (hist + i) * stride
			quantizeHeads(newK[i*stride:(i+1)*stride], tq, tsc, nKV, r.HeadDim)
			DequantHeads(tq, tsc, nKV, r.HeadDim, dstK[d:d+stride])
			quantizeHeads(newV[i*stride:(i+1)*stride], tq, tsc, nKV, r.HeadDim)
			DequantHeads(tq, tsc, nKV, r.HeadDim, dstV[d:d+stride])
		}
		return base, hist + K
	}
	if seg1 > 0 {
		len1 := seg1 * stride
		src1 := slot0 * stride
		copy(dstK[:len1], r.K[src1:src1+len1])
		copy(dstV[:len1], r.V[src1:src1+len1])
	}
	if seg2 := hist - seg1; seg2 > 0 {
		len1 := seg1 * stride
		len2 := seg2 * stride
		copy(dstK[len1:len1+len2], r.K[:len2])
		copy(dstV[len1:len1+len2], r.V[:len2])
	}
	copy(dstK[hist*stride:(hist+K)*stride], newK[:K*stride])
	copy(dstV[hist*stride:(hist+K)*stride], newV[:K*stride])
	return base, hist + K
}

// DequantGlobalLayer dequantizes a global (append-forever) int8 layer's stored
// K/V into the caller's f32 scratch dstK/dstV (≥ storedRows*kvDim) for the
// batched prefill's f32 attention. Returns the stored row count. The new keys
// were already quantized into the layer by the prefill's Append loop, so this
// dequant covers history + new uniformly (matching decode's dequant reads).
func (c *Cache) DequantGlobalLayer(layer, kvDim int, dstK, dstV []float32) int {
	if c.Quant == KVQF16 {
		n, rs := c.StoredRows(layer, kvDim), c.RowStride(kvDim)
		if rs != kvDim { // a padded store widens row by row into compact scratch
			for p := range n {
				widenF16Into(dstK[p*kvDim:(p+1)*kvDim], c.KeysH[layer][p*rs:p*rs+kvDim])
				widenF16Into(dstV[p*kvDim:(p+1)*kvDim], c.ValsH[layer][p*rs:p*rs+kvDim])
			}
			return n
		}
		if n > 0 {
			widenF16Into(dstK[:n*kvDim], c.KeysH[layer][:n*kvDim])
			widenF16Into(dstV[:n*kvDim], c.ValsH[layer][:n*kvDim])
		}
		return n
	}
	nKV := kvDim / c.HeadDim
	n := len(c.KeysQ[layer]) / kvDim
	if n == 0 {
		return 0
	}
	totalRows := n * nKV
	totalFloats := n * kvDim
	DequantKVRows(dstK[:totalFloats], dstV[:totalFloats], c.KeysQ[layer][:totalFloats], c.ValsQ[layer][:totalFloats],
		c.KeyScale[layer][:totalRows], c.ValScale[layer][:totalRows], totalRows, c.HeadDim) // split by rows, bit-identical
	return n
}

// CommitBatch writes a batched prefill's K new positions into the ring, after
// attention has read the (now safe to overwrite) history. The ring keeps the last
// W of [startPos, startPos+K).
func (c *Cache) CommitBatch(layer, startPos, K int, newK, newV []float32) {
	r := c.Rings[layer]
	stride := len(newK) / K
	if r.Stride == 0 {
		r.Stride = stride
		r.alloc()
	}
	if r.Quant.widened() {
		for i := range K {
			o := i * stride
			r.Write(startPos+i, newK[o:o+stride], newV[o:o+stride])
		}
		return
	}
	effK := min(K, r.W)
	effStart := startPos + K - effK
	srcOffset := (K - effK) * stride
	slot0 := effStart % r.W
	seg1 := min(effK, r.W-slot0)
	mir := 0 // offset of the mirror copy, or 0 when there is none
	if r.Mirrored() {
		mir = r.W * r.Stride
	}
	if seg1 > 0 {
		len1 := seg1 * stride
		copy(r.K[slot0*stride:(slot0+seg1)*stride], newK[srcOffset:srcOffset+len1])
		copy(r.V[slot0*stride:(slot0+seg1)*stride], newV[srcOffset:srcOffset+len1])
		if mir > 0 {
			copy(r.K[mir+slot0*stride:mir+(slot0+seg1)*stride], newK[srcOffset:srcOffset+len1])
			copy(r.V[mir+slot0*stride:mir+(slot0+seg1)*stride], newV[srcOffset:srcOffset+len1])
		}
	}
	if seg2 := effK - seg1; seg2 > 0 {
		len1 := seg1 * stride
		len2 := seg2 * stride
		copy(r.K[:len2], newK[srcOffset+len1:srcOffset+len1+len2])
		copy(r.V[:len2], newV[srcOffset+len1:srcOffset+len1+len2])
		if mir > 0 {
			copy(r.K[mir:mir+len2], newK[srcOffset+len1:srcOffset+len1+len2])
			copy(r.V[mir:mir+len2], newV[srcOffset+len1:srcOffset+len1+len2])
		}
	}
	if startPos+K > r.Count {
		r.Count = startPos + K
	}
}

// isLocal reports whether layer is a ring (sliding-window) layer.
func (c *Cache) isLocal(layer int) bool { return c.Rings[layer] != nil }

// StoredRows is the logical number of positions held for a layer: the ring's
// count (it physically keeps only W) for a local layer, else the append-forever
// length / width.
func (c *Cache) StoredRows(layer, kvDim int) int {
	if r := c.Rings[layer]; r != nil {
		return r.Count
	}
	if c.Quant == KVI8 {
		return len(c.KeysQ[layer]) / kvDim
	}
	if c.Quant == KVQF16 {
		return len(c.KeysH[layer]) / c.RowStride(kvDim)
	}
	return len(c.KeysBuf[layer]) / c.RowStride(kvDim)
}

// WindowStart returns the first key index a query at absolute position pos
// may attend to on a local (sliding-window) layer. Gemma's window of size W
// admits keys j with pos−W < j ≤ pos, i.e. the W most recent keys
// [pos−W+1, pos] — matching HF's sliding mask (key j attends iff pos−j < W).
// Global layers (and an unset window) attend from 0.
//
// Takes pos explicitly rather than reading c.PosN: within one forward, c.PosN
// only advances on the last layer's Append, so reading it post-Append would
// shift the window by one on that layer alone. The query position is stable.
func (c *Cache) WindowStart(pos int, global bool) int {
	if global {
		return 0
	}
	return kvmgr.WindowNeedStart(pos, c.Window)
}

// SetImageBlocks marks position ranges [start,end) (e.g. an image's token block)
// as bidirectional for batched-prefill attention: a query inside a block sees the
// whole block, not just the causal prefix. nil/empty ⇒ fully causal (the
// text-only default). Multimodal prefill sets this; see docs/multimodal.md §5a.
func (c *Cache) SetImageBlocks(blocks [][2]int) { c.ImgBlocks = blocks }

// attendHi returns the inclusive upper key bound for a query at pos: pos for a
// causal (text) position, or end-1 if pos lies in a bidirectional image block
// [start,end) — so [start,pos] ∪ (pos,end) = [start,end) becomes visible. With no
// image blocks it always returns pos, so the seam is inert for text-only decode.
func (c *Cache) attendHi(pos int) int {
	for _, b := range c.ImgBlocks {
		if pos >= b[0] && pos < b[1] {
			return b[1] - 1
		}
	}
	return pos
}

// DeepstackRows is Cache.deepstack: the image runs (absolute, in prompt order; several for a multi-image turn) and
// one set per early layer, each holding every run's rows concatenated in run order.
type DeepstackRows struct {
	Spans []ImageSpan
	Rows  [][]float32
}

func layerStates[T any](n int, match func(int) bool, makeOne func() *T) []*T {
	out := make([]*T, n)
	for l := range n {
		if match(l) {
			out[l] = makeOne()
		}
	}
	return out
}

// NewCacheForArch allocates and configures a Cache for architecture a.
func NewCacheForArch(a *darch.Architecture, capHint int, kvI8, kvH bool, knobs KnobSource) *Cache {
	kvCapHint := capHint
	if a.MLA != nil {
		kvCapHint = 0
	}
	quant := KVQF32
	if kvI8 && a.KVInt8OK() {
		quant = KVI8
	} else if kvH && a.KVF16OK() {
		quant = KVQF16
	}
	if quant != KVQF32 {
		kvCapHint = 0
	}
	c := NewCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, kvCapHint, func(l int) bool {
		return a.HasNoAttentionKVAt(l) || a.RingLayerAt(l) || a.Gemma4KVSrcAt(l) != l
	})
	c.ReserveWidths(kvCapHint, a.KVDimAt)
	c.Knobs, c.Scr = knobs, NewDecodeScratch(a, knobs)
	if quant != KVQF32 {
		c.SetQuant(quant, capHint)
	}
	c.EnableRings(a.SlidingWindow, func(l int) bool { return !a.RingLayerAt(l) })
	defer c.SetRowPad(a, kvCapHint)
	if a.Gemma4 != nil {
		c.ManualPos = true
	}
	if a.Qwen35 != nil {
		c.ManualPos = true
		c.Delta = layerStates(a.NumLayers, a.IsLinearLayer, func() *DeltaState { return NewDeltaState(*a.Qwen35) })
	}
	if a.LFM2 != nil {
		c.ManualPos = true
		c.Conv = layerStates(a.NumLayers, a.IsConvLayer, NewShortConvState)
	}
	if a.Granite != nil {
		c.ManualPos = true
		mp := archparts.Mamba2Params{NHeads: a.Granite.NHeads, HeadDim: a.Granite.HeadDim, DState: a.Granite.DState, NGroups: a.Granite.NGroups, DConv: a.Granite.DConv, Hidden: a.HiddenDim}
		c.Mamba = layerStates(a.NumLayers, a.IsMambaLayer, func() *Mamba2State { return NewMamba2State(mp) })
	}
	if a.Nemotron != nil {
		c.ManualPos = true
		mp := archparts.Mamba2Params{NHeads: a.Nemotron.NHeads, HeadDim: a.Nemotron.HeadDim, DState: a.Nemotron.DState, NGroups: a.Nemotron.NGroups, DConv: a.Nemotron.DConv, Hidden: a.HiddenDim}
		c.Mamba = layerStates(a.NumLayers, func(l int) bool { return a.Nemotron.BlockKind[l] == archparts.NemoMamba }, func() *Mamba2State { return NewMamba2State(mp) })
	}
	if a.MLA != nil {
		c.ManualPos = true
		c.MLALatent = make([][]float32, a.NumLayers)
		for l := range c.MLALatent {
			if d := a.KVDimAt(l); d > 0 && capHint > 0 {
				c.MLALatent[l] = make([]float32, 0, capHint*d)
			}
		}
	}
	if a.KDA != nil {
		c.ManualPos = true
		c.KDA = layerStates(a.NumLayers, a.IsLinearLayer, func() *KDAState { return NewKDAState(*a.KDA) })
	}
	if a.Inkling2 != nil {
		// Every visit holds KV and four conv windows; manualPos because walkToken advances once per token after the visits.
		c.ManualPos = true
		c.Ink = layerStates(a.NumLayers, func(int) bool { return true }, func() *inkling2.ConvState { return new(inkling2.ConvState) })
	}
	return c
}
