package cpuengine

// CacheVision holds per-sequence multimodal attention and position state: bidirectional image token blocks,
// Qwen2.5-VL 3D m-RoPE positions and compression delta, and Qwen3-VL DeepStack residual injections.
type CacheVision struct {
	ImgBlocks  [][2]int // multimodal: position ranges [start,end) that attend bidirectionally (image blocks); nil = all-causal
	MRopePos   [][3]int // Qwen2.5-VL m-RoPE: per-sequence-position (t,h,w) rotary positions; nil = scalar RoPE. Indexed by absolute sequence position.
	MRopeDelta int      // m-RoPE position delta: decode positions past the prefill are scalar seqPos+rope.MRoPEDelta (the image compresses positions, so delta is usually negative).
	// Deepstack is Qwen3-VL's DeepStack for the prefill that carries it: after decoder layer l, for l < len(rows), rows[l]
	// ([n*hidden], one row per image position) is added to the hidden state at the image positions [start, start+n), as HF's
	// _deepstack_process does. nil for every other prefill; a decode step never touches it.
	Deepstack *DeepstackRows
}

// DeepstackRows is CacheVision.Deepstack: the image runs (absolute, in prompt order; several for a multi-image turn) and
// one set per early layer, each holding every run's rows concatenated in run order.
type DeepstackRows struct {
	Spans []ImageSpan
	Rows  [][]float32
}

// ResetMultimodal clears the per-sequence multimodal state (image attention blocks, Qwen2.5-VL m-RoPE positions and delta) so a
// reused cache does not leak a prior sequence's image blocks or offsets into a fresh one. Like recurrent state it has no
// per-position rewind, so it is cleared only on a full reset. No-op on text-only caches.
func (v *CacheVision) ResetMultimodal() {
	v.ImgBlocks = nil
	v.MRopePos = nil
	v.MRopeDelta = 0
}

// SetImageBlocks marks position ranges [start,end) (e.g. an image's token block)
// as bidirectional for batched-prefill attention: a query inside a block sees the
// whole block, not just the causal prefix. nil/empty => fully causal (the
// text-only default). Multimodal prefill sets this; see docs/multimodal.md §5a.
func (v *CacheVision) SetImageBlocks(blocks [][2]int) { v.ImgBlocks = blocks }

// prefillCut returns where a pass of a prefill starting at position start should end, given its rows [lo, hi): hi, unless that
// falls inside an image block, which a pass must hold whole (its rows attend to each other). The cut then moves back to the
// block's start, or forward to its end when the block starts at or before lo.
func (v *CacheVision) prefillCut(start, lo, hi int) int {
	return CutOutsideBlocks(v.ImgBlocks, start, lo, hi)
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

// attendHi returns the inclusive upper key bound for a query at pos: pos for a
// causal (text) position, or end-1 if pos lies in a bidirectional image block
// [start,end) — so [start,pos] U (pos,end) = [start,end) becomes visible. With no
// image blocks it always returns pos, so the seam is inert for text-only decode.
func (v *CacheVision) attendHi(pos int) int {
	for _, b := range v.ImgBlocks {
		if pos >= b[0] && pos < b[1] {
			return b[1] - 1
		}
	}
	return pos
}
