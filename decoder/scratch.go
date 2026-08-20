package decoder

import "github.com/townsendmerino/aikit/linalg"

// decodeScratch holds the per-stream float buffers the single-token forward
// reuses across decode steps, so steady-state decode allocates ~nothing per
// layer (Phase 1 of the perf campaign). One lives on each KVCache — a cache is
// one generation stream, so the buffers are never shared concurrently.
//
// Liveness (why these few buffers suffice): within a layer, `norm` is the
// normalized input to attention, then re-derived as the input to the MLP
// (sequential, no overlap); `sub` is the attention output added to the residual,
// then the MLP output (also sequential). The residual `h` persists across the
// layer loop; the rest are local to one matmul/attention.
type decodeScratch struct {
	h        []float32 // [hidden] residual stream (overwritten by the embedding each step)
	norm     []float32 // [hidden] normalized layer input (pre-attn, then pre-mlp)
	sub      []float32 // [hidden] attention output, then MLP output (added to h)
	sub2     []float32 // [hidden] parallel-block (Cohere) MLP output, held while attn output lives in sub
	q        []float32 // [qDim]
	k, v     []float32 // [kvDim]
	ctx      []float32 // [qDim] attention context before the O-projection
	gate     []float32 // [inter] gatedMLP gate
	up       []float32 // [inter] gatedMLP up
	scores   []float32 // [>=nKeys] attention scores; grown on demand as context extends
	attnGate []float32 // [>=nH or nH*hd] Laguna attention output-gate scratch (g_proj result, pre-softplus); nil for every other family
	logits   []float32 // [vocab]

	// Gather buffers for the MoE decode path, which routes single-token attention
	// through attendBatchedHeads (K=1) so it uses the SAME acc64 matmul as the
	// batched MoE prefill — bit-identical, so the discrete router never sees a
	// prefill↔decode numerical discontinuity. akh/avt grow with context like scores.
	aqh, akh, avt, ach []float32

	// localK/localV assemble a local (ring) layer's [base, pos] read window for the
	// K=1 MoE decode path (resident ring history + this token's K/V); ≤ W rows.
	localK, localV []float32

	ws        *linalg.Workspace // W8A8 activation-quant scratch (zero-alloc Into/Batch)
	qkvOps    [3]linalg.W8A8Op  // reused q/k/v batch ops
	gateUpOps [2]linalg.W8A8Op  // reused gate/up batch ops

	loraTmp []float32 // [>=r] compute-time LoRA A·x scratch (#7); nil until an adapter is active

	// MoE decode scratch (mlp.go moeMLP): eliminates the router-logits/accumulator/
	// expert-gate-up allocations moeMLP otherwise makes every call, layer after
	// layer — nE+2·hidden+2·sc floats per call, the bulk of decode's per-token
	// allocation on MoE families. nil for non-MoE architectures. moeMLP falls back
	// to allocating when passed a nil scr (the batched-prefill call site in
	// forwardn.go, which builds its own per-K-batch scratch and has no cache.scr).
	moeLogits         []float32 // [NumExperts]
	moeOut, moeExpOut []float32 // [HiddenDim]
	moeGate, moeUp    []float32 // [max(IntermediateDim, SharedIntermediateDim)]
}

// loraBuf returns a length-r scratch for the compute-time LoRA A·x intermediate,
// growing the (per-stream, reused) backing array once on demand.
func (s *decodeScratch) loraBuf(r int) []float32 {
	if cap(s.loraTmp) < r {
		s.loraTmp = make([]float32, r)
	}
	return s.loraTmp[:r]
}

func newDecodeScratch(a *Architecture) *decodeScratch {
	// maxHeads, not NumHeads: a family with per-layer query heads (Laguna) must size
	// q/ctx for its WIDEST layer, since one scratch is reused across every layer.
	qDim, kvDim := a.maxHeads()*a.HeadDim, a.NumKVHeads*a.HeadDim
	// Note: aikit's opt-in worker pool (Workspace.SetWorkers) is intentionally NOT
	// used — goinfer's end-to-end sweep showed it neutral-to-slightly-slower than
	// the spawn path (the batch=1 fork/join cost is a floor, not pool-fixable).
	//
	// The W8A8 (int8) decode matmuls run through this Workspace (matmulInto), so its
	// PER-WORKSPACE threshold is what makes int8 decode parallelize — NOT the process
	// global. Setting it here means every decode stream (library Load, serve, tests)
	// gets it automatically and race-free, the same way the int4 path self-configures
	// per-call (weightmat.go int4ParThreshold). See tune.go DefaultDecodeParallelThreshold.
	ws := &linalg.Workspace{}
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	s := &decodeScratch{
		h:      make([]float32, a.HiddenDim),
		norm:   make([]float32, a.HiddenDim),
		sub:    make([]float32, a.HiddenDim),
		sub2:   make([]float32, a.HiddenDim),
		q:      make([]float32, qDim),
		k:      make([]float32, kvDim),
		v:      make([]float32, kvDim),
		ctx:    make([]float32, qDim),
		gate:   make([]float32, a.IntermediateDim),
		up:     make([]float32, a.IntermediateDim),
		logits: make([]float32, a.VocabSize),
		ws:     ws,
	}
	if a.MoE != nil {
		sc := max(a.MoE.SharedIntermediateDim, a.MoE.IntermediateDim)
		s.moeLogits = make([]float32, a.MoE.NumExperts)
		s.moeOut = make([]float32, a.HiddenDim)
		s.moeExpOut = make([]float32, a.HiddenDim)
		s.moeGate = make([]float32, sc)
		s.moeUp = make([]float32, sc)
	}
	return s
}

// scoresBuf returns a length-n scores buffer, reusing the backing array when it
// is large enough and growing it (once) as the context extends.
func (s *decodeScratch) scoresBuf(n int) []float32 {
	if cap(s.scores) < n {
		s.scores = make([]float32, n)
	}
	return s.scores[:n]
}

// gateBuf returns a length-n buffer for Laguna's attention output gate (the
// g_proj result, before softplus), grown once on demand. n is nH or nH*hd
// depending on the gate's granularity, so it is sized from the weight's row
// count rather than assumed.
func (s *decodeScratch) gateBuf(n int) []float32 {
	if cap(s.attnGate) < n {
		s.attnGate = make([]float32, n)
	}
	return s.attnGate[:n]
}

// attnBatchBufs returns the K=1 scratch slices attendBatchedHeads needs for the
// MoE decode path (qh[hd], kh[nKeys*hd], vt[nKeys*hd], scores[nKeys], ch[hd]),
// growing the context-sized backings once as decode extends.
func (s *decodeScratch) attnBatchBufs(nKeys, hd int) (qh, kh, vt, scores, ch []float32) {
	if cap(s.aqh) < hd {
		s.aqh = make([]float32, hd)
		s.ach = make([]float32, hd)
	}
	if n := nKeys * hd; cap(s.akh) < n {
		g := max(2*cap(s.akh), n) // headroom: nKeys grows by 1 each decode step, so exact-fit reallocs+zeroes this multi-KB buffer every token
		s.akh = make([]float32, g)
		s.avt = make([]float32, g)
	}
	return s.aqh[:hd], s.akh[:nKeys*hd], s.avt[:nKeys*hd], s.scoresBuf(nKeys), s.ach[:hd]
}

// localBufs returns two length-n scratch slices (grown once on demand) for
// assembling a local layer's contiguous read window in the K=1 MoE decode path.
func (s *decodeScratch) localBufs(n int) (lk, lv []float32) {
	if cap(s.localK) < n {
		g := max(2*cap(s.localK), n) // headroom: window grows each step; avoid a per-token realloc+zero
		s.localK = make([]float32, g)
		s.localV = make([]float32, g)
	}
	return s.localK[:n], s.localV[:n]
}
