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
	h      []float32 // [hidden] residual stream (overwritten by the embedding each step)
	norm   []float32 // [hidden] normalized layer input (pre-attn, then pre-mlp)
	sub    []float32 // [hidden] attention output, then MLP output (added to h)
	q      []float32 // [qDim]
	k, v   []float32 // [kvDim]
	ctx    []float32 // [qDim] attention context before the O-projection
	gate   []float32 // [inter] gatedMLP gate
	up     []float32 // [inter] gatedMLP up
	scores []float32 // [>=nKeys] attention scores; grown on demand as context extends
	logits []float32 // [vocab]

	ws        *linalg.Workspace // W8A8 activation-quant scratch (zero-alloc Into/Batch)
	qkvOps    [3]linalg.W8A8Op  // reused q/k/v batch ops
	gateUpOps [2]linalg.W8A8Op  // reused gate/up batch ops
}

func newDecodeScratch(a *Architecture) *decodeScratch {
	qDim, kvDim := a.NumHeads*a.HeadDim, a.NumKVHeads*a.HeadDim
	// Note: aikit's opt-in worker pool (Workspace.SetWorkers) is intentionally NOT
	// used — goinfer's end-to-end sweep showed it neutral-to-slightly-slower than
	// the spawn path (the batch=1 fork/join cost is a floor, not pool-fixable).
	return &decodeScratch{
		h:      make([]float32, a.HiddenDim),
		norm:   make([]float32, a.HiddenDim),
		sub:    make([]float32, a.HiddenDim),
		q:      make([]float32, qDim),
		k:      make([]float32, kvDim),
		v:      make([]float32, kvDim),
		ctx:    make([]float32, qDim),
		gate:   make([]float32, a.IntermediateDim),
		up:     make([]float32, a.IntermediateDim),
		logits: make([]float32, a.VocabSize),
		ws:     &linalg.Workspace{},
	}
}

// scoresBuf returns a length-n scores buffer, reusing the backing array when it
// is large enough and growing it (once) as the context extends.
func (s *decodeScratch) scoresBuf(n int) []float32 {
	if cap(s.scores) < n {
		s.scores = make([]float32, n)
	}
	return s.scores[:n]
}
