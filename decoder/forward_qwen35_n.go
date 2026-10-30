package decoder

import (
	"context"
)

// The batched Qwen3.5 forward is the batched walker over the family's program, for PromptHidden (Route B's hidden state,
// docs/tasks/task-constrained-confidence.md), prefix sharing and the image prefill. The per-token forward reads every weight once
// per token; here every projection is one M=K matmul, so a weight is read once per prompt. The DeltaNet's conv, gates, recurrence
// and gated norm stay sequential per token (deltaNetBatch), and attention runs with f64 accumulation (gatedAttnBatch).
//
// It is not claimed bit-identical to the per-token forward: a batched f32 or quantized matmul may reduce in a different order from
// its matvec. TestPromptHidden_batchedMatchesSequential bounds the difference.
//
// qwen35BatchN reports whether it applies. Olmo Hybrid (PlainFullAttn, per-layer norm placement), a capture request and a dense
// layer pager take the per-token path.
func (m *Model) qwen35BatchN(K int, cache *KVCache) bool {
	return cache.mropePos == nil && m.qwen35BatchNAnyPos(K, cache)
}

// qwen35BatchNAnyPos is qwen35BatchN without its m-RoPE exclusion: the image prefill (prefillLogitsQwen35VL) runs this forward
// over a cache that carries m-RoPE positions, which runLayersQwen35N rotates the full-attention layers' q and k by. qwen35BatchN
// keeps the exclusion for its other callers (prefix sharing documents "no m-RoPE" as a precondition).
func (m *Model) qwen35BatchNAnyPos(K int, cache *KVCache) bool {
	a := m.w.arch
	return K > 1 && a.qwen35 != nil && !a.qwen35.PlainFullAttn && a.NormPlacementLinear == nil &&
		a.NormPlacement == NormPre2 && m.layerPager == nil && cache.captureLayers == nil && cache.treeMask == nil &&
		!cache.localAny && m.w.Embed.Rows() != 0 &&
		(a.MoE != nil || a.Act == ActSiLU)
}

// runLayersQwen35N runs K embedded rows h [K*HiddenDim] through every layer on the batched walker, advances cache by K, and
// returns the post-final-norm hidden states [K*HiddenDim]. h is consumed. The caller checks qwen35BatchN first.
func (m *Model) runLayersQwen35N(reqCtx context.Context, h []float32, cache *KVCache) ([]float32, error) {
	return m.walkBatch(reqCtx, m.program(), h, cache, false, len(h)/m.w.arch.HiddenDim, nil, nil)
}
