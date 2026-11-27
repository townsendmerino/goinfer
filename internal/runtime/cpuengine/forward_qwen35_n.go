package cpuengine

import (
	"context"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// Qwen35BatchN The batched Qwen3.5 forward is the batched walker over the family's program, for PromptHidden (Route B's hidden state), prefix sharing and the image prefill. The per-token forward reads every weight once
// per token; here every projection is one M=K Matmul, so a weight is read once per prompt. The DeltaNet's conv, gates, recurrence
// and gated norm stay sequential per token (deltaNetBatch), and attention runs with f64 accumulation (gatedAttnBatch).
//
// It is not claimed bit-identical to the per-token forward: a batched f32 or quantized Matmul may reduce in a different order from
// its matvec. TestPromptHidden_batchedMatchesSequential bounds the difference.
//
// qwen35BatchN reports whether it applies. Olmo Hybrid (PlainFullAttn, per-layer norm placement), a capture request and a dense
// layer pager take the per-token path.
func (m *Engine) Qwen35BatchN(K int, cache *Cache) bool {
	return cache.MRopePos == nil && m.Qwen35BatchNAnyPos(K, cache)
}

// Qwen35BatchNAnyPos is qwen35BatchN without its m-RoPE exclusion: the image prefill (prefillLogitsQwen35VL) runs this forward
// over a cache that carries m-RoPE positions, which runLayersQwen35N rotates the full-attention layers' q and k by. qwen35BatchN
// keeps the exclusion for its other callers (prefix sharing documents "no m-RoPE" as a precondition).
func (m *Engine) Qwen35BatchNAnyPos(K int, cache *Cache) bool {
	a := m.W.Arch
	return K > 1 && a.Qwen35 != nil && !a.Qwen35.PlainFullAttn && a.NormPlacementLinear == nil &&
		a.NormPlacement == darch.NormPre2 && m.LayerPager == nil && cache.CaptureLayers == nil && cache.TreeMask == nil &&
		!cache.LocalAny && m.W.Embed.Rows() != 0 &&
		(a.MoE != nil || a.Act == darch.ActSiLU)
}

// RunLayersQwen35N runs K embedded rows h [K*HiddenDim] through every layer on the batched walker, advances cache by K, and
// returns the post-final-norm hidden states [K*HiddenDim]. h is consumed. The caller checks qwen35BatchN first.
func (m *Engine) RunLayersQwen35N(reqCtx context.Context, h []float32, cache *Cache) ([]float32, error) {
	return m.WalkBatch(reqCtx, m.Program(), h, cache, false, len(h)/m.W.Arch.HiddenDim, nil, nil)
}
