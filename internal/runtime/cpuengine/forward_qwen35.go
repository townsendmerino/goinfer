package cpuengine

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

import "github.com/townsendmerino/goinfer/internal/runtime/hybridattn"

// The Qwen3.5/3.6 family (qwen3_5_moe, qwen3_5, qwen3_next, olmo_hybrid), a hybrid: most layers are Gated DeltaNet (linear
// attention, recurrent state in the cache), the rest gated softmax attention (KV cache), each followed by a routed+shared MoE or
// a dense SwiGLU. Its per-token program is the generic layer with Architecture.mixerOps' mixers; runLayersQwen35N is the batched
// twin.

// qwen35AttentionWith is the gated softmax attention of a full-attention layer (opGatedAttn):
// double-width q_proj (query ‖ gate per head), per-head QK-norm, partial RoPE,
// causal GQA attention over the KV cache, then × sigmoid(gate) before o_proj.
func (m *Engine) qwen35AttentionWith(p *gatedAttnParams, n []float32, lw *darch.LayerWeights, arch *darch.Architecture, cache *Cache, layer, pos int) []float32 {
	a := lw.QAttn
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim

	// q_proj emits [query ‖ gate] per head; split them.
	cache.Scr.hq.Prepare(m.Be, &a.QProj, n, 1) // n quantized once for q, k and v
	qg := MatvecWMPre(m.Be, &a.QProj, &cache.Scr.hq, n)
	q := make([]float32, nH*hd)
	gate := make([]float32, nH*hd)
	hybridattn.SplitQGate(qg, q, gate, nH, hd)
	k := MatvecWMPre(m.Be, &a.KProj, &cache.Scr.hq, n)
	v := MatvecWMPre(m.Be, &a.VProj, &cache.Scr.hq, n)

	// QK-norm (query half only) then partial RoPE; rope.At is rope.Apply unless the cache carries m-RoPE positions (an image turn).
	applyQKNorm(p.QKNorm, q, k, a.QNorm, a.KNorm, nH, nKV, hd)
	applyRope(&p.Rope, q, k, nH, nKV, hd, pos, cache.MRopePos, cache.MRopeDelta)

	cache.Append(layer, k, v)
	ctx := cache.Scr.ctxBuf(nH * hd) // from the scratch; attendQueryP clears it
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	AttendQueryP(p.Attn, q, ctx, cache.Scr.ScoresBuf(nKeys), cache, layer, pos, arch)

	// Output gate, then o_proj.
	hybridattn.QGateContext(ctx, gate)
	return MatvecWM(m.Be, &a.OProj, ctx)
}
