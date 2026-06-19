package decoder

import (
	"os"
	"strconv"
)

// Granite-4.0-H (granitemoehybrid) forward path — the hybrid: each layer is either a
// Mamba-2 selective-scan mixer (recurrent state in the cache) or GQA softmax
// attention (KV cache), every layer a routed+shared MoE. Four Granite scalars apply:
// embedding_multiplier scales the embedding, attention_multiplier is the attention
// q·k scale (arch.AttnScale), residual_multiplier scales every residual add, and
// logits_scaling divides the final logits (applied at the shared head). Parity-first
// f32, allocate-per-call, one token per call (prefill drives it sequentially so the
// Mamba-2 recurrence sees every token), mirroring runLayersQwen35.
func (m *Model) runLayersGranite(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	g := arch.granite
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos()
	mp := mamba2Params{
		NHeads: g.NHeads, HeadDim: g.HeadDim, DState: g.DState,
		NGroups: g.NGroups, DConv: g.DConv, Hidden: hidden, NormGroups: 1,
	}

	embMul, residMul := g.EmbMul, g.ResidMul
	if os.Getenv("GOINFER_SSM_NOMUL") != "" { // isolation seam (matches GraniteResidentParams)
		embMul, residMul = 1, 1
	}
	h := make([]float32, hidden)
	m.w.Embed.Row(id, h)
	for i := range h {
		h[i] *= embMul // embedding_multiplier
	}

	stopLayer := -1 // GOINFER_SSM_STOP_LAYER debug: truncate to localize the resident SSM bug
	if v := os.Getenv("GOINFER_SSM_STOP_LAYER"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			stopLayer = n
		}
	}
	for l := 0; l < arch.NumLayers; l++ {
		if stopLayer >= 0 && l > stopLayer {
			break
		}
		lw := &m.w.Layers[l]

		// Sequence-mixer sub-block (Pre2: norm → mix → ×residual_multiplier → add).
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		var mix []float32
		if arch.isMambaLayer(l) {
			mix = mamba2Step(n, lw.mamba, mp, eps, cache.mamba[l])
		} else {
			mix = m.graniteAttention(n, lw, arch, cache, l, pos)
		}
		for i := range h {
			h[i] += mix[i] * residMul
		}

		// MoE FFN sub-block (Pre2). post_attention_layernorm is the pre-MLP norm.
		if os.Getenv("GOINFER_SSM_SKIPFFN") == "" {
			n2 := append([]float32(nil), h...)
			rmsNorm(n2, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
			ffn, err := moeMLP(n2, lw, arch, m.be, m.pager)
			if err != nil {
				return nil, err
			}
			for i := range h {
				h[i] += ffn[i] * residMul
			}
		}
	}
	cache.Advance() // manualPos: the mamba layers never Append, so step pos once per token
	return h, nil
}

// graniteAttention is a plain GQA softmax attention layer: q/k/v/o projections, full
// rotary RoPE, causal attention over the KV cache scaled by attention_multiplier
// (arch.AttnScale) — no QK-norm, no bias, no output gate.
func (m *Model) graniteAttention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	hidden := arch.HiddenDim
	q := make([]float32, nH*hd)
	k := make([]float32, nKV*hd)
	v := make([]float32, nKV*hd)
	matmul(m.be, &lw.QProj, n, q, 1)
	matmul(m.be, &lw.KProj, n, k, 1)
	matmul(m.be, &lw.VProj, n, v, 1)

	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)
	applyRoPE(q, nH, hd, pos, invFreq, ms)
	applyRoPE(k, nKV, hd, pos, invFreq, ms)

	cache.Append(layer, k, v)
	ctx := make([]float32, nH*hd)
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full attention*/, arch)

	out := make([]float32, hidden)
	matmul(m.be, &lw.OProj, ctx, out, 1)
	return out
}
