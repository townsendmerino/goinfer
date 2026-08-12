package decoder

import "math"

// Llama 4 (llama4_text) forward path — the iRoPE text decoder. Each block is a standard
// Pre2 residual stack (input_layernorm → attention → +residual → post_attention_layernorm →
// FFN → +residual), but two things vary PER LAYER:
//
//   - Attention: RoPE layers (no_rope_layers[l]==1) apply interleaved RoPE over the full
//     head_dim then a parameter-free L2 (RMS-over-head-dim) QK-norm; NoPE layers skip RoPE
//     and instead scale the query by an attention "temperature"
//     (log1p(floor((pos+1)/floor_scale))·attn_scale + 1) for length generalization.
//   - FFN: dense layers vs MoE layers (moe_layers) — handled by the shared mlp() dispatch
//     (dense ⇒ gatedMLP; MoE ⇒ moeMLP with top-1 sigmoid routing + an ungated shared expert).
//
// Chunked (local) attention on the RoPE layers reduces to full causal for sequences below
// attention_chunk_size, which the parity gates use — so this attends full-causal. Parity-
// first f32, one token per call; canBatchN excludes it.
func (m *Model) runLayersLlama4(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos()
	if cache.scr == nil {
		cache.scr = newDecodeScratch(arch)
	}
	h := make([]float32, hidden)
	m.w.Embed.Row(id, h)

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		attn := m.llama4Attention(n, lw, arch, cache, l, pos)
		for i := range h {
			h[i] += attn[i]
		}
		n2 := append([]float32(nil), h...)
		rmsNorm(n2, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
		ffn := make([]float32, hidden)
		if arch.llama4.isMoE[l] {
			m.llama4MoE(n2, ffn, lw, arch) // input-scaled experts + ungated shared (Llama 4-specific)
		} else if err := mlp(n2, ffn, lw, arch, m.be, cache.scr, m.pager, nil); err != nil {
			return nil, err // dense layer → gatedMLP
		}
		for i := range h {
			h[i] += ffn[i]
		}
	}
	// No explicit Advance: every layer runs standard attention + cache.Append, so the
	// last layer's Append auto-advances pos (like llama/qwen). manualPos is NOT set.
	return h, nil
}

// llama4Attention is one Llama 4 attention layer: GQA with the per-layer iRoPE behavior.
// RoPE layers rope (interleaved, full head_dim) then L2-normalize q/k per head; NoPE layers
// scale q by the attention temperature instead. Standard causal attention over the KV cache
// at 1/√head_dim.
func (m *Model) llama4Attention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	lp := arch.llama4
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	hidden := arch.HiddenDim
	eps := arch.NormEps
	q := make([]float32, nH*hd)
	k := make([]float32, nKV*hd)
	v := make([]float32, nKV*hd)
	matmul(m.be, &lw.QProj, n, q, 1)
	matmul(m.be, &lw.KProj, n, k, 1)
	matmul(m.be, &lw.VProj, n, v, 1)

	if lp.useRope[layer] {
		invFreq := arch.ropeInvFreq(layer)
		mlaRope(q, nH, hd, pos, invFreq, 1, true) // interleaved (complex) RoPE over the full head_dim
		mlaRope(k, nKV, hd, pos, invFreq, 1, true)
		if lp.useQKNorm { // parameter-free L2 (RMS-over-head-dim) QK-norm, AFTER rope (HF order)
			l2NormHeads(q, nH, hd, eps)
			l2NormHeads(k, nKV, hd, eps)
		}
	} else if lp.attnTemp { // NoPE layers: attention-temperature tuning on the query
		scale := float32(math.Log1p(math.Floor(float64(pos+1)/lp.floorScale))*lp.attnScale + 1.0)
		for i := range q {
			q[i] *= scale
		}
	}

	cache.Append(layer, k, v)
	ctx := make([]float32, nH*hd)
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full causal*/, arch)

	out := make([]float32, hidden)
	matmul(m.be, &lw.OProj, ctx, out, 1)
	return out
}

// llama4MoE is Llama 4's mixture-of-experts FFN. Unlike the standard moeMLP (which scales
// each expert's OUTPUT by its router weight), Llama 4 scales the expert's INPUT by the
// router score — routed_in = score·h, then expert(routed_in) — and since the expert is
// SwiGLU (nonlinear), input-scaling ≠ output-scaling, so it gets its own path. The score is
// sigmoid(logit) of the top-k experts (top-1 by default); the always-on shared expert runs
// on the unscaled h and is added ungated. out = shared(h) + Σ_topk expert_e(sigmoid(logit_e)·h).
func (m *Model) llama4MoE(h, out []float32, lw *LayerWeights, arch *Architecture) {
	moe := arch.MoE
	hidden := arch.HiddenDim
	logits := make([]float32, moe.NumExperts)
	matmul(m.be, &lw.Router, h, logits, 1)
	idx, _ := topK(logits, moe.TopK) // top-k by raw logit (== top-k by sigmoid, monotonic)

	// P6: one gate/up pair for the token, shared by the shared expert and every routed one. They
	// run sequentially, so k pairs were never simultaneously live.
	sc := max(moe.SharedIntermediateDim, moe.IntermediateDim)
	l4gate, l4up := make([]float32, sc), make([]float32, sc)

	// Shared expert on the unscaled input, ungated.
	swiGLUExpert(&lw.SharedExpert, h, out, moe.SharedIntermediateDim, m.be, l4gate, l4up)

	scaled := make([]float32, hidden)
	expOut := make([]float32, hidden)
	for _, e := range idx {
		w := sigmoidf(logits[e]) // router weight = sigmoid of the (unbiased) logit
		for i := range h {
			scaled[i] = w * h[i]
		}
		swiGLUExpert(&lw.Experts[e], scaled, expOut, moe.IntermediateDim, m.be, l4gate, l4up)
		for i := range out {
			out[i] += expOut[i]
		}
	}
}

// l2NormHeads applies Llama 4's parameter-free QK-norm — a per-head RMS over head_dim with
// no learned weight: x ← x · rsqrt(mean(x²) + eps). vec is [heads, headDim] flattened.
func l2NormHeads(vec []float32, heads, headDim int, eps float64) {
	for hh := range heads {
		seg := vec[hh*headDim : (hh+1)*headDim]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		scale := 1.0 / math.Sqrt(ss/float64(headDim)+eps)
		for i := range seg {
			seg[i] = float32(float64(seg[i]) * scale)
		}
	}
}
