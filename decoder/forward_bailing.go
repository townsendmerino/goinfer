package decoder

// Bailing Hybrid (Ling 3.0, model_type "bailing_hybrid") forward path — Multi-head Latent
// Attention alternating with Kimi Delta Attention (KDA), every LayerGroupSize-th layer MLA and
// the rest KDA, over a DeepSeekMoE FFN. Both mixers are reused, not reimplemented: MLA via
// forward_deepseek.go's own mlaAttention (parameterized for this family's own tensor-name prefix
// and optional output gate — see mlaParams' own comment), KDA via kda.go's kdaMixerStep (the one
// genuinely new primitive, a per-channel-decay delta rule — see kdaParams' own comment). The
// block is a standard Pre2 residual stack, identical for both mixer kinds — verified against the
// real modeling_bailing_moe_v3.py's BailingMoeV3DecoderLayer.forward, which applies
// input_layernorm/post_attention_layernorm the SAME way regardless of attention_layer_type
// (unlike Olmo Hybrid, which needed NormPlacementLinear for exactly this reason) — so the FFN
// reuses the generic mlp() dispatch (dense prefix on l < first_k_dense_replace, MoE elsewhere).
// Parity-first f32, one token per call; canBatchN excludes both the MLA and KDA attention kinds.
func (m *Model) runLayersBailingHybrid(id int, cache *KVCache) ([]float32, error) {
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

		// Attention sub-block (Pre2).
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		var attn []float32
		if arch.isLinearLayer(l) {
			attn = kdaMixerStep(m.be, n, lw.kda, *arch.kda, hidden, eps, cache.kda[l])
		} else {
			attn = m.mlaAttention(n, lw, arch, cache, l, pos)
		}
		for i := range h {
			h[i] += attn[i]
		}

		// FFN sub-block (Pre2). mlp() routes dense (l < FirstKDense, Experts nil) vs MoE.
		n2 := append([]float32(nil), h...)
		rmsNorm(n2, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
		ffn := make([]float32, hidden)
		if err := mlp(n2, ffn, lw, arch, m.be, cache.scr, m.pager, nil); err != nil {
			return nil, err
		}
		for i := range h {
			h[i] += ffn[i]
		}
	}
	// manualPos: MLA layers append only the latent (not via Append), and KDA layers never
	// Append at all, so neither can drive the last-layer position trigger reliably.
	cache.Advance()
	return h, nil
}
