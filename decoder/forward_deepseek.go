package decoder

import "math"

// DeepSeek-V2/V3 (deepseek_v2 / deepseek_v3) forward path — Multi-head Latent Attention
// over a DeepSeekMoE FFN. MLA is the third efficient-attention coverage axis (latent-KV):
// K/V are compressed to a shared low-rank latent (kv_lora_rank), and ONLY that latent (‖ a
// per-position rope-carrying key) is cached — ~576 floats/token vs the ~41k a reconstructed
// full K+V would need. Per-head K/V are rebuilt from the latent each step (the "naive" path,
// bit-identical to HF; the "absorb" optimization that folds kv_b_proj into q/o is a perf
// follow-up, not needed for parity). Decoupled RoPE rides on a separate qk_rope_head_dim
// slice of Q and the shared latent key; the no-rope dims and the (different-width) V skip it.
//
// The block is a standard Pre2 residual stack — input_layernorm → MLA → +residual →
// post_attention_layernorm → MoE/dense → +residual — so the FFN reuses the generic mlp()
// dispatch (dense prefix on l < first_k_dense_replace, DeepSeekMoE elsewhere). Parity-first
// f32, one token per call (the latent append + causal attend mirror the other own-path
// families); canBatchN excludes the MLA attention-kind.
func (m *Model) runLayersDeepseek(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos()
	if cache.scr == nil { // dense-prefix gatedMLP reuses scratch; tests may skip runLayers' setup
		cache.scr = newDecodeScratch(arch)
	}

	h := make([]float32, hidden)
	m.w.Embed.Row(id, h)

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		// Attention sub-block (Pre2).
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		attn := m.mlaAttention(n, lw, arch, cache, l, pos)
		for i := range h {
			h[i] += attn[i]
		}
		// FFN sub-block (Pre2). post_attention_layernorm is the pre-MLP norm; mlp()
		// routes dense (l < FirstKDense, Experts nil) vs DeepSeekMoE per layer.
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
	cache.Advance() // manualPos: MLA appends only the latent (not via Append), so step pos once per token
	return h, nil
}

// mlaAttention is one Multi-head Latent Attention layer. It projects the query (through
// the q_a/q_b LoRA bottleneck or a direct q_proj), down-projects K/V to the compressed
// latent + rope key (appended to the cache), then attends causally over every stored
// latent by reconstructing per-head k_nope/v via kv_b_proj and roping the per-position
// key. The q·k width (qk_head_dim) differs from the v width (v_head_dim), so this can't
// reuse the uniform attendQuery — it's the new primitive.
func (m *Model) mlaAttention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	p := arch.mla
	w := lw.mla
	hidden := arch.HiddenDim
	eps := arch.NormEps
	H := arch.NumHeads
	qkNope, qkRope, vHead := p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim
	qkHead := qkNope + qkRope
	rank := p.KVLoRARank
	latDim := rank + qkRope
	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer) // YaRN attention_factor folded into cos/sin (1.0 when none)

	// 1. Query: optional q_a→norm→q_b LoRA bottleneck, else a direct q_proj. Layout
	// per head: [qk_nope | qk_rope].
	var q []float32
	if p.QLoRARank > 0 {
		qa := matvec(w.qAProj, p.QLoRARank, hidden, n)
		rmsNorm(qa, w.qALayernorm, 1, p.QLoRARank, eps, false)
		q = matvec(w.qBProj, H*qkHead, p.QLoRARank, qa)
	} else {
		q = matvec(w.qProj, H*qkHead, hidden, n)
	}

	// 2. KV down-projection → [latent (kv_lora_rank) | rope key (qk_rope_head_dim)].
	// The whole vector is the cached latent payload.
	kvDown := matvec(w.kvAProj, latDim, hidden, n)
	cache.AppendLatent(layer, kvDown)

	// 3. RoPE the query's rope dims (per head), gathered out of the [nope|rope] layout.
	qRope := make([]float32, H*qkRope)
	for hh := 0; hh < H; hh++ {
		copy(qRope[hh*qkRope:(hh+1)*qkRope], q[hh*qkHead+qkNope:hh*qkHead+qkHead])
	}
	mlaRope(qRope, H, qkRope, pos, invFreq, ms, p.ropeInterleave)

	// 4. Attend causally over every stored latent. Reconstruct per-head k_nope/v from
	// each position's latent (kv_b_proj on the normalized latent) and rope its shared key.
	nKeys := len(cache.Latent(layer)) / latDim
	kvUpDim := H * (qkNope + vHead)
	scores := make([]float32, H*nKeys)      // [head, key]
	vKeys := make([]float32, nKeys*H*vHead) // reconstructed V per (key, head)
	krj := make([]float32, qkRope)
	for j := 0; j < nKeys; j++ {
		lat := cache.Latent(layer)[j*latDim : (j+1)*latDim]
		cn := append([]float32(nil), lat[:rank]...)
		rmsNorm(cn, w.kvALayernorm, 1, rank, eps, false)
		kvUp := matvec(w.kvBProj, kvUpDim, rank, cn) // per head: [k_nope | v]
		copy(krj, lat[rank:latDim])
		mlaRope(krj, 1, qkRope, j, invFreq, ms, p.ropeInterleave) // key rope at its own position
		for hh := 0; hh < H; hh++ {
			base := hh * (qkNope + vHead)
			kNope := kvUp[base : base+qkNope]
			v := kvUp[base+qkNope : base+qkNope+vHead]
			var s float32
			for d := 0; d < qkNope; d++ {
				s += q[hh*qkHead+d] * kNope[d]
			}
			for d := 0; d < qkRope; d++ {
				s += qRope[hh*qkRope+d] * krj[d]
			}
			scores[hh*nKeys+j] = s * float32(arch.AttnScale)
			copy(vKeys[(j*H+hh)*vHead:(j*H+hh+1)*vHead], v)
		}
	}

	// 5. Per-head softmax over keys, weighted sum of V → context [H, v_head_dim].
	ctx := make([]float32, H*vHead)
	for hh := 0; hh < H; hh++ {
		row := scores[hh*nKeys : (hh+1)*nKeys]
		mx := float32(math.Inf(-1))
		for _, s := range row {
			if s > mx {
				mx = s
			}
		}
		var sum float32
		for j := range row {
			row[j] = float32(math.Exp(float64(row[j] - mx)))
			sum += row[j]
		}
		out := ctx[hh*vHead : (hh+1)*vHead]
		for j := 0; j < nKeys; j++ {
			wgt := row[j] / sum
			v := vKeys[(j*H+hh)*vHead : (j*H+hh+1)*vHead]
			for d := 0; d < vHead; d++ {
				out[d] += wgt * v[d]
			}
		}
	}

	// 6. Output projection (over the v_head_dim-wide context).
	return matvec(w.oProj, hidden, H*vHead, ctx)
}

// mlaRope rotates the rope-carrying dims of a [heads, ropeDim] vector at absolute
// position pos. DeepSeek's rope_interleave (V3 default) lays the rotary dims out as
// adjacent pairs (x0,x1),(x2,x3),…; de-interleaving each head into [evens|odds] and then
// applying the standard NeoX rotate_half (applyRoPE) yields a layout P(interleaved-RoPE),
// and since the SAME permutation P hits both query and key, the q·k dot product is
// identical to HF's interleaved RoPE. rope_interleave=false skips the de-interleave (plain
// NeoX). The rotary table (invFreq) is sized to ropeDim via Architecture.RotaryDim.
func mlaRope(vec []float32, heads, ropeDim, pos int, invFreq []float64, scale float64, interleave bool) {
	if interleave {
		half := ropeDim / 2
		tmp := make([]float32, ropeDim)
		for hh := 0; hh < heads; hh++ {
			src := vec[hh*ropeDim : (hh+1)*ropeDim]
			for i := 0; i < half; i++ {
				tmp[i] = src[2*i]
				tmp[half+i] = src[2*i+1]
			}
			copy(src, tmp)
		}
	}
	applyRoPE(vec, heads, ropeDim, pos, invFreq, scale)
}
