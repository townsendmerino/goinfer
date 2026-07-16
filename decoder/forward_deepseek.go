package decoder

import (
	"math"
	"os"
	"sync"
)

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

// mlaAttention dispatches to the absorb path (default — folds kv_b_proj into q/o so no
// per-key K/V is ever reconstructed; the perf path) or the naive reconstruct-per-key
// path (parity reference + GOINFER_MLA_NAIVE / m.mlaNaive fallback). The two are
// algebraically identical (cosine ~1.0, not bit-exact — f32 reorder), gated by
// TestMLAAbsorb_parity on the tiny golden and the V2-Lite/Moonlight real-model gates.
func (m *Model) mlaAttention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	if mlaForceNaive || mlaNaiveEnv() {
		return m.mlaAttentionNaive(n, lw, arch, cache, layer, pos)
	}
	return m.mlaAttentionAbsorb(n, lw, arch, cache, layer, pos)
}

// mlaAttentionNaive is one Multi-head Latent Attention layer. It projects the query
// (through the q_a/q_b LoRA bottleneck or a direct q_proj), down-projects K/V to the
// compressed latent + rope key (appended to the cache), then attends causally over every
// stored latent by reconstructing per-head k_nope/v via kv_b_proj and roping the
// per-position key. The q·k width (qk_head_dim) differs from the v width (v_head_dim), so
// this can't reuse the uniform attendQuery — it's the new primitive. This is the
// parity-reference path (bit-identical to HF); the production default is the absorb path.
func (m *Model) mlaAttentionNaive(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
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
	for hh := range H {
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
	for j := range nKeys {
		lat := cache.Latent(layer)[j*latDim : (j+1)*latDim]
		cn := append([]float32(nil), lat[:rank]...)
		rmsNorm(cn, w.kvALayernorm, 1, rank, eps, false)
		kvUp := matvec(w.kvBProj, kvUpDim, rank, cn) // per head: [k_nope | v]
		copy(krj, lat[rank:latDim])
		mlaRope(krj, 1, qkRope, j, invFreq, ms, p.ropeInterleave) // key rope at its own position
		for hh := range H {
			base := hh * (qkNope + vHead)
			kNope := kvUp[base : base+qkNope]
			v := kvUp[base+qkNope : base+qkNope+vHead]
			var s float32
			for d := range qkNope {
				s += q[hh*qkHead+d] * kNope[d]
			}
			for d := range qkRope {
				s += qRope[hh*qkRope+d] * krj[d]
			}
			scores[hh*nKeys+j] = s * float32(arch.AttnScale)
			copy(vKeys[(j*H+hh)*vHead:(j*H+hh+1)*vHead], v)
		}
	}

	// 5. Per-head softmax over keys, weighted sum of V → context [H, v_head_dim].
	ctx := make([]float32, H*vHead)
	for hh := range H {
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
		for j := range nKeys {
			wgt := row[j] / sum
			v := vKeys[(j*H+hh)*vHead : (j*H+hh+1)*vHead]
			for d := range vHead {
				out[d] += wgt * v[d]
			}
		}
	}

	// 6. Output projection (over the v_head_dim-wide context).
	return matvec(w.oProj, hidden, H*vHead, ctx)
}

// mlaAttentionAbsorb is the perf path: the "absorb" trick that never reconstructs
// per-head K/V. kv_b_proj's per-head blocks are W_UK (the k_nope rows, [qkNope×rank])
// and W_UV (the v rows, [vHead×rank]). Two algebraic identities remove the per-key
// reconstruction matvec that dominates the naive path for long context:
//
//	q_nope · (W_UK·c) = (W_UKᵀ·q_nope) · c          ← score in rank-space
//	Σ_j p_j·(W_UV·c_j) = W_UV · (Σ_j p_j·c_j)        ← one W_UV matvec per head, post-softmax
//
// So q is projected through W_UK ONCE per head (→ [rank]), each cached latent is scored
// by a rank-dim dot (no reconstruction), and V is collapsed to a rank-dim weighted sum
// then lifted by W_UV once. Cost drops from O(nKeys·H·(qkNope+vHead)·rank) to
// O(nKeys·H·rank) + O(H·(qkNope+vHead)·rank). Algebraically equal to the naive path
// (cosine ~1.0; f32 reorder, not bit-exact). The latent append, query projection, and
// RoPE are identical to the naive path.
func (m *Model) mlaAttentionAbsorb(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	p := arch.mla
	w := lw.mla
	hidden := arch.HiddenDim
	eps := arch.NormEps
	H := arch.NumHeads
	qkNope, qkRope, vHead := p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim
	qkHead := qkNope + qkRope
	rank := p.KVLoRARank
	latDim := rank + qkRope
	hRow := qkNope + vHead // per-head output block in kv_b_proj (k_nope rows ‖ v rows)
	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)

	// 1. Query (identical to naive): q_a→norm→q_b LoRA bottleneck, or a direct q_proj.
	var q []float32
	if p.QLoRARank > 0 {
		qa := matvec(w.qAProj, p.QLoRARank, hidden, n)
		rmsNorm(qa, w.qALayernorm, 1, p.QLoRARank, eps, false)
		q = matvec(w.qBProj, H*qkHead, p.QLoRARank, qa)
	} else {
		q = matvec(w.qProj, H*qkHead, hidden, n)
	}

	// 2. KV down-projection → [latent | rope key]; the whole vector is cached.
	kvDown := matvec(w.kvAProj, latDim, hidden, n)
	cache.AppendLatent(layer, kvDown)

	// 3. RoPE the query's rope dims (per head), gathered from the [nope|rope] layout.
	qRope := make([]float32, H*qkRope)
	for hh := range H {
		copy(qRope[hh*qkRope:(hh+1)*qkRope], q[hh*qkHead+qkNope:hh*qkHead+qkHead])
	}
	mlaRope(qRope, H, qkRope, pos, invFreq, ms, p.ropeInterleave)

	// 4a. Absorb W_UK into the query: qNopeAbs[h] = W_UK_hᵀ · q_nope_h  ([rank] per head).
	// W_UK_h is the k_nope rows of head h: kv_b_proj rows [h*hRow : h*hRow+qkNope].
	qNopeAbs := make([]float32, H*rank)
	for hh := range H {
		dst := qNopeAbs[hh*rank : (hh+1)*rank]
		qn := q[hh*qkHead : hh*qkHead+qkNope]
		for d := range qkNope {
			row := w.kvBProj[(hh*hRow+d)*rank : (hh*hRow+d)*rank+rank]
			qd := qn[d]
			for c := range rank {
				dst[c] += qd * row[c]
			}
		}
	}

	// 4b. Score each cached latent in rank-space — no per-key K reconstruction. cnAll
	// holds the normalized latents for reuse in the V collapse (step 5).
	nKeys := len(cache.Latent(layer)) / latDim
	scores := make([]float32, H*nKeys)
	cnAll := make([]float32, nKeys*rank)
	krj := make([]float32, qkRope)
	for j := range nKeys {
		lat := cache.Latent(layer)[j*latDim : (j+1)*latDim]
		cn := cnAll[j*rank : (j+1)*rank]
		copy(cn, lat[:rank])
		rmsNorm(cn, w.kvALayernorm, 1, rank, eps, false)
		copy(krj, lat[rank:latDim])
		mlaRope(krj, 1, qkRope, j, invFreq, ms, p.ropeInterleave)
		for hh := range H {
			qa := qNopeAbs[hh*rank : (hh+1)*rank]
			var s float32
			for c := range rank {
				s += qa[c] * cn[c]
			}
			qr := qRope[hh*qkRope : (hh+1)*qkRope]
			for d := range qkRope {
				s += qr[d] * krj[d]
			}
			scores[hh*nKeys+j] = s * float32(arch.AttnScale)
		}
	}

	// 5. Per-head softmax, then collapse V to a rank-space weighted latent sum.
	wsum := make([]float32, H*rank)
	for hh := range H {
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
		acc := wsum[hh*rank : (hh+1)*rank]
		for j := range nKeys {
			wgt := row[j] / sum
			cn := cnAll[j*rank : (j+1)*rank]
			for c := range rank {
				acc[c] += wgt * cn[c]
			}
		}
	}

	// 6. Absorb W_UV: ctx_h = W_UV_h · wsum_h. W_UV_h is the v rows of head h:
	// kv_b_proj rows [h*hRow+qkNope : h*hRow+hRow].
	ctx := make([]float32, H*vHead)
	for hh := range H {
		acc := wsum[hh*rank : (hh+1)*rank]
		out := ctx[hh*vHead : (hh+1)*vHead]
		for e := range vHead {
			row := w.kvBProj[(hh*hRow+qkNope+e)*rank : (hh*hRow+qkNope+e)*rank+rank]
			var s float32
			for c := range rank {
				s += row[c] * acc[c]
			}
			out[e] = s
		}
	}

	// 7. Output projection (over the v_head_dim-wide context).
	return matvec(w.oProj, hidden, H*vHead, ctx)
}

// mlaForceNaive forces the naive (reconstruct-per-key) MLA path over the absorb default.
// It lives here (not on Model in the shared core file) so flipping the MLA path only
// touches the MLA families' parity deps. Set by TestMLAAbsorb_parity / _speed to compare
// the two paths; production leaves it false and uses the env override below. Tests in
// this package run serially, so a package var is safe.
var mlaForceNaive bool

// mlaNaiveEnv reports whether GOINFER_MLA_NAIVE forces the naive MLA path process-wide
// (read once) — the global fallback/debug switch, distinct from mlaForceNaive.
func mlaNaiveEnv() bool {
	mlaNaiveOnce.Do(func() { mlaNaiveFlag = os.Getenv("GOINFER_MLA_NAIVE") != "" })
	return mlaNaiveFlag
}

var (
	mlaNaiveOnce sync.Once
	mlaNaiveFlag bool
)

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
		for hh := range heads {
			src := vec[hh*ropeDim : (hh+1)*ropeDim]
			for i := range half {
				tmp[i] = src[2*i]
				tmp[half+i] = src[2*i+1]
			}
			copy(src, tmp)
		}
	}
	applyRoPE(vec, heads, ropeDim, pos, invFreq, scale)
}
