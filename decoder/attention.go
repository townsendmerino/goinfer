package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// addBias adds a per-output bias vector to a projection result in place
// (Qwen2's q/k/v projections). len(b) must equal len(x).
func addBias(x, b []float32) {
	for i := range x {
		x[i] += b[i]
	}
}

// causalAttention runs one decoder block's grouped-query causal attention
// for a single decode step (the query is the one new position; keys/values
// come from the KV cache plus this step's own K/V). Every per-family knob
// (QKV bias, QK-norm, RoPE base/scaling vs. learned positions, sliding
// window, attention scale) is read from arch, so this one body serves all
// supported families.
//
// Shapes / steps:
//
//	h        [HiddenDim]                  the current position's hidden state (post pre-norm)
//	QProj    [NumHeads*HeadDim, HiddenDim]
//	KProj/VProj [NumKVHeads*HeadDim, HiddenDim]
//	OProj    [HiddenDim, NumHeads*HeadDim]
//
//	1. q = QProj·h (NumHeads heads); k,v = KProj·h, VProj·h (NumKVHeads heads),
//	   plus the optional q/k/v bias (arch.QKVBias — Qwen2).
//	2. QK-norm: rmsNorm each q and k head over HeadDim, if arch.QKNorm
//	   (Gemma 3, Qwen3).
//	3. RoPE q,k at absolute position cache.Pos() with the per-layer inv-freq
//	   table (Gemma local 10k vs. global 1e6; llama3 scaling), unless the
//	   family uses learned absolute positions (arch.LearnedPosEmbed — GPT-2).
//	4. cache.Append(layer, k, v)
//	5. for each query head, attend over cache keys in
//	   [cache.WindowStart(global), cache.Pos()) — the GQA group maps query
//	   head h → kv head h/(NumHeads/NumKVHeads). scale by 1/sqrt(QueryPreAttnScalar).
//	6. softmax (shared kernel) → weighted sum of values → ctx
//	7. out = OProj·ctx ; caller applies post-attn norm + residual add.
//
// causalAttention computes one position's attention block and writes the
// output-projected result into out ([hidden]); the caller applies the post-attn
// norm + residual add. The q/k/v/ctx/scores buffers are reused from the cache's
// per-stream scratch — no per-call allocation in steady-state decode.
func causalAttention(
	layer int,
	h []float32,
	out []float32,
	lw *LayerWeights,
	arch *Architecture,
	cache *KVCache,
	be Backend,
) error {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	global := arch.isGlobalLayer(layer)
	pos := cache.Pos() // this token's absolute position (stable across layers in one forward)

	// 1. Project to q/k/v for the new position (scratch buffers; matmul fully
	// overwrites each). When all three are W8A8 they run in ONE batched dispatch
	// (the activation is quantized once, weights read in place — no concat), which
	// is the per-token dispatch cut without disturbing the prequant aliasing.
	scr := cache.scr
	q, k, v := scr.q, scr.k, scr.v
	if lw.QProj.isW8A8() && lw.KProj.isW8A8() && lw.VProj.isW8A8() {
		scr.qkvOps[0] = linalg.W8A8Op{BQ: lw.QProj.q8, Scales: lw.QProj.scales, Dst: q, N: lw.QProj.rows}
		scr.qkvOps[1] = linalg.W8A8Op{BQ: lw.KProj.q8, Scales: lw.KProj.scales, Dst: k, N: lw.KProj.rows}
		scr.qkvOps[2] = linalg.W8A8Op{BQ: lw.VProj.q8, Scales: lw.VProj.scales, Dst: v, N: lw.VProj.rows}
		linalg.MatmulBTW8A8Batch(scr.ws, h, 1, lw.QProj.cols, scr.qkvOps[:])
	} else {
		lw.QProj.matmulInto(scr.ws, be, h, q, 1)
		lw.KProj.matmulInto(scr.ws, be, h, k, 1)
		lw.VProj.matmulInto(scr.ws, be, h, v, 1)
	}
	if arch.QKVBias {
		addBias(q, lw.QBias)
		addBias(k, lw.KBias)
		addBias(v, lw.VBias)
	}

	// 2. QK-norm (Gemma 3, Qwen3): RMSNorm over head_dim, per head, before RoPE.
	if arch.QKNorm {
		rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
		rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
	}

	// 3. RoPE at pos with the per-layer inv-freq table (Gemma: local 10k vs
	// global 1e6 base; Llama-3: llama3 scaling baked in; Mellum: YaRN on full
	// layers, plain on sliding; Phi: partial rotary). The mscale folds YaRN's
	// attention_factor into the rotation (1.0 elsewhere). GPT-2 uses learned
	// absolute positions instead, so it skips RoPE.
	if !arch.LearnedPosEmbed {
		invFreq := arch.ropeInvFreq(layer)
		ms := arch.ropeMscale(layer)
		applyRoPE(q, nH, hd, pos, invFreq, ms)
		applyRoPE(k, nKV, hd, pos, invFreq, ms)
	}

	// 4. Append this position's K/V, then attend over the stored history.
	cache.Append(layer, k, v)
	ctx := cache.scr.ctx
	nKeys := len(cache.Keys(layer)) / kvDim
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, global, arch)

	// 7. Output projection into the caller's buffer (+ bias for GPT-2); the
	// caller applies the post-attn norm + residual.
	lw.OProj.matmulInto(scr.ws, be, ctx, out, 1)
	if arch.OutBias {
		addBias(out, lw.OBias)
	}
	return nil
}

// attendQuery computes the attention context for a single query q (already
// RoPE'd / QK-normed, and whose K/V are already appended to the cache) over the
// layer's stored keys/values in the causal/window range, into ctx ([qDim]):
// per (GQA) head, scaled dot-product scores → softmax → weighted sum of values.
// scores is a reusable buffer with len ≥ the number of stored keys. Shared by
// causalAttention (M=1) and the batched forwardN, so the math has one home.
func attendQuery(q, ctx, scores []float32, cache *KVCache, layer, pos int, global bool, arch *Architecture) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	start := cache.WindowStart(pos, global)
	scale := arch.AttnScale // query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV       // GQA: query heads per KV head

	clear(ctx) // accumulated into below
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := dot * scale
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}

		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			vHead := vals[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}
}
