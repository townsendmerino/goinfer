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
		matmulW8A8Batch(be, scr.ws, h, 1, lw.QProj.cols, scr.qkvOps[:])
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
	ctx := cache.scr.ctx
	if arch.MoE != nil {
		// MoE: route single-token attention through the SAME acc64 batched kernel
		// the prefill uses (at K=1), so decode and prefill are bit-identical and the
		// discrete top-k router never diverges across the prefill↔decode boundary.
		// (Dense tolerates the f32 attendQuery — cosine ≥0.99 — so it stays scalar.)
		var keys, vals []float32
		var base, nKeys int
		if cache.rings[layer] != nil {
			// Local ring layer: defer the write past the read and assemble the
			// [base, pos] window (resident history + this token's K/V), mirroring the
			// deferred-write batched prefill so decode↔prefill stay bit-identical.
			rows := pos - cache.WindowStart(pos, global) + 1
			lk, lv := scr.localBufs(rows * kvDim)
			base, nKeys = cache.batchReadLocal(layer, pos, 1, k, v, lk, lv)
			keys, vals = lk[:nKeys*kvDim], lv[:nKeys*kvDim]
		} else {
			cache.Append(layer, k, v)
			keys, vals, base, nKeys = cache.Keys(layer), cache.Vals(layer), 0, cache.storedRows(layer, kvDim)
		}
		qh, kh, vt, sc, ch := scr.attnBatchBufs(nKeys, hd)
		attendBatchedHeads(q, ctx, keys, vals, base, cache, layer, pos, 1, global, arch, true, qh, kh, vt, sc, ch)
		if cache.rings[layer] != nil {
			cache.commitBatch(layer, pos, 1, k, v)
			if !cache.manualPos && layer == cache.numLayers-1 {
				cache.pos++ // commitBatch doesn't step pos; mirror Append's last-layer advance
			}
		}
	} else {
		cache.Append(layer, k, v)
		nKeys := cache.storedRows(layer, kvDim) // logical count (ring layers store only W)
		attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, global, arch)
	}

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
	if cache.quant == kvI8 {
		attendQueryI8(q, ctx, scores, cache, layer, pos, global, arch)
		return
	}
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	// Local (sliding-window) layers store only the last W positions in a ring; the
	// K/V for absolute key s live at row s%W. nKeys becomes the logical count, and
	// the window guarantees every read s ∈ [start, nKeys) ≥ count-W is resident.
	// wrap=0 ⇒ global append-forever, row = s (unchanged). scores stays indexed by
	// absolute s (its scratch is already context-sized).
	wrap := 0
	if r := cache.rings[layer]; r != nil {
		keys, vals, nKeys, wrap = r.k, r.v, r.count, r.w
	}
	rowBase := func(s int) int {
		if wrap > 0 {
			return (s % wrap) * kvDim
		}
		return s * kvDim
	}
	start := cache.WindowStart(pos, global)
	scale := arch.AttnScale // query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV       // GQA: query heads per KV head

	clear(ctx) // accumulated into below
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
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
			vHead := vals[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}
}

// attendQueryI8 is attendQuery over an int8 KV cache (kvI8): the query head is
// quantized once per head, key scores are the integer dot DotI8 rescaled by
// qScale·kScale[s,head] (the f64-accum scalar loop becomes an SDOT — decode
// attention gets faster, not just smaller), and the V-weighted sum dequantizes
// inline. Handles both ring (local, s%W) and append-forever (global) int8 layers.
func attendQueryI8(q, ctx, scores []float32, cache *KVCache, layer, pos int, global bool, arch *Architecture) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd

	// Storage selection: a ring (local) layer reads its int8 fields with row s%W;
	// a global layer reads the parallel append-forever int8 arrays at row s.
	var kQ, vQ []int8
	var kSc, vSc []float32
	var nKeys, wrap int
	if r := cache.rings[layer]; r != nil {
		kQ, vQ, kSc, vSc, nKeys, wrap = r.kq, r.vq, r.ksc, r.vsc, r.count, r.w
	} else {
		kQ, vQ, kSc, vSc, nKeys = cache.keysQ[layer], cache.valsQ[layer], cache.keyScale[layer], cache.valScale[layer], len(cache.keysQ[layer])/kvDim
	}
	phys := func(s int) int {
		if wrap > 0 {
			return s % wrap
		}
		return s
	}
	start := cache.WindowStart(pos, global)
	scale := arch.AttnScale
	group := nH / nKV
	qq := make([]int8, hd) // quantized query head (reused across keys)

	clear(ctx)
	for qh := range nH {
		kvh := qh / group
		qScale := float64(linalg.QuantizeRowInt8(q[qh*hd:qh*hd+hd], qq))

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			row, srow := phys(s)*kvDim+kvh*hd, phys(s)*nKV+kvh
			dot := linalg.DotI8(qq, kQ[row:row+hd])
			sc := float64(dot) * qScale * float64(kSc[srow]) * scale
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
			row, srow := phys(s)*kvDim+kvh*hd, phys(s)*nKV+kvh
			vs := vSc[srow]
			vrow := vQ[row : row+hd]
			for d := range hd {
				oHead[d] += w * vs * float32(vrow[d])
			}
		}
	}
}
