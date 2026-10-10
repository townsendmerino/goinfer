package decoder

import (
	"math"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// addBias adds b into x in place, for architectures with q/k/v/o bias (Qwen2's projections, GPT-2's output). Callers pass equal
// lengths; only the first min(len(x), len(b)) elements are touched.
func addBias(x, b []float32) {
	n := min(len(x), len(b))
	if n == 0 {
		return
	}
	x = x[:n]
	b = b[:n]
	_ = x[n-1]
	_ = b[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = x[i+3]
		_ = b[i+3]
		x[i] += b[i]
		x[i+1] += b[i+1]
		x[i+2] += b[i+2]
		x[i+3] += b[i+3]
	}
	for ; i < n; i++ {
		x[i] += b[i]
	}
}

// causalAttention runs one decoder block's grouped-query causal attention for a single decode step (the query is the one new
// position; keys/values come from the KV cache plus this step's own K/V) and writes the output-projected result into out
// [hidden]; the caller applies the post-attn norm and residual add. Every per-family knob (QKV bias, QK-norm, RoPE or learned
// positions, sliding window, attention scale) is read from arch, so one body serves all families. The q/k/v/ctx/scores buffers
// are reused from the cache's per-stream scratch, so steady-state decode allocates nothing.
//
// Steps: q/k/v projections (plus optional bias and compute-time LoRA), QK-norm, RoPE at cache.Pos() with the per-layer inv-freq
// table (skipped for learned absolute positions), cache.Append, attention over [WindowStart, Pos] with query head h reading kv
// head h/(NumHeads/NumKVHeads), output gating, then OProj.
func causalAttention(
	layer int,
	h []float32,
	out []float32,
	lw *LayerWeights,
	arch *Architecture,
	cache *KVCache,
	be Backend,
	lora *loraLayerDelta, // compute-time LoRA deltas for this layer (#7); nil = none
) error {
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	global := arch.isGlobalLayer(layer)
	pos := cache.Pos() // this token's absolute position (stable across layers in one forward)

	// 1. Project to q/k/v for the new position (scratch buffers; matmul fully
	// overwrites each). When all three are W8A8 they run in ONE batched dispatch
	// (the activation is quantized once, weights read in place — no concat), which
	// is the per-token dispatch cut without disturbing the prequant aliasing.
	scr := cache.scr
	// The scratch is sized for the WIDEST layer (maxHeads); slice it to this layer's
	// query width so every downstream reader sees the right length. Identical to
	// scr.q for every family whose head count is uniform.
	q, k, v := scr.q[:nH*hd], scr.k, scr.v
	var dt0 time.Time
	if decodeTiming {
		dt0 = time.Now()
		defer func() { atomic.AddInt64(&dtAttnAll, int64(time.Since(dt0))) }()
	}
	if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
		scr.qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
		scr.qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
		scr.qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
		matmulW8A8Batch(be, scr.ws, h, 1, lw.QProj.Cols(), scr.qkvOps[:], lw.QProj.ActQuantGroup())
	} else if w4a8BatchEnabled && isW4A8(&lw.QProj) && isW4A8(&lw.KProj) && isW4A8(&lw.VProj) {
		// Fused q/k/v W4A8, mirroring the W8A8 batch above. group is shared across the three (same layer, same quant config), matching
		// MatmulBTW4A8Batch's single-scalar signature.
		var group int
		scr.qkvOpsW4[0], group = wmW4A8Op(&lw.QProj, q)
		scr.qkvOpsW4[1], _ = wmW4A8Op(&lw.KProj, k)
		scr.qkvOpsW4[2], _ = wmW4A8Op(&lw.VProj, v)
		scr.ws.SetThreshold(int4ParThreshold)
		matmulW4A8Batch(be, scr.ws, h, 1, lw.QProj.Cols(), group, scr.qkvOpsW4[:], lw.QProj.ActQuantGroup())
	} else {
		// h is quantized once for q, k and v when they run the CPU W4A8 path, not once per matmul.
		scr.hq.prepare(be, &lw.QProj, h, 1)
		matmulIntoPre(scr.ws, be, &lw.QProj, &scr.hq, h, q, 1)
		matmulIntoPre(scr.ws, be, &lw.KProj, &scr.hq, h, k, 1)
		matmulIntoPre(scr.ws, be, &lw.VProj, &scr.hq, h, v, 1)
	}
	if decodeTiming {
		atomic.AddInt64(&dtQKV, int64(time.Since(dt0)))
	}
	// Compute-time LoRA (#7): add the low-rank delta to each projection output,
	// exactly where merge would have folded it into the weight — before QK-norm/RoPE.
	if lora != nil {
		applyLoRA(lora.q, h, q, scr)
		applyLoRA(lora.k, h, k, scr)
		applyLoRA(lora.v, h, v, scr)
	}
	if arch.QKVBias {
		addBias(q, lw.QBias)
		addBias(k, lw.KBias)
		addBias(v, lw.VBias)
	}

	// 2. QK-norm (Gemma 3, Qwen3): RMSNorm over head_dim, per head, before RoPE. Olmo 3/Olmo
	// Hybrid (QKNormWhole) normalize the WHOLE projected vector as one statistic instead —
	// same rmsNorm call, rows/dim swapped (see QKNormWhole's own comment).
	if arch.QKNorm {
		if arch.QKNormWhole {
			rmsNorm(q, lw.QNorm, 1, nH*hd, arch.NormEps, arch.RMSAddOne)
			rmsNorm(k, lw.KNorm, 1, nKV*hd, arch.NormEps, arch.RMSAddOne)
		} else {
			rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
			rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
		}
	}

	// 3. RoPE at pos with the per-layer inv-freq table (Gemma: local 10k vs
	// global 1e6 base; Llama-3: llama3 scaling baked in; Mellum: YaRN on full
	// layers, plain on sliding; Phi: partial rotary). The mscale folds YaRN's
	// attention_factor into the rotation (1.0 elsewhere). GPT-2 uses learned
	// absolute positions instead, so it skips RoPE.
	if !arch.LearnedPosEmbed && !arch.isNoPELayer(layer) {
		invFreq := arch.ropeInvFreq(layer)
		ms := arch.ropeMscale(layer)
		ropeAt(q, nH, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
		ropeAt(k, nKV, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
	}

	// 3.5. Attention-temperature tuning (Ministral 3): a position-dependent scale on the query,
	// ON TOP OF RoPE (not instead of it — see AttnTempBeta's own comment on why this can't reuse
	// llama4Architecture's own-forward attnTemp branch, which is an either/or with RoPE).
	// AttnTempBeta == 0 for every family that doesn't set it, so this is a no-op elsewhere.
	if arch.AttnTempBeta != 0 {
		scale := float32(1 + arch.AttnTempBeta*math.Log1p(math.Floor(float64(pos)/arch.AttnTempOrigMaxPos)))
		for i := range q {
			q[i] *= scale
		}
	}

	// 4. Append this position's K/V, then attend over the stored history through the same attendBatchedHeads kernel (at K=1) that the
	// batched prefill/verify (forwardN) uses, with f64 accumulation (acc64), so decode is bit-identical to the batched forward for
	// dense and MoE alike. Same-model speculative decoding requires it: the target's batched verify must reproduce sequential greedy
	// exactly, and f32's QKᵀ/AV reduction is M-dependent (K=1 decode differs from M=K verify) while f64 is order-independent. Gate:
	// TestForwardN_matchesSequential and TestSpeculativeGreedyParity. The three cases mirror forwardN's: ring window, int8-KV global
	// (dequant to f32 scratch), f32 global (append-forever).
	ctx := cache.scr.ctx[:nH*hd]
	acc64 := true
	switch {
	case cache.rings[layer] != nil && ringDirectDecode && cache.rings[layer].quant != kvI8:
		// Local ring layer, f32 (ringDirectDecode): store this token's K/V first, then read the window [base, pos] in place as one
		// contiguous slice. The write before the read is safe because the slot it takes holds position pos-W, the one row just outside
		// the window [pos-W+1, pos] (or is empty before the first wrap). The copy path below moves every resident row into scratch each
		// token; attention sees the same rows in the same order, so this is bit-identical.
		r := cache.rings[layer]
		r.write(pos, k, v)
		base := max(pos-r.w+1, 0)
		nKeys := pos - base + 1
		wk, wv := r.window(base, nKeys)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, !acc64 && cache.treeMask == nil, acc64)
		attendBatchedHeads(q, ctx, wk, wv, base, cache, layer, pos, 1, global, arch, acc64, pool)
		if !cache.manualPos && layer == cache.numLayers-1 {
			cache.pos++ // the write above doesn't step pos; mirror Append's last-layer advance
		}
	case cache.rings[layer] != nil:
		// Local ring layer: defer the write past the read and assemble the [base, pos]
		// window (resident history + this token's K/V), as the batched prefill does.
		rows := pos - cache.WindowStart(pos, global) + 1
		lk, lv := scr.localBufs(rows * kvDim)
		base, nKeys := cache.batchReadLocal(layer, pos, 1, k, v, lk, lv)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, !acc64 && cache.treeMask == nil, acc64)
		attendBatchedHeads(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], base, cache, layer, pos, 1, global, arch, acc64, pool)
		cache.commitBatch(layer, pos, 1, k, v)
		if !cache.manualPos && layer == cache.numLayers-1 {
			cache.pos++ // commitBatch doesn't step pos; mirror Append's last-layer advance
		}
	case cache.quant == kvI8:
		// int8-KV global: Append quantizes the new K/V into the layer, then dequant the
		// full history into f32 scratch for the matmul (mirrors forwardN's int8 branch).
		cache.Append(layer, k, v)
		lk, lv := scr.localBufs(cache.storedRows(layer, kvDim) * kvDim)
		nKeys := cache.dequantGlobalLayer(layer, kvDim, lk, lv)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, !acc64 && cache.treeMask == nil, acc64)
		attendBatchedHeads(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], 0, cache, layer, pos, 1, global, arch, acc64, pool)
	default:
		// f32 global (append-forever): read the whole stored history.
		cache.Append(layer, k, v)
		nKeys := cache.storedRows(layer, kvDim)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, !acc64 && cache.treeMask == nil, acc64)
		attendBatchedHeads(q, ctx, cache.Keys(layer), cache.Vals(layer), 0, cache, layer, pos, 1, global, arch, acc64, pool)
	}

	// 6b. Output gating, applied to the attention context before o_proj: ctx *= gate(g_proj · h), where h is this layer's
	// post-input_layernorm hidden state (the tensor q/k/v were projected from, so no extra tap). applyAttnGate dispatches the
	// activation (Laguna softplus; Spark-X2.5 sigmoid, arch.AttnGate). No-op for every other family.
	if arch.hasAttnOutputGate() {
		applyAttnGate(scr, be, lw, arch, h, ctx, nH, hd)
	}

	// 7. Output projection into the caller's buffer (+ bias for GPT-2); the
	// caller applies the post-attn norm + residual.
	var dt1 time.Time
	if decodeTiming {
		dt1 = time.Now()
	}
	matmulInto(scr.ws, be, &lw.OProj, ctx, out, 1)
	if decodeTiming {
		atomic.AddInt64(&dtO, int64(time.Since(dt1)))
	}
	if lora != nil {
		applyLoRA(lora.o, ctx, out, scr)
	}
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
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	// Local (sliding-window) layers store only the last W positions in a ring: absolute key s lives at row s%W, nKeys becomes the
	// logical count, and the window guarantees every read s in [start, nKeys) is resident. wrap=0 means global append-forever,
	// row = s. scores stays indexed by absolute s (its scratch is already context-sized).
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
	start := max(cache.WindowStart(pos, global), arch.attnChunkStart(layer, pos))
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
			sc := softcapScore(dot*scale, arch.AttnLogitSoftcap)
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
			addScaled(oHead, vHead, w)
		}
	}
}

// attendQueryI8 is attendQuery over an int8 KV cache (kvI8): the query head is
// quantized once per head, key scores are the integer dot DotI8 rescaled by
// qScale·kScale[s,head] (the f64-accum scalar loop becomes an SDOT — decode
// attention gets faster, not just smaller), and the V-weighted sum dequantizes
// inline. Handles both ring (local, s%W) and append-forever (global) int8 layers.
func attendQueryI8(q, ctx, scores []float32, cache *KVCache, layer, pos int, global bool, arch *Architecture) {
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
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
	// Chunked layers here too, for symmetry with attendQuery: a start rule that lives in only one of two twins is how they drift.
	// (llama4 is f32-only, so this path is not reachable for it today.)
	start := max(cache.WindowStart(pos, global), arch.attnChunkStart(layer, pos))
	scale := arch.AttnScale
	group := nH / nKV
	var qqBuf [256]int8
	var qq []int8
	if hd <= len(qqBuf) {
		qq = qqBuf[:hd]
	} else {
		qq = make([]int8, hd)
	}

	clear(ctx)
	for qh := range nH {
		kvh := qh / group
		qScale := float64(linalg.QuantizeRowInt8(q[qh*hd:qh*hd+hd], qq))

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			row, srow := phys(s)*kvDim+kvh*hd, phys(s)*nKV+kvh
			dot := linalg.DotI8(qq, kQ[row:row+hd])
			sc := softcapScore(float64(dot)*qScale*float64(kSc[srow])*scale, arch.AttnLogitSoftcap)
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
		_ = oHead[hd-1]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			row, srow := phys(s)*kvDim+kvh*hd, phys(s)*nKV+kvh
			wvs := w * vSc[srow]
			vrow := vQ[row : row+hd]
			_ = vrow[hd-1]
			d := 0
			for ; d+3 < hd; d += 4 {
				oHead[d] += wvs * float32(vrow[d])
				oHead[d+1] += wvs * float32(vrow[d+1])
				oHead[d+2] += wvs * float32(vrow[d+2])
				oHead[d+3] += wvs * float32(vrow[d+3])
			}
			for ; d < hd; d++ {
				oHead[d] += wvs * float32(vrow[d])
			}
		}
	}
}

// applyAttnGate applies the output gate to the attention context in place, before the output projection: ctx *= gate(g_proj · h),
// h being the post-input_layernorm hidden state, the gate per head (broadcast) or per element. Laguna's gate is softplus,
// Spark-X2.5's sigmoid (arch.AttnGate); applyGateRow and applySigmoidGateRow hold the math.
//
// Two parity details are load-bearing (modeling_laguna.py). Softplus is taken in float32: the vendor upcasts before the
// nonlinearity, and goinfer's activations are already f32, but a bf16 port would be wrong. And the gate reads the layer input
// (post-input_layernorm), not the attention output, which would be a different model.
//
// Granularity is read from the weight's row count (nH is per-head, nH*hd per-element), not from config.gating: released
// checkpoints disagree with their own config (Laguna-XS.2 declares gating: true yet ships a per-head g_proj), the vendor's
// spelling-to-granularity rule is generation-specific, and the shape is unambiguous because hd > 1. arch.laguna.GatePerHead
// records what the config declared and is used only to flag a mismatch at load.
func applyAttnGate(scr *decodeScratch, be Backend, lw *LayerWeights, arch *Architecture, h, ctx []float32, nH, hd int) {
	gates := scr.gateBuf(lw.GProj.Rows())
	matmulInto(scr.ws, be, &lw.GProj, h, gates, 1)
	if arch.AttnGate == GateSigmoid {
		applySigmoidGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
		return
	}
	applyGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
}

// applyGateRow multiplies one position's attention context by its softplus gate, in place. It is the single home for the gate
// math: causalAttention calls it at K=1 and the batched forward calls it per row, so the two paths cannot drift (separate copies
// once left the gate off batched prefill, which shows as a plausible cosine, not a crash). perHead means gates has nH entries,
// broadcast across each head's hd channels; otherwise nH*hd entries, one per channel.
func applyGateRow(gates, ctx []float32, perHead bool, nH, hd int) {
	if perHead {
		for head := range nH {
			g := softplus32(gates[head])
			row := ctx[head*hd : head*hd+hd]
			for i := range row {
				row[i] *= g
			}
		}
		return
	}
	for i := range ctx {
		ctx[i] *= softplus32(gates[i])
	}
}

// applySigmoidGateRow is applyGateRow with a sigmoid gate (Bailing Hybrid's MLA output gate; the vendor's F.sigmoid(gate) in
// modeling_bailing_moe_v3.py) instead of softplus; the perHead/per-element convention is the same.
func applySigmoidGateRow(gates, ctx []float32, perHead bool, nH, hd int) {
	if perHead {
		for head := range nH {
			g := sigmoidf(gates[head])
			row := ctx[head*hd : head*hd+hd]
			for i := range row {
				row[i] *= g
			}
		}
		return
	}
	for i := range ctx {
		ctx[i] *= sigmoidf(gates[i])
	}
}

// softplus32 is log(1+exp(x)) with the standard large-x guard: for x above the
// threshold exp(x) overflows while log1p(exp(x)) is x to within f32 resolution, so
// returning x avoids an Inf that would otherwise poison the whole head. torch's
// F.softplus applies the same linear fallback (its default threshold is 20).
func softplus32(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

// softcapScore is Gemma 2's attention-score soft-capping (attn_logit_softcapping): cap·tanh(s/cap), applied to the
// scaled q·k score before the mask and softmax, as HF's Gemma2Attention does. cap 0 leaves the score unchanged, so every
// other family takes the plain path. Every CPU scoring site calls it: attendQuery, attendQueryI8 and attendBatchedHeads.
func softcapScore(s, c float64) float64 {
	if c == 0 {
		return s
	}
	return c * math.Tanh(s/c)
}
