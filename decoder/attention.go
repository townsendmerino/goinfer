package decoder

import (
	"math"
	"sync/atomic"

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

// attendQuery computes the attention context for a single query q (already
// RoPE'd / QK-normed, and whose K/V are already appended to the cache) over the
// layer's stored keys/values in the causal/window range, into ctx ([qDim]):
// per (GQA) head, scaled dot-product scores → softmax → weighted sum of values.
// scores is a reusable buffer with len ≥ the number of stored keys. It has no
// production caller; tests call it as a scalar attention reference.
func attendQuery(q, ctx, scores []float32, cache *KVCache, layer, pos int, global bool, arch *Architecture) {
	if cache.quant == kvI8 {
		attendQueryI8(q, ctx, scores, cache, layer, pos, global, arch)
		return
	}
	attendQueryP(arch.attnParamsQueryFor(cache, layer, global), q, ctx, scores, cache, layer, pos, arch)
}

// attendQueryPFanned counts attendQueryP calls that fanned their heads out, so a test can tell the schedule was taken.
var attendQueryPFanned atomic.Int64

// attendQueryP is attendQuery over an f32 cache with the window, chunk, scale and softcap given: the program walker's form. Heads fan out
// over the attention workers (scalarHeadWorkers), each with its own score buffer from the cache's scratch; scores is the buffer the
// single-worker path uses. Every head's arithmetic is the one-key-at-a-time loop's, so the result does not depend on the fan-out.
func attendQueryP(p attentionParams, q, ctx, scores []float32, cache *KVCache, layer, pos int, arch *Architecture) {
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
	start := p.Start(pos)
	scale := p.Scale  // query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV // GQA: query heads per KV head

	clear(ctx) // accumulated into below
	head := func(qh int, scores []float32) {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		s := start
		var base [8]int
		var dots [8]float64
		for ; s+8 <= nKeys; s += 8 {
			for j := range base {
				base[j] = rowBase(s+j) + kvh*hd
			}
			dotsF64x8(qHead, keys, &base, &dots)
			for j, dot := range dots {
				sc := softcapScore(dot*scale, p.Softcap)
				scores[s+j] = float32(sc)
				if sc > maxS {
					maxS = sc
				}
			}
		}
		for ; s < nKeys; s++ {
			kHead := keys[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := softcapScore(dot*scale, p.Softcap)
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
	workers := scalarHeadWorkers(nH, nKeys-start, hd)
	if workers <= 1 || cache.scr == nil {
		for qh := range nH {
			head(qh, scores)
		}
		return
	}
	attendQueryPFanned.Add(1)
	pool := cache.scr.headScoresPool(workers, nKeys)
	runSplitAligned(len(pool), nH, 1, func(w, lo, hi int) {
		sc := pool[w].scores[:nKeys]
		for qh := lo; qh < hi; qh++ {
			head(qh, sc)
		}
	})
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

// applyGateRow multiplies one position's attention context by its softplus gate, in place, before the output projection:
// ctx *= softplus(g_proj · x). It is the single home for the gate math: the per-token walker calls it at K=1 and the batched
// walker per row (opOutGate), so the two paths cannot drift (separate copies once left the gate off batched prefill, which
// shows as a plausible cosine, not a crash). perHead means gates has nH entries, broadcast across each head's hd channels;
// otherwise nH*hd entries, one per channel.
//
// Two parity details are load-bearing (modeling_laguna.py). Softplus is taken in float32, as the vendor upcasts before the
// nonlinearity. And x is the layer input after its input norm, not the attention output, which would be a different model.
//
// Callers read granularity from the weight's row count (nH is per-head, nH*hd per-element), not from config.gating: released
// checkpoints disagree with their own config (Laguna-XS.2 declares gating: true yet ships a per-head g_proj), and the shape is
// unambiguous because hd > 1. arch.laguna.GatePerHead records what the config declared and only flags a mismatch at load.
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
	if st := attnScoreStatsForTest.Load(); st != nil {
		st.observe(s)
	}
	return c * math.Tanh(s/c)
}

// attnScoreStatsForTest, when set, counts the scores softcapScore caps: how many, how many past |15|, and the largest
// |score|, before the cap. OP1's real Gemma 2 gate reads it (docs/tasks/task-forward-architecture-2026-10.md §4.6, G2-R's
// planted arm). Set only through StartAttnScoreStatsForTest (goinfer_testhooks); nil costs one atomic load per capped
// score and nothing for a model without a cap.
var attnScoreStatsForTest atomic.Pointer[attnScoreStats]

type attnScoreStats struct {
	n, over15  atomic.Int64
	maxAbsBits atomic.Uint64
}

func (st *attnScoreStats) observe(s float64) {
	st.n.Add(1)
	a := math.Abs(s)
	if a > 15 {
		st.over15.Add(1)
	}
	for {
		old := st.maxAbsBits.Load()
		if a <= math.Float64frombits(old) || st.maxAbsBits.CompareAndSwap(old, math.Float64bits(a)) {
			return
		}
	}
}

// scalarHeadWorkers is how many workers the per-head scalar attention (attendSinksToken, attendQueryP) fans its heads over: one
// below about 32k multiply-adds a layer, where the fan-out costs more than it saves, else one per head up to maxAttnWorkers. It
// changes no number: each head's arithmetic is the same on any worker.
func scalarHeadWorkers(nH, span, hd int) int {
	if nH*span*hd < 1<<15 {
		return 1
	}
	return min(nH, maxAttnWorkers)
}

// dotsF64x8 sets out[j] to the float64 dot of q with the key row at keys[base[j]:base[j]+len(q)], for 8 keys at once. The eight
// chains are independent and each sums d in order, in the one-key loop's expression form (acc += float64(q[d]) * float64(k[d])),
// so each arch contracts every chain exactly as it contracts that loop: the result is bit-identical to scoring the keys one by one.
func dotsF64x8(q, keys []float32, base *[8]int, out *[8]float64) {
	hd := len(q)
	k0, k1, k2, k3 := keys[base[0]:base[0]+hd], keys[base[1]:base[1]+hd], keys[base[2]:base[2]+hd], keys[base[3]:base[3]+hd]
	k4, k5, k6, k7 := keys[base[4]:base[4]+hd], keys[base[5]:base[5]+hd], keys[base[6]:base[6]+hd], keys[base[7]:base[7]+hd]
	var a0, a1, a2, a3, a4, a5, a6, a7 float64
	for d, qv := range q {
		qd := float64(qv)
		a0 += qd * float64(k0[d])
		a1 += qd * float64(k1[d])
		a2 += qd * float64(k2[d])
		a3 += qd * float64(k3[d])
		a4 += qd * float64(k4[d])
		a5 += qd * float64(k5[d])
		a6 += qd * float64(k6[d])
		a7 += qd * float64(k7[d])
	}
	*out = [8]float64{a0, a1, a2, a3, a4, a5, a6, a7}
}
