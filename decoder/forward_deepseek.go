package decoder

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// DeepSeek-V2/V3 (deepseek_v2 / deepseek_v3): Multi-head Latent Attention over a DeepSeekMoE FFN. K/V are compressed to a shared
// low-rank latent (kv_lora_rank), and only that latent (plus a per-position rope-carrying key) is cached, far smaller than a
// reconstructed full K+V. Attention either rebuilds per-head K/V from the latent each step (mlaAttentionNaive, bit-identical to HF)
// or, by default, works in latent space without ever rebuilding them (mlaAttentionAbsorb). Decoupled RoPE rides on a separate
// qk_rope_head_dim slice of Q and the shared latent key; the no-rope dims and the (different-width) V skip it. The layer is the
// pre-norm program of Architecture.layerOps with one opMLA as its attention sublayer.

// mlaAttentionWith is one Multi-head Latent Attention sublayer (opMLA): the absorb path by default (it folds kv_b_proj into q and o,
// so no per-key K/V is ever rebuilt) or the naive reconstruct-per-key path (the parity reference). The two are algebraically
// identical (cosine ~1.0, not bit-exact), gated by TestMLAAbsorb_parity on the tiny golden and the real-model gates. Which runs is a
// knob (GOINFER_MLA_NAIVE) and a test seam, not part of the program, as fast attention is.
func (m *Model) mlaAttentionWith(op *mlaOpParams, n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	if mlaForceNaive || m.knobs.mlaNaive() {
		return m.mlaAttentionNaive(op, n, lw, arch, cache, layer, pos)
	}
	return m.mlaAttentionAbsorb(op, n, lw, arch, cache, layer, pos)
}

// mlaAttentionNaive is one Multi-head Latent Attention layer. It projects the query
// (through the q_a/q_b LoRA bottleneck or a direct q_proj), down-projects K/V to the
// compressed latent + rope key (appended to the cache), then attends causally over every
// stored latent by reconstructing per-head k_nope/v via kv_b_proj and roping the
// per-position key. The q·k width (qk_head_dim) differs from the v width (v_head_dim), so
// this can't reuse the uniform attendQuery — it's the new primitive. This is the
// parity-reference path (bit-identical to HF); the production default is the absorb path.
func (m *Model) mlaAttentionNaive(op *mlaOpParams, n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	p := arch.mla
	w := lw.mla
	hidden := arch.HiddenDim
	eps := op.Eps
	H := arch.NumHeads
	qkNope, qkRope, vHead := p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim
	qkHead := qkNope + qkRope
	rank := p.KVLoRARank
	latDim := rank + qkRope
	invFreq := op.InvFreq
	ms := op.Mscale // YaRN attention_factor folded into cos/sin (1.0 when none)

	// 1. Query: optional q_a→norm→q_b LoRA bottleneck, else a direct q_proj. Layout
	// per head: [qk_nope | qk_rope].
	var q []float32
	if p.QLoRARank > 0 {
		qa := mlaMatvec(w.qAProj, &w.qAQ, p.QLoRARank, hidden, n)
		rmsNorm(qa, w.qALayernorm, 1, p.QLoRARank, eps, false)
		q = mlaMatvec(w.qBProj, &w.qBQ, H*qkHead, p.QLoRARank, qa)
	} else {
		q = mlaMatvec(w.qProj, &w.qQ, H*qkHead, hidden, n)
	}

	// 2. KV down-projection → [latent (kv_lora_rank) | rope key (qk_rope_head_dim)].
	// The whole vector is the cached latent payload.
	kvDown := mlaMatvec(w.kvAProj, &w.kvAQ, latDim, hidden, n)
	cache.AppendLatent(layer, kvDown)

	// 3. RoPE the query's rope dims (per head), gathered out of the [nope|rope] layout.
	qRope := make([]float32, H*qkRope)
	for hh := range H {
		copy(qRope[hh*qkRope:(hh+1)*qkRope], q[hh*qkHead+qkNope:hh*qkHead+qkHead])
	}
	mlaRope(qRope, H, qkRope, pos, invFreq, ms, op.Interleave)

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
		mlaRope(krj, 1, qkRope, j, invFreq, ms, op.Interleave) // key rope at its own position
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
			scores[hh*nKeys+j] = s * float32(op.Scale)
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

	// 5.5. Bailing Hybrid's optional MLA output gate (Laguna-shaped, sigmoid-activated — see
	// applySigmoidGateRow's own comment). nil for every DeepSeek family.
	if w.gProj != nil {
		gate := matvec(w.gProj, len(w.gProj)/hidden, hidden, n)
		applySigmoidGateRow(gate, ctx, op.HeadWise, H, vHead)
	}

	// 6. Output projection (over the v_head_dim-wide context).
	return mlaMatvec(w.oProj, &w.oQ, hidden, H*vHead, ctx)
}

// mlaAttentionAbsorb is the perf path: one row's opMLA through the absorb trick (mlaAbsorbAttend), which never reconstructs
// per-head K/V. The latent append, query projection and RoPE are identical to the naive path. mlaBatch is its batched form and must
// compute what this does, row by row.
func (m *Model) mlaAttentionAbsorb(op *mlaOpParams, n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	p := arch.mla
	w := lw.mla
	rank, qkRope := p.KVLoRARank, p.QKRopeHeadDim

	// 1. Query (identical to naive): q_a→norm→q_b LoRA bottleneck, or a direct q_proj.
	q := mlaQuery(op, p, w, arch.HiddenDim, arch.NumHeads, n)

	// 2. KV down-projection → [latent | rope key]; the whole vector is cached.
	kvDown := mlaMatvec(w.kvAProj, &w.kvAQ, rank+qkRope, arch.HiddenDim, n)
	cache.AppendLatent(layer, kvDown)

	// 3–6. Normalise and rope every cached latent, then attend in rank space.
	lat := cache.Latent(layer)
	nKeys := len(lat) / (rank + qkRope)
	cn, kr := make([]float32, nKeys*rank), make([]float32, nKeys*qkRope)
	mlaPrepLatents(op, p, w, lat, 0, nKeys, cn, kr)
	ctx := mlaAbsorbAttend(op, p, w, arch.NumHeads, q, pos, cn, kr, nKeys, nil)

	// 6.5. Bailing Hybrid's optional MLA output gate (see mlaAttentionNaive's own comment).
	if w.gProj != nil {
		gate := matvec(w.gProj, len(w.gProj)/arch.HiddenDim, arch.HiddenDim, n)
		applySigmoidGateRow(gate, ctx, op.HeadWise, arch.NumHeads, p.VHeadDim)
	}

	// 7. Output projection (over the v_head_dim-wide context).
	return mlaMatvec(w.oProj, &w.oQ, arch.HiddenDim, arch.NumHeads*p.VHeadDim, ctx)
}

// mlaQuery is opMLA's query for one row n: the q_a→norm→q_b LoRA bottleneck, or a direct q_proj. Per head: [qk_nope | qk_rope].
func mlaQuery(op *mlaOpParams, p *mlaParams, w *mlaWeights, hidden, H int, n []float32) []float32 {
	qkHead := p.QKNopeHeadDim + p.QKRopeHeadDim
	if p.QLoRARank > 0 {
		qa := mlaMatvec(w.qAProj, &w.qAQ, p.QLoRARank, hidden, n)
		rmsNorm(qa, w.qALayernorm, 1, p.QLoRARank, op.Eps, false)
		return mlaMatvec(w.qBProj, &w.qBQ, H*qkHead, p.QLoRARank, qa)
	}
	return mlaMatvec(w.qProj, &w.qQ, H*qkHead, hidden, n)
}

// mlaPrepLatents writes keys [from, to) of the cached latents lat as the absorb path reads them: cn[j] the kv_a-normalised latent
// ([rank]) and kr[j] the rope key rotated at its own position j ([qk_rope_head_dim]). Each key's values depend on that key alone, so
// a pass may prepare them once and share them across its rows.
func mlaPrepLatents(op *mlaOpParams, p *mlaParams, w *mlaWeights, lat []float32, from, to int, cn, kr []float32) {
	rank, qkRope := p.KVLoRARank, p.QKRopeHeadDim
	latDim := rank + qkRope
	for j := from; j < to; j++ {
		l := lat[j*latDim : (j+1)*latDim]
		c := cn[j*rank : (j+1)*rank]
		copy(c, l[:rank])
		rmsNorm(c, w.kvALayernorm, 1, rank, op.Eps, false)
		r := kr[j*qkRope : (j+1)*qkRope]
		copy(r, l[rank:latDim])
		mlaRope(r, 1, qkRope, j, op.InvFreq, op.Mscale, op.Interleave)
	}
}

// mlaAbsorbScratch holds one row's absorb buffers so a batched pass's workers reuse them; nil allocates. Every buffer is fully
// written before it is read, except qNopeAbs, wsum and ctx, which mlaAbsorbAttend clears.
type mlaAbsorbScratch struct {
	qRope, qNopeAbs, scores, wsum, ctx []float32
}

func (s *mlaAbsorbScratch) take(buf *[]float32, n int) []float32 {
	if s == nil {
		return make([]float32, n)
	}
	if cap(*buf) < n {
		*buf = make([]float32, n)
	}
	return (*buf)[:n]
}

// mlaAbsorbAttend is the absorb path's attention for one query q (mlaQuery's layout) at position pos over the first nKeys prepared
// latents (mlaPrepLatents): it never reconstructs per-head K/V. kv_b_proj's per-head blocks are W_UK (the k_nope rows,
// [qkNope×rank]) and W_UV (the v rows, [vHead×rank]); two identities remove the per-key reconstruction:
//
//	q_nope · (W_UK·c) = (W_UKᵀ·q_nope) · c          ← score in rank-space
//	Σ_j p_j·(W_UV·c_j) = W_UV · (Σ_j p_j·c_j)        ← one W_UV matvec per head, post-softmax
//
// Cost O(nKeys·H·rank) + O(H·(qkNope+vHead)·rank). Algebraically equal to the naive path (cosine ~1.0; f32 reorder, not
// bit-exact). It returns the [H*v_head_dim] context, in s.ctx when s is non-nil. q's rope dims are not modified.
func mlaAbsorbAttend(op *mlaOpParams, p *mlaParams, w *mlaWeights, H int, q []float32, pos int, cn, kr []float32, nKeys int, s *mlaAbsorbScratch) []float32 {
	qkNope, qkRope, vHead := p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim
	qkHead := qkNope + qkRope
	rank := p.KVLoRARank
	hRow := qkNope + vHead // per-head output block in kv_b_proj (k_nope rows ‖ v rows)

	// 3. RoPE the query's rope dims (per head), gathered from the [nope|rope] layout.
	var qRope, qNopeAbs, scores, wsum, ctx []float32
	if s != nil {
		qRope, qNopeAbs, scores = s.take(&s.qRope, H*qkRope), s.take(&s.qNopeAbs, H*rank), s.take(&s.scores, H*nKeys)
		wsum, ctx = s.take(&s.wsum, H*rank), s.take(&s.ctx, H*vHead)
		clear(qNopeAbs)
		clear(wsum)
	} else {
		qRope, qNopeAbs, scores = make([]float32, H*qkRope), make([]float32, H*rank), make([]float32, H*nKeys)
		wsum, ctx = make([]float32, H*rank), make([]float32, H*vHead)
	}
	for hh := range H {
		copy(qRope[hh*qkRope:(hh+1)*qkRope], q[hh*qkHead+qkNope:hh*qkHead+qkHead])
	}
	mlaRope(qRope, H, qkRope, pos, op.InvFreq, op.Mscale, op.Interleave)

	// 4a. Absorb W_UK into the query: qNopeAbs[h] = W_UK_hᵀ · q_nope_h  ([rank] per head).
	// W_UK_h is the k_nope rows of head h: kv_b_proj rows [h*hRow : h*hRow+qkNope].
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

	// 4b. Score each latent in rank-space — no per-key K reconstruction.
	for j := range nKeys {
		c := cn[j*rank : (j+1)*rank]
		krj := kr[j*qkRope : (j+1)*qkRope]
		for hh := range H {
			qa := qNopeAbs[hh*rank : (hh+1)*rank]
			var sc float32
			for i := range rank {
				sc += qa[i] * c[i]
			}
			qr := qRope[hh*qkRope : (hh+1)*qkRope]
			for d := range qkRope {
				sc += qr[d] * krj[d]
			}
			scores[hh*nKeys+j] = sc * float32(op.Scale)
		}
	}

	// 5. Per-head softmax, then collapse V to a rank-space weighted latent sum.
	for hh := range H {
		row := scores[hh*nKeys : (hh+1)*nKeys]
		mx := float32(math.Inf(-1))
		for _, sc := range row {
			if sc > mx {
				mx = sc
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
			c := cn[j*rank : (j+1)*rank]
			for i := range rank {
				acc[i] += wgt * c[i]
			}
		}
	}

	// 6. Absorb W_UV: ctx_h = W_UV_h · wsum_h. W_UV_h is the v rows of head h:
	// kv_b_proj rows [h*hRow+qkNope : h*hRow+hRow].
	for hh := range H {
		acc := wsum[hh*rank : (hh+1)*rank]
		out := ctx[hh*vHead : (hh+1)*vHead]
		for e := range vHead {
			row := w.kvBProj[(hh*hRow+qkNope+e)*rank : (hh*hRow+qkNope+e)*rank+rank]
			var sc float32
			for i := range rank {
				sc += row[i] * acc[i]
			}
			out[e] = sc
		}
	}
	return ctx
}

// mlaForceNaive forces the naive (reconstruct-per-key) MLA path over the absorb default.
// It lives here (not on Model in the shared core file) so flipping the MLA path only
// touches the MLA families' parity deps. Set by TestMLAAbsorb_parity / _speed to compare
// the two paths; production leaves it false and uses the env override below. Tests in
// this package run serially, so a package var is safe.
var mlaForceNaive bool

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

// mlaBatchRuns counts mlaBatch calls that took the batched form, so its bit-identity gate can prove it ran.
var mlaBatchRuns atomic.Int64

// mlaPassScratch holds a batched pass's opMLA buffers across its layers (prefillPassScratch.mla).
type mlaPassScratch struct {
	q, qa, kv, cn, kr, ctx []float32
	rows                   []mlaAbsorbScratch // one per attention worker
}

// mlaBatch runs opMLA over the K rows of x ([K*hidden]) at positions startPos.., appending their latents, and writes each row's
// output to out: what mlaAttentionWith computes row by row, so a row's result is the per-token walker's (TestMLABatch_matchesSequential).
// The projections run at M=K (linalg.MatmulBT is M-invariant), the pass's latents are prepared once, and the rows attend in
// parallel, each over its own prefix. The naive path, an int8 projection and Bailing Hybrid's gate take the rows one at a time.
func (m *Model) mlaBatch(op *mlaOpParams, x, out []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, startPos, K int,
	ps *prefillPassScratch) {
	p, w := arch.mla, lw.mla
	hidden, H := arch.HiddenDim, arch.NumHeads
	if mlaForceNaive || m.knobs.mlaNaive() || w.gProj != nil ||
		w.qAQ.Rows() > 0 || w.qBQ.Rows() > 0 || w.qQ.Rows() > 0 || w.kvAQ.Rows() > 0 || w.oQ.Rows() > 0 {
		for i := range K {
			copy(out[i*hidden:(i+1)*hidden], m.mlaAttentionWith(op, x[i*hidden:(i+1)*hidden], lw, arch, cache, layer, startPos+i))
		}
		return
	}
	mlaBatchRuns.Add(1)
	rank, qkRope, vHead := p.KVLoRARank, p.QKRopeHeadDim, p.VHeadDim
	latDim, qkHead := rank+qkRope, p.QKNopeHeadDim+qkRope
	s := ps.mlaScratch()

	q := ps.take(&s.q, K*H*qkHead)
	if r := p.QLoRARank; r > 0 {
		qa := ps.take(&s.qa, K*r)
		linalg.MatmulBT(x, w.qAProj, qa, K, hidden, r)
		for i := range K {
			rmsNorm(qa[i*r:(i+1)*r], w.qALayernorm, 1, r, op.Eps, false)
		}
		linalg.MatmulBT(qa, w.qBProj, q, K, r, H*qkHead)
	} else {
		linalg.MatmulBT(x, w.qProj, q, K, hidden, H*qkHead)
	}
	kv := ps.take(&s.kv, K*latDim)
	linalg.MatmulBT(x, w.kvAProj, kv, K, hidden, latDim)
	for i := range K {
		cache.AppendLatent(layer, kv[i*latDim:(i+1)*latDim])
	}

	// Row i sees the latents the per-token walker would: everything stored before the pass and the pass's rows up to its own.
	lat := cache.Latent(layer)
	nTot := len(lat) / latDim
	cn, kr := ps.take(&s.cn, nTot*rank), ps.take(&s.kr, nTot*qkRope)
	mlaPrepLatents(op, p, w, lat, 0, nTot, cn, kr)

	ctx := ps.take(&s.ctx, K*H*vHead)
	workers := min(runtime.GOMAXPROCS(0), K)
	if len(s.rows) < workers {
		s.rows = make([]mlaAbsorbScratch, workers)
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for wi := range workers {
		wg.Go(func() {
			rs := &s.rows[wi]
			for {
				n := int(next.Add(1)) - 1
				if n >= K {
					return
				}
				i := K - 1 - n // the deepest rows first: they cost the most
				c := mlaAbsorbAttend(op, p, w, H, q[i*H*qkHead:(i+1)*H*qkHead], startPos+i, cn, kr, nTot-K+i+1, rs)
				copy(ctx[i*H*vHead:(i+1)*H*vHead], c)
			}
		})
	}
	wg.Wait()
	linalg.MatmulBT(ctx, w.oProj, out, K, H*vHead, hidden)
}
