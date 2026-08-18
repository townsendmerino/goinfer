package decoder

import (
	"fmt"
	"math"
)

// gpt-oss forward (own path, CPU-only). gpt-oss is a sparse-MoE family whose two
// ops diverge from the generic descriptor forward: attention adds a learned
// per-head SINK to the softmax denominator (an escape valve that bleeds attention
// mass, with no value), and the MoE experts use a clamped interleaved-SwiGLU with
// an α-scaled sigmoid, a +1 on the linear branch, and per-expert biases. The layer
// skeleton is otherwise plain pre-norm (NormPre2): everything else — embedding,
// norms, residuals, final head — is the shared path. Isolating gpt-oss here keeps
// the sink/clamped-activation out of the 20 other families' hot softmax/MLP kernels.
// Parity-first and arch-neutral (no SIMD): §1 says the capability matters more than
// speed on x86, and bench numbers are deferred (docs/task-mxfp4-gptoss.md §6.6).

// runLayersGptOss embeds token id and runs the block stack, returning the residual
// stream after the last layer (pre-final-norm) — the same contract as runLayers, so
// forward() applies the final norm + LM head unchanged.
func (m *Model) runLayersGptOss(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if m.w.Embed.Rows() == 0 {
		return nil, fmt.Errorf("decoder.forward(gptoss): weights not loaded %w [M1]", errNotImplemented)
	}
	if cache.scr == nil { // caches from NewKVCache directly (tests); Model.NewCache sets it
		cache.scr = newDecodeScratch(arch)
	}
	hidden := arch.HiddenDim
	h := make([]float32, hidden)
	m.w.Embed.Row(id, h) // no embedding scale, no learned positions (gpt-oss is RoPE)

	norm := make([]float32, hidden)
	attnOut := make([]float32, hidden)
	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]

		// h += attn(preAttnNorm(h))
		copy(norm, h)
		normalize(arch, norm, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		if err := m.gptOssAttention(l, norm, attnOut, lw, cache); err != nil {
			return nil, err
		}
		for i := range h {
			h[i] += attnOut[i]
		}

		// h += moe(preMLPNorm(h))
		copy(norm, h)
		normalize(arch, norm, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		moeOut, err := m.gptOssMoE(norm, lw, arch)
		if err != nil {
			return nil, err
		}
		for i := range h {
			h[i] += moeOut[i]
		}
		cache.captureResidual(l, h)
	}
	return h, nil
}

// gptOssAttention runs one layer's GQA attention with a per-head learned sink. The
// query at position pos attends over keys in [WindowStart(pos), pos] — a sliding
// window on the local (even) layers, the full prefix on the global (odd) ones — and
// each head's softmax denominator gains one extra term exp(sink_h) with no value:
//
//	scores_s = (q·k_s)·scale               for s in [start, pos]
//	m        = max(max_s scores_s, sink_h)
//	denom    = Σ_s exp(scores_s − m) + exp(sink_h − m)
//	ctx      = Σ_s (exp(scores_s − m)/denom)·v_s
//
// f64 score/denominator accumulation (parity-first). K/V are append-forever f32
// (the cache keeps rings off for gpt-oss); the window is applied at read time.
func (m *Model) gptOssAttention(layer int, h, out []float32, lw *LayerWeights, cache *KVCache) error {
	arch := m.w.arch
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	global := arch.isGlobalLayer(layer)
	pos := cache.Pos()

	q := make([]float32, qDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	matmul(m.be, &lw.QProj, h, q, 1)
	matmul(m.be, &lw.KProj, h, k, 1)
	matmul(m.be, &lw.VProj, h, v, 1)
	addBias(q, lw.QBias) // gpt-oss carries q/k/v biases
	addBias(k, lw.KBias)
	addBias(v, lw.VBias)

	// RoPE (YaRN): the per-layer inv-freq table with the mscale (attention_factor)
	// folded into cos/sin. gpt-oss has no QK-norm.
	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)
	ropeAt(q, nH, hd, pos, invFreq, ms, nil, nil, 0, arch.ropeInterleave)
	ropeAt(k, nKV, hd, pos, invFreq, ms, nil, nil, 0, arch.ropeInterleave)

	// Append this position's K/V, then read the (windowed) history.
	cache.Append(layer, k, v)
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	start := cache.WindowStart(pos, global)
	scale := arch.AttnScale
	group := nH / nKV

	ctx := make([]float32, qDim)
	scores := make([]float32, nKeys)
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			base := s*kvDim + kvh*hd
			kHead := keys[base : base+hd]
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
		// The sink is an extra logit in the softmax (no key/value); it participates
		// in the max-shift and the denominator, then is dropped.
		sinkLogit := float64(lw.AttnSinks[qh])
		if sinkLogit > maxS {
			maxS = sinkLogit
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		sum += math.Exp(sinkLogit - maxS)
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			base := s*kvDim + kvh*hd
			vHead := vals[base : base+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}

	matmul(m.be, &lw.OProj, ctx, out, 1)
	if lw.OBias != nil {
		addBias(out, lw.OBias)
	}
	return nil
}

// gptOssMoE runs the sparse FFN: a router (with a per-expert logit bias) selects the
// top-k experts, softmax over the selected logits gives the mixing weights, and the
// chosen gpt-oss experts are combined. Taking the softmax over ONLY the top-k logits
// is identical to softmax-over-all + renormalize (the normalizer cancels), so the
// weights match HF's GptOssTopKRouter exactly.
func (m *Model) gptOssMoE(h []float32, lw *LayerWeights, arch *Architecture) ([]float32, error) {
	moe := arch.MoE
	hidden := arch.HiddenDim

	logits := make([]float32, moe.NumExperts)
	matmul(m.be, &lw.Router, h, logits, 1)
	if lw.RouterBias != nil {
		addBias(logits, lw.RouterBias) // gpt-oss: a plain router-logit bias (added before top-k)
	}
	idx, topv := topK(logits, moe.TopK)
	wts := softmaxF32(topv) // softmax over the top-k selected logits

	out := make([]float32, hidden)
	expOut := make([]float32, hidden)
	for j, e := range idx {
		m.gptOssExpert(&lw.Experts[e], h, expOut, moe.IntermediateDim)
		w := wts[j]
		for i := range out {
			out[i] += w * expOut[i]
		}
	}
	return out, nil
}

// gptOssExpert evaluates one gpt-oss expert into dst[:hidden]. Unlike the plain
// SwiGLU expert it clamps both branches, gates with an α-scaled sigmoid, adds 1 to
// the linear branch, and carries per-projection biases (matching HF GptOssExperts):
//
//	gate = clamp(Gate·h + gateBias, max=limit)
//	up   = clamp(Up·h + upBias, [-limit, limit])
//	glu  = gate · sigmoid(α·gate)
//	dst  = Down·((up + 1)·glu) + downBias
func (m *Model) gptOssExpert(ex *expertWeights, h, dst []float32, inter int) {
	gp := m.w.arch.gptoss
	alpha := float32(gp.SwigluAlpha)
	limit := float32(gp.SwigluLimit)

	gate := make([]float32, inter)
	up := make([]float32, inter)
	matmul(m.be, &ex.Gate, h, gate, 1)
	matmul(m.be, &ex.Up, h, up, 1)
	if ex.GateBias != nil {
		addBias(gate, ex.GateBias)
	}
	if ex.UpBias != nil {
		addBias(up, ex.UpBias)
	}
	for i := range gate {
		g := gate[i]
		if g > limit { // clamp max only
			g = limit
		}
		u := up[i]
		if u > limit { // clamp both sides
			u = limit
		} else if u < -limit {
			u = -limit
		}
		glu := g * float32(1.0/(1.0+math.Exp(-float64(alpha*g))))
		gate[i] = (u + 1) * glu
	}
	matmul(m.be, &ex.Down, gate, dst, 1)
	if ex.DownBias != nil {
		addBias(dst, ex.DownBias)
	}
}
