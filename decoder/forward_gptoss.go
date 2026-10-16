package decoder

import (
	"math"
	"unsafe"
)

// gpt-oss's two operations that differ from the generic ones: attention adds a learned per-head SINK to the softmax denominator
// (an escape valve that bleeds attention mass, with no value), and the MoE experts use a clamped interleaved-SwiGLU with an
// α-scaled sigmoid, a +1 on the linear branch, and per-expert biases. Its program is otherwise the generic pre-norm layer
// (Architecture.layerOps). Parity-first and arch-neutral (no SIMD); see docs/completed/task-mxfp4-gptoss.md.

// attendSinksToken is the attnSinksF64 attention: it appends this position's k and v to layer l, then writes into ctx the
// attention of q over keys [start, pos], where start is p's window, with one extra logit per head, the learned sink, joining the
// max-shift and the denominator and carrying no value:
//
//	scores_s = (q·k_s)·scale               for s in [start, pos]
//	m        = max(max_s scores_s, sink_h)
//	denom    = Σ_s exp(scores_s − m) + exp(sink_h − m)
//	ctx      = Σ_s (exp(scores_s − m)/denom)·v_s
//
// Scores and the denominator accumulate in float64. It reads the layer's f32 rows as appended, so the cache keeps no ring for it
// (Architecture.kvRingsOK).
func attendSinksToken(p attentionParams, q, k, v, ctx []float32, cache *KVCache, arch *Architecture, lw *LayerWeights, l, pos int) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	cache.Append(l, k, v)
	keys, vals := cache.Keys(l), cache.Vals(l)
	nKeys := len(keys) / kvDim
	start := p.start(pos)
	scale := p.Scale
	group := nH / nKV

	clear(ctx)
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
}

// gptOssMoE runs the sparse FFN with arch's routing and expert constants; gptOssMoEWith is the program walker's form.
func (m *Model) gptOssMoE(h []float32, lw *LayerWeights, arch *Architecture) ([]float32, error) {
	return m.gptOssMoEWith(h, lw, &ffnParams{Form: ffnGptOss, MoE: *arch.MoE, Alpha: arch.gptoss.SwigluAlpha, Limit: arch.gptoss.SwigluLimit}, arch.HiddenDim)
}

// gptOssMoEWith runs the sparse FFN: a router (with a per-expert logit bias) selects the top-k experts, softmax over the selected
// logits gives the mixing weights, and the chosen gpt-oss experts are combined. Taking the softmax over ONLY the top-k logits is
// identical to softmax-over-all + renormalize (the normalizer cancels), so the weights match HF's GptOssTopKRouter exactly.
func (m *Model) gptOssMoEWith(h []float32, lw *LayerWeights, p *ffnParams, hidden int) ([]float32, error) {
	moe := &p.MoE
	logits := make([]float32, moe.NumExperts)
	matmul(m.be, &lw.Router, h, logits, 1)
	if lw.RouterBias != nil {
		addBias(logits, lw.RouterBias) // a plain router-logit bias, added before top-k
	}
	idx, topv := topK(logits, moe.TopK)
	wts := softmaxF32(topv)

	// Touch every routed expert before evaluating it, as moeMLP does, so the pager's budget and LRU see gpt-oss's demand.
	if m.pager != nil {
		for _, e := range idx {
			m.pager.touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	out := make([]float32, hidden)
	expOut := make([]float32, hidden)
	for j, e := range idx {
		gptOssExpertWith(m.be, &lw.Experts[e], h, expOut, moe.IntermediateDim, float32(p.Alpha), float32(p.Limit))
		w := wts[j]
		for i := range out {
			out[i] += w * expOut[i]
		}
	}
	return out, nil
}

// gptOssExpertWith evaluates one gpt-oss expert. Unlike the plain SwiGLU expert it clamps both branches, gates with an α-scaled
// sigmoid, adds 1 to the linear branch, and carries per-projection biases (matching HF GptOssExperts):
//
//	gate = clamp(Gate·h + gateBias, max=limit)
//	up   = clamp(Up·h + upBias, [-limit, limit])
//	glu  = gate · sigmoid(α·gate)
//	dst  = Down·((up + 1)·glu) + downBias
func gptOssExpertWith(be Backend, ex *expertWeights, h, dst []float32, inter int, alpha, limit float32) {
	gate := make([]float32, inter)
	up := make([]float32, inter)
	matmul(be, &ex.Gate, h, gate, 1)
	matmul(be, &ex.Up, h, up, 1)
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
	matmul(be, &ex.Down, gate, dst, 1)
	if ex.DownBias != nil {
		addBias(dst, ex.DownBias)
	}
}
