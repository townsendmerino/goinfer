package decoder

import (
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// gpt-oss's two operations that differ from the generic ones: attention adds a learned per-head SINK to the softmax denominator
// (an escape valve that bleeds attention mass, with no value), and the MoE experts use a clamped interleaved-SwiGLU with an
// α-scaled sigmoid, a +1 on the linear branch, and per-expert biases. Its program is otherwise the generic pre-norm layer
// (Architecture.layerOps). Parity-first and arch-neutral (no SIMD);

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
// (Architecture.kvRingsOK). Heads fan out over the attention workers, each with its own score buffer from the
// cache's scratch, and each head's arithmetic is the one-key-at-a-time loop's, so the result does not depend on the fan-out.
func attendSinksToken(p attentionParams, q, k, v, ctx []float32, cache *KVCache, arch *Architecture, lw *LayerWeights, l, pos int) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	cache.Append(l, k, v)
	keys, vals := cache.Keys(l), cache.Vals(l)
	nKeys := len(keys) / kvDim
	start := p.Start(pos)
	scale := p.Scale
	group := nH / nKV

	clear(ctx)
	head := func(qh int, scores []float32) {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		s := start
		var base [8]int
		var dots [8]float64
		for ; s+8 <= nKeys; s += 8 {
			for j := range base {
				base[j] = (s+j)*kvDim + kvh*hd
			}
			dotsF64x8(qHead, keys, &base, &dots)
			for j, dot := range dots {
				sc := dot * scale
				scores[s+j] = float32(sc)
				if sc > maxS {
					maxS = sc
				}
			}
		}
		for ; s < nKeys; s++ {
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
			addScaled(oHead, vals[base:base+hd], w)
		}
	}
	workers := scalarHeadWorkers(nH, nKeys-start, hd)
	if workers <= 1 || cache.scr == nil {
		var scores []float32
		if cache.scr != nil {
			scores = cache.scr.scoresBuf(nKeys)
		} else {
			scores = make([]float32, nKeys)
		}
		for qh := range nH {
			head(qh, scores)
		}
		return
	}
	pool := cache.scr.headScoresPool(workers, nKeys)
	runSplitAligned(len(pool), nH, 1, func(w, lo, hi int) {
		sc := pool[w].scores[:nKeys]
		for qh := lo; qh < hi; qh++ {
			head(qh, sc)
		}
	})
}

// gptOssMoE runs the sparse FFN with arch's routing and expert constants; gptOssMoEWith is the program walker's form.
func (m *Model) gptOssMoE(h []float32, lw *LayerWeights, arch *Architecture) ([]float32, error) {
	return m.gptOssMoEWith(h, lw, &ffnParams{Form: ffnGptOss, MoE: *arch.MoE, Alpha: arch.gptoss.SwigluAlpha, Limit: arch.gptoss.SwigluLimit}, arch.HiddenDim, nil)
}

// gptOssMoEWith runs the sparse FFN: a router (with a per-expert logit bias) selects the top-k experts, softmax over the selected
// logits gives the mixing weights, and the chosen gpt-oss experts are combined. Taking the softmax over ONLY the top-k logits is
// identical to softmax-over-all + renormalize (the normalizer cancels), so the weights match HF's GptOssTopKRouter exactly.
//
// When every chosen expert's gate and up run the CPU W4A8 kernel (gptOssExpertBatch), the 2k gate and up projections share
// one quantized h and one fork/join; the downs stay per expert, since their inputs differ. Each output column is the same dot
// product either way, so the result is bit-identical to the per-expert loop. The returned slice is scr's when scr is set,
// valid until the next call on scr; a nil scr allocates.
func (m *Model) gptOssMoEWith(h []float32, lw *LayerWeights, p *ffnParams, hidden int, scr *decodeScratch) ([]float32, error) {
	moe := &p.MoE
	var s *gptOssScratch
	if scr != nil {
		if scr.gptOss == nil {
			scr.gptOss = new(gptOssScratch)
		}
		s = scr.gptOss
	} else {
		s = new(gptOssScratch)
	}
	inter := moe.IntermediateDim
	s.size(moe.NumExperts, moe.TopK, inter, hidden)
	matmul(m.be, &lw.Router, h, s.logits, 1)
	if lw.RouterBias != nil {
		addBias(s.logits, lw.RouterBias) // a plain router-logit bias, added before top-k
	}
	idx, topv := topK(s.logits, moe.TopK)
	wts := softmaxF32(topv)

	// Touch every routed expert before evaluating it, as moeMLP does, so the pager's budget and LRU see gpt-oss's demand.
	if m.pager != nil {
		for _, e := range idx {
			m.pager.touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	alpha, limit := float32(p.Alpha), float32(p.Limit)
	out := s.out
	clear(out)
	batched := m.gptOssExpertBatch(lw, idx, h, s, scr)
	for j, e := range idx {
		ex := &lw.Experts[e]
		gate, up := s.gateUp[2*j*inter:(2*j+1)*inter], s.gateUp[(2*j+1)*inter:(2*j+2)*inter]
		if !batched {
			matmul(m.be, &ex.Gate, h, gate, 1)
			matmul(m.be, &ex.Up, h, up, 1)
		}
		gptOssExpertDown(m.be, ex, gate, up, s.expOut, alpha, limit)
		w := wts[j]
		for i := range out {
			out[i] += w * s.expOut[i]
		}
	}
	return out, nil
}

// gptOssScratch is gptOssMoEWith's per-stream scratch: router logits, the chosen experts' gate and up outputs (expert j's
// gate at gateUp[2j*inter:], its up right after), one expert's output, the combined output, and the batch's ops.
type gptOssScratch struct {
	logits, gateUp, expOut, out []float32
	ops                         []linalg.W4A8Op
}

// size makes s's buffers fit nExp experts, k chosen, inter and hidden; a call with the same shape allocates nothing.
func (s *gptOssScratch) size(nExp, k, inter, hidden int) {
	grow := func(b []float32, n int) []float32 {
		if cap(b) < n {
			return make([]float32, n)
		}
		return b[:n]
	}
	s.logits = grow(s.logits, nExp)
	s.gateUp = grow(s.gateUp, 2*k*inter)
	s.expOut = grow(s.expOut, hidden)
	s.out = grow(s.out, hidden)
	if cap(s.ops) < 2*k {
		s.ops = make([]linalg.W4A8Op, 2*k)
	}
	s.ops = s.ops[:2*k]
}

// gptOssExpertBatchCalls counts the layers whose expert gate and up ran as one batch, so a test can see the path was taken.
var gptOssExpertBatchCalls atomic.Int64

// gptOssExpertBatch runs the chosen experts' gate and up projections as one linalg.MatmulBTW4A8Batch into s.gateUp and
// reports true, or does nothing and reports false. It runs only under w4a8BatchEnabled and only when every one of the 2k
// weights takes the CPU W4A8 kernel (cpuW4A8) with the same K, weight group and activation group, so a staged backend's
// per-matmul consult and every other weight kind keep the per-expert path. M is 1, which is what makes a repacked-only
// (row4) expert safe in the batch (see quantizeBatchedProjWM).
func (m *Model) gptOssExpertBatch(lw *LayerWeights, idx []int, h []float32, s *gptOssScratch, scr *decodeScratch) bool {
	if !w4a8BatchEnabled || len(idx) == 0 {
		return false
	}
	inter := len(s.gateUp) / (2 * len(idx))
	first := &lw.Experts[idx[0]].Gate
	K, actGroup := first.Cols(), first.ActQuantGroup()
	group := -1
	for j, e := range idx {
		ex := &lw.Experts[e]
		for b, w := range [2]*linalg.WeightMat{&ex.Gate, &ex.Up} {
			if !cpuW4A8(m.be, w) || w.Cols() != K || w.ActQuantGroup() != actGroup || w.Rows() != inter {
				return false
			}
			o := 2*j + b
			op, g := wmW4A8Op(w, s.gateUp[o*inter:(o+1)*inter])
			if group >= 0 && g != group {
				return false
			}
			group = g
			s.ops[o] = op
		}
	}
	var ws *linalg.Workspace
	if scr != nil {
		ws = scr.ws
	} else {
		pooled, _ := matmulWorkspace(nil)
		defer matmulWSPool.Put(pooled)
		ws = pooled
	}
	ws.SetThreshold(int4ParThreshold)
	ws.SetActQuantGroup(actGroup)
	linalg.MatmulBTW4A8Batch(ws, h, 1, K, group, s.ops)
	gptOssExpertBatchCalls.Add(1)
	return true
}

// gptOssExpertDown finishes one gpt-oss expert from its gate and up projections (gate is overwritten with the activation).
// Unlike the plain SwiGLU expert it clamps both branches, gates with an α-scaled sigmoid, adds 1 to the linear branch, and
// carries per-projection biases (matching HF GptOssExperts):
//
//	gate = clamp(Gate·h + gateBias, max=limit)
//	up   = clamp(Up·h + upBias, [-limit, limit])
//	glu  = gate · sigmoid(α·gate)
//	dst  = Down·((up + 1)·glu) + downBias
func gptOssExpertDown(be Backend, ex *expertWeights, gate, up, dst []float32, alpha, limit float32) {
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
