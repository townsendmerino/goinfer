package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// g4traceHidden, when non-nil, receives the residual stream after each layer
// (layer -1 = post-embedding) on every Gemma 4 token through the program walker. Debug/test only:
// set it around the final token's forward to capture a last-position layer trace
// for diffing against the HF reference. The callback must copy h if it retains it.
var g4traceHidden func(layer int, h []float32)

// g4traceQKV, when non-nil, receives the post-norm/RoPE q, k, v for each layer on
// every Gemma 4 token (k/v nil on KV-shared layers). Debug/test only.
var g4traceQKV func(layer int, q, k, v []float32)

// Gemma 4 diverges from the generic layer in per-layer head_dim and KV-head count, scale-less v-norm, cross-layer KV sharing (the
// last N layers reuse an earlier layer's KV), per-attention-type RoPE (proportional / partial-rotary on the global layers), the
// Per-Layer-Embedding (PLE) residual branch, and a per-layer output scalar. Its program is Architecture.gemma4LayerOps; the
// operations below are its own. Buffers are allocated per call (parity-first, not the perf path).

// gemma4Token is one Gemma 4 position's state across its layers: the quantized activation shared by a layer's q/k/v and by its
// gate/up, the attention scratch (sized to the widest head_dim and this position's key count), and the PLE inputs.
type gemma4Token struct {
	hq       w4a8Act
	sc       g4attnScratch
	perLayer []float32
	pleDim   int
}

// newGemma4Token makes position pos's state. The PLE inputs are computed from h, the embedding, so it is made before layer 0.
func (m *Model) newGemma4Token(prog *layerProgram, h []float32, pleID, pos int) *gemma4Token {
	arch := m.w.arch
	g := arch.gemma4
	nH := arch.NumHeads
	maxHd := max(g.GlobalHeadDim, arch.HeadDim)
	nkMax := pos + 1
	t := &gemma4Token{pleDim: g.HiddenSizePerLayerInput, sc: g4attnScratch{
		kh:     make([]float32, nkMax*maxHd),
		vt:     make([]float32, nkMax*maxHd),
		qstack: make([]float32, nH*maxHd),
		scores: make([]float32, nH*nkMax),
		cstack: make([]float32, nH*maxHd),
	}}
	t.perLayer = make([]float32, arch.NumLayers*t.pleDim)
	if prog.Embed.PLE && t.pleDim > 0 {
		m.gemma4PLEInputs(h, pleID, t.perLayer)
	}
	return t
}

// ple is layer l's PLE input.
func (t *gemma4Token) ple(l int) []float32 { return t.perLayer[l*t.pleDim : (l+1)*t.pleDim] }

// gemma4AttnToken is one Gemma 4 attention sublayer (opGemma4Attn) for position pos, x → out: q (normed, roped); on a layer that
// owns its KV, k (normed, roped) and v (from k when the checkpoint ties them, then normed without a weight), appended; then
// attention over p.KVSrc's keys from the window's start, and o_proj.
func (m *Model) gemma4AttnToken(p *gemma4AttnParams, x, out []float32, lw *LayerWeights, cache *KVCache, l, pos int, t *gemma4Token) {
	arch, be := m.w.arch, m.be
	nH, hd, nKV := arch.NumHeads, arch.headDimAt(l), arch.kvHeadsAt(l)
	q := make([]float32, nH*hd)
	t.hq.prepare(be, &lw.QProj, x, 1)
	matmulPre(be, &lw.QProj, &t.hq, x, q, 1)
	rmsNorm(q, lw.QNorm, nH, hd, p.Eps, p.AddOne)
	applyRoPE(q, nH, hd, pos, p.InvFreq, p.Mscale)
	if p.OwnsKV {
		k := make([]float32, nKV*hd)
		matmulPre(be, &lw.KProj, &t.hq, x, k, 1)
		v := make([]float32, nKV*hd)
		if lw.VFromK { // attention_k_eq_v: V is v_norm(k_proj output)
			copy(v, k)
		} else {
			matmulPre(be, &lw.VProj, &t.hq, x, v, 1)
		}
		rmsNorm(k, lw.KNorm, nKV, hd, p.Eps, p.AddOne)
		applyRoPE(k, nKV, hd, pos, p.InvFreq, p.Mscale)
		rmsNormNoWeight(v, nKV, hd, p.Eps) // scale-less v_norm, no RoPE
		cache.Append(l, k, v)
		if g4traceQKV != nil {
			g4traceQKV(l, q, k, v)
		}
	} else if g4traceQKV != nil {
		g4traceQKV(l, q, nil, nil)
	}
	keys, vals := cache.Keys(p.KVSrc), cache.Vals(p.KVSrc)
	nKeys := len(keys) / (nKV * hd)
	ctx := make([]float32, nH*hd)
	start := 0
	if p.Window > 0 {
		start = max(pos-p.Window+1, 0)
	}
	gemma4Attend(q, ctx, keys, vals, nH, nKV, hd, start, nKeys, p.Scale, &t.sc)
	matmul(be, &lw.OProj, ctx, out, 1)
}

// gemma4DenseFFN is a dense Gemma 4 layer's GeGLU at its own width, x → out, sharing the position's quantized activation.
func (m *Model) gemma4DenseFFN(act ActKind, x, out []float32, lw *LayerWeights, width int, hq *w4a8Act) error {
	if act != ActGeluTanh {
		return fmt.Errorf("decoder: Gemma 4's dense FFN is GELU-tanh, not activation %d", act)
	}
	be := m.be
	gate := make([]float32, width)
	up := make([]float32, width)
	hq.prepare(be, &lw.GateProj, x, 1)
	matmulPre(be, &lw.GateProj, hq, x, gate, 1)
	matmulPre(be, &lw.UpProj, hq, x, up, 1)
	if len(gate) < activationFanoutThreshold {
		geglu(gate, up)
	} else {
		parallelElementwise(len(gate), func(lo, hi int) {
			geglu(gate[lo:hi], up[lo:hi])
		})
	}
	matmul(be, &lw.DownProj, gate, out, 1)
	return nil
}

// gemma4PLEBranch adds the per-layer-embedding branch to h: gate → gelu × the layer's PLE input → proj → norm → +residual. sub is
// scratch of the hidden width.
func (m *Model) gemma4PLEBranch(h, pl []float32, lw *LayerWeights, ne *normEpsParams, sub []float32) {
	px := make([]float32, len(pl))
	matmul(m.be, &lw.PLEGate, h, px, 1)
	parallelElementwise(len(px), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			px[i] = geluTanh(px[i]) * pl[i]
		}
	})
	matmul(m.be, &lw.PLEProj, px, sub, 1)
	rmsNormInto(sub, sub, lw.PostPLENorm, 1, len(h), ne.Eps, ne.AddOne)
	addResidual(h, sub)
}

// gemma4InvFreq builds the RoPE inverse-frequency table. For full rotary,
// rotaryDim == headDim and the result is headDim/2 entries. For proportional
// (partial) rotary, rotaryDim < headDim: the first rotaryDim/2 entries are the
// real frequencies (spaced by 1/headDim, NOT 1/rotaryDim) and the rest are zero,
// so applyRoPE (which rotates len(invFreq) pairs split at headDim/2) leaves the
// trailing dims unrotated — matching HF's "proportional" RoPE.
func gemma4InvFreq(headDim, rotaryDim int, base float64) []float64 {
	half := headDim / 2
	inv := make([]float64, half)
	angles := rotaryDim / 2
	for i := range angles {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
	}
	return inv // entries [angles:half] stay 0 → no rotation there
}

// rmsNormNoWeight applies a scale-less RMSNorm per head (Gemma 4's v_norm:
// with_scale=False — normalize V by RMS, no learned weight).
func rmsNormNoWeight(x []float32, rows, dim int, eps float64) {
	for r := range rows {
		v := x[r*dim : r*dim+dim]
		var ss float64
		for _, e := range v {
			ss += float64(e) * float64(e)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		for i := range v {
			v[i] *= inv
		}
	}
}

// g4attnScratch holds the batched-over-heads gemma4 attention buffers: the gathered K_head [n,hd] / V_headᵀ [hd,n] for one KV
// head, the stacked query-head group, its scores, and the context. Allocated once per token (newGemma4Token, sized to the
// widest head_dim and current key count) and reused across layers.
type g4attnScratch struct {
	kh, vt, qstack, scores, cstack []float32
}

// gemma4Attend computes one token's grouped-query attention over keys
// [start, nKeys), with explicit per-layer dims (head_dim / KV-head count vary by
// layer). It batches the query heads sharing a KV head into one A·Bᵀ matmul: the
// token is a single query position, so every head attends the same key range
// (no causal mask). Per KV head — gather K_head [n,hd] + V_headᵀ [hd,n] once,
// QKᵀ = MatmulBT(group_q, K_head), a scaled softmax per row, scores·V =
// MatmulBT(scores, V_headᵀ), scatter into ctx. This moves the QKᵀ/scores·V
// scalar triple-loops (the per-token O(L²) prefill cost) onto the SIMD kernel.
// scale is arch.AttnScale (1.0 for gemma4 — the q/k-norm weights absorb it).
// Parity is argmax + cosine, not bit-identical (float64→f32 SIMD).
func gemma4Attend(q, ctx, keys, vals []float32, nH, nKV, hd, start, nKeys int, scale float64, sc *g4attnScratch) {
	kvDim := nKV * hd
	group := nH / nKV
	n := nKeys - start
	for kvh := range nKV {
		kh, vt := sc.kh[:n*hd], sc.vt[:hd*n]
		for j := range n {
			base := (start+j)*kvDim + kvh*hd
			copy(kh[j*hd:j*hd+hd], keys[base:base+hd])
			vrow := vals[base : base+hd]
			for d := range hd {
				vt[d*n+j] = vrow[d]
			}
		}
		qs := sc.qstack[:group*hd]
		for g := range group {
			qhead := kvh*group + g
			copy(qs[g*hd:g*hd+hd], q[qhead*hd:qhead*hd+hd])
		}
		// QKᵀ: scores[group,n] = group_q[group,hd] · K_head[n,hd]ᵀ
		scores := sc.scores[:group*n]
		linalg.MatmulBT(qs, kh, scores, group, hd, n)
		for g := range group {
			rowS := scores[g*n : g*n+n]
			maxS := math.Inf(-1)
			for j := range n {
				v := float64(rowS[j]) * scale
				rowS[j] = float32(v)
				if v > maxS {
					maxS = v
				}
			}
			var sum float64
			for j := range n {
				e := math.Exp(float64(rowS[j]) - maxS)
				rowS[j] = float32(e)
				sum += e
			}
			inv := 1.0 / sum
			for j := range n {
				rowS[j] = float32(float64(rowS[j]) * inv)
			}
		}
		// scores·V: ctx_group[group,hd] = scores[group,n] · V_head[n,hd]
		//                               = MatmulBT(scores, V_headᵀ[hd,n])
		cs := sc.cstack[:group*hd]
		linalg.MatmulBT(scores, vt, cs, group, n, hd)
		for g := range group {
			qhead := kvh*group + g
			copy(ctx[qhead*hd:qhead*hd+hd], cs[g*hd:g*hd+hd])
		}
	}
}

// gemma4PLEInputs writes one position's Per-Layer-Embedding inputs into dst ([NumLayers*pleDim]):
// (token_identity + context_aware) / √2, where token_identity = per_layer_token_embd[pleTokenID] × √pleDim and
// context_aware = RMSNorm per layer segment of (per_layer_model_proj(h) × hidden^-0.5). h is the position's inputs_embeds (already
// × EmbedScale). The CPU forward and the resident embedding (embedResidentInto, which hands a GPU backend [h ‖ these]) both call
// it, so the two cannot compute PLE differently.
func (m *Model) gemma4PLEInputs(h []float32, pleTokenID int, dst []float32) {
	m.gemma4PLEInputsScratch(h, pleTokenID, dst, nil)
}

// gemma4PLEInputsScratch is gemma4PLEInputs with its [NumLayers·pleDim] context-aware buffer taken from scratch when it is that
// long (a prefill embedding row after row, MEM6), else allocated.
func (m *Model) gemma4PLEInputsScratch(h []float32, pleTokenID int, dst, scratch []float32) {
	arch := m.w.arch
	pleDim := arch.gemma4.HiddenSizePerLayerInput
	m.w.PerLayerTokenEmbed.Row(pleTokenID, dst)
	tScale := float32(math.Sqrt(float64(pleDim)))
	for i := range dst {
		dst[i] *= tScale
	}
	ctxAware := scratch
	if len(ctxAware) != arch.NumLayers*pleDim {
		ctxAware = make([]float32, arch.NumLayers*pleDim)
	}
	matmul(m.be, &m.w.PerLayerModelProj, h, ctxAware, 1)
	cScale := float32(1.0 / math.Sqrt(float64(arch.HiddenDim)))
	for i := range ctxAware {
		ctxAware[i] *= cScale
	}
	inv2 := float32(1.0 / math.Sqrt2)
	for l := 0; l < arch.NumLayers; l++ {
		row := ctxAware[l*pleDim : (l+1)*pleDim]
		normalize(arch, row, m.w.PerLayerProjNorm, nil, pleDim) // RMSNorm over pleDim
		pl := dst[l*pleDim : (l+1)*pleDim]
		for i := range pl {
			pl[i] = (pl[i] + row[i]) * inv2
		}
	}
}
