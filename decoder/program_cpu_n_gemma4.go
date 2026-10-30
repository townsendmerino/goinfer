package decoder

import (
	"fmt"
	"math"
)

// batchRows is what a batched pass knows about its rows beyond their embeddings: the token id behind each row and the prompt's
// bidirectional image or audio spans, in absolute positions of a prompt that starts at 0. Only Gemma 4's operations read it.
type batchRows struct {
	ids   []int
	spans []ImageSpan
}

// gemma4Pass is a batched pass's Gemma 4 state: each row's PLE inputs, computed from the embeddings before layer 0, and the
// attention scratch every (row, layer) call reuses.
type gemma4Pass struct {
	rows      *batchRows
	perLayer  []float32 // [K, NumLayers, pleDim]; nil without PLE
	rowStride int       // NumLayers*pleDim
	pleDim    int
	sc        g4attnScratch
}

// newGemma4Pass computes the pass's PLE inputs from h ([K*hidden], the embeddings with image rows spliced) and sizes the attention
// scratch for the widest range a row can attend (startPos+K keys). A row's PLE input is (token_identity + context_aware) / √2;
// context_aware depends only on the row's own embedding, so it is one M=K matmul. An image row takes the pad id's token identity.
func (m *Model) newGemma4Pass(prog *layerProgram, h []float32, rows *batchRows, startPos, K int) *gemma4Pass {
	arch := m.w.arch
	g4 := arch.gemma4
	hidden, nH := arch.HiddenDim, arch.NumHeads
	maxHd := max(g4.GlobalHeadDim, arch.HeadDim)
	nkMax := startPos + K
	gp := &gemma4Pass{rows: rows, pleDim: g4.HiddenSizePerLayerInput, rowStride: arch.NumLayers * g4.HiddenSizePerLayerInput, sc: g4attnScratch{
		kh:     make([]float32, nkMax*maxHd),
		vt:     make([]float32, nkMax*maxHd),
		qstack: make([]float32, nH*maxHd),
		scores: make([]float32, nH*nkMax),
		cstack: make([]float32, nH*maxHd),
	}}
	if !prog.Embed.PLE || gp.pleDim == 0 {
		return gp
	}
	pleDim, rowStride := gp.pleDim, gp.rowStride
	gp.perLayer = make([]float32, K*rowStride)
	ctxAware := make([]float32, K*rowStride)
	matmul(m.be, &m.w.PerLayerModelProj, h, ctxAware, K)
	cScale := float32(1.0 / math.Sqrt(float64(hidden)))
	for i := range ctxAware {
		ctxAware[i] *= cScale
	}
	tScale := float32(math.Sqrt(float64(pleDim)))
	inv2 := float32(1.0 / math.Sqrt2)
	for row := range K {
		tid := rows.ids[row]
		if _, in := imageSpanRow(rows.spans, startPos+row); in && walkBatchPlantForTest != plantBatchImagePLEToken {
			tid = g4.PadTokenID
		}
		rowPL := gp.perLayer[row*rowStride : (row+1)*rowStride]
		m.w.PerLayerTokenEmbed.Row(tid, rowPL)
		for i := range rowPL {
			rowPL[i] *= tScale
		}
		rowCtx := ctxAware[row*rowStride : (row+1)*rowStride]
		for l := range arch.NumLayers {
			seg := rowCtx[l*pleDim : (l+1)*pleDim]
			normalize(arch, seg, m.w.PerLayerProjNorm, nil, pleDim)
			plSeg := rowPL[l*pleDim : (l+1)*pleDim]
			for i := range plSeg {
				plSeg[i] = (plSeg[i] + seg[i]) * inv2
			}
		}
	}
	return gp
}

// gemma4BatchRange is the inclusive key range [lo, hi] a query at pos may attend: the causal range from the window's start
// (window 0 is global), widened to the whole block when pos is inside a bidirectional span. The union is one interval, since pos
// is in both. This is transformers' or_masks(windowed_causal, blockwise_overlay) (masking_utils.py), which applies to global and
// sliding layers alike (docs/multimodal.md, P7 entry); a query outside every span gets the plain range.
func gemma4BatchRange(pos, window int, spans []ImageSpan) (lo, hi int) {
	hi = pos
	if window > 0 {
		lo = max(pos-window+1, 0)
	}
	for _, s := range spans {
		if pos >= s.Pos && pos < s.Pos+s.Len {
			lo = min(lo, s.Pos)
			if walkBatchPlantForTest != plantBatchNoBidirectional {
				hi = max(hi, s.Pos+s.Len-1)
			}
			break
		}
	}
	return
}

// gemma4AttnBatch is opGemma4Attn over K rows of x into out: q at M=K, normed and rotated per row; on a layer that owns its KV, k
// and v likewise, every row appended before any row attends, so a row inside a span sees the block's later rows; then each row
// attends p.KVSrc's keys over its gemma4BatchRange, and o_proj at M=K.
func (m *Model) gemma4AttnBatch(p *gemma4AttnParams, x, out []float32, lw *LayerWeights, cache *KVCache, l, startPos, K int, gp *gemma4Pass, hq *w4a8Act) {
	arch, be := m.w.arch, m.be
	nH, hd, nKV := arch.NumHeads, arch.headDimAt(l), arch.kvHeadsAt(l)
	q := make([]float32, K*nH*hd)
	hq.prepare(be, &lw.QProj, x, K)
	matmulPre(be, &lw.QProj, hq, x, q, K)
	rmsNorm(q, lw.QNorm, K*nH, hd, p.Eps, p.AddOne)
	for row := range K {
		applyRoPE(q[row*nH*hd:(row+1)*nH*hd], nH, hd, startPos+row, p.InvFreq, p.Mscale)
	}
	if p.OwnsKV {
		k := make([]float32, K*nKV*hd)
		matmulPre(be, &lw.KProj, hq, x, k, K)
		v := make([]float32, K*nKV*hd)
		if lw.VFromK { // attention_k_eq_v: V is v_norm(k_proj output)
			copy(v, k)
		} else {
			matmulPre(be, &lw.VProj, hq, x, v, K)
		}
		rmsNorm(k, lw.KNorm, K*nKV, hd, p.Eps, p.AddOne)
		rmsNormNoWeight(v, K*nKV, hd, p.Eps) // scale-less v_norm, no RoPE
		for row := range K {
			kRow := k[row*nKV*hd : (row+1)*nKV*hd]
			applyRoPE(kRow, nKV, hd, startPos+row, p.InvFreq, p.Mscale)
			cache.Append(l, kRow, v[row*nKV*hd:(row+1)*nKV*hd])
		}
	}
	keys, vals := cache.Keys(p.KVSrc), cache.Vals(p.KVSrc)
	ctx := make([]float32, K*nH*hd)
	for row := range K {
		lo, hi := gemma4BatchRange(startPos+row, p.Window, gp.rows.spans)
		gemma4Attend(q[row*nH*hd:(row+1)*nH*hd], ctx[row*nH*hd:(row+1)*nH*hd], keys, vals, nH, nKV, hd, lo, hi+1, p.Scale, &gp.sc)
	}
	matmul(be, &lw.OProj, ctx, out, K)
}

// gemma4DenseFFNBatch is a dense Gemma 4 layer's GeGLU over K rows at the layer's own width, x → out, gate and up sharing the
// rows' quantized activation.
func (m *Model) gemma4DenseFFNBatch(act ActKind, x, out []float32, lw *LayerWeights, width, K int, hq *w4a8Act) error {
	if act != ActGeluTanh {
		return fmt.Errorf("decoder: Gemma 4's dense FFN is GELU-tanh, not activation %d", act)
	}
	be := m.be
	gate := make([]float32, K*width)
	up := make([]float32, K*width)
	hq.prepare(be, &lw.GateProj, x, K)
	matmulPre(be, &lw.GateProj, hq, x, gate, K)
	matmulPre(be, &lw.UpProj, hq, x, up, K)
	parallelElementwise(len(gate), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
	})
	matmul(be, &lw.DownProj, gate, out, K)
	return nil
}

// gemma4PLEBatch is opPLE over the K rows of h, in place: gate → GELU-tanh × the row's layer-l PLE input → proj → norm → + the
// residual.
func (m *Model) gemma4PLEBatch(h []float32, l, K int, lw *LayerWeights, ne *normEpsParams, gp *gemma4Pass) {
	be, hidden, pleDim := m.be, m.w.arch.HiddenDim, gp.pleDim
	px := make([]float32, K*pleDim)
	matmul(be, &lw.PLEGate, h, px, K)
	for row := range K {
		pl := gp.perLayer[row*gp.rowStride+l*pleDim : row*gp.rowStride+(l+1)*pleDim]
		pxRow := px[row*pleDim : (row+1)*pleDim]
		for i := range pxRow {
			pxRow[i] = geluTanh(pxRow[i]) * pl[i]
		}
	}
	pout := make([]float32, K*hidden)
	matmul(be, &lw.PLEProj, px, pout, K)
	for row := range K {
		r := pout[row*hidden : (row+1)*hidden]
		rmsNormInto(r, r, lw.PostPLENorm, 1, hidden, ne.Eps, ne.AddOne)
	}
	for i := range h {
		h[i] += pout[i]
	}
}
