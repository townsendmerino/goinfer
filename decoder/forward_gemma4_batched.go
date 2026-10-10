package decoder

import (
	"context"
	"fmt"
	"math"
)

// gemma4AttendRange returns the inclusive absolute key range [lo,hi] a query at pos (on a layer with the given global/window
// setting) may attend to, given the bidirectional image/audio blocks in spans. With no block containing pos it is the plain
// causal/windowed range, WindowStart(pos, global)..pos, the same range the sequential path implicitly uses.
//
// A query inside a block sees its own block only, and the result is always a single interval, never two disjoint ones: for a query
// at pos in block [b0,b1) the causal/windowed interval is [windowStart(pos), pos] and the block interval is [b0, b1-1]; since
// b0 <= pos <= b1-1 and windowStart(pos) <= pos, pos is in both, so the union is connected:
// [min(windowStart(pos), b0), max(pos, b1-1)]. A query outside every block gets the plain range (HF's blockwise term requires
// block[q]>=0 too). This matches transformers' masking_utils.py, whose create_causal_mask and create_sliding_window_causal_mask
// both apply or_masks(windowed_causal, blockwise_overlay(block_ids)) with no layer-type gate, so global and sliding layers both
// get this treatment (docs/multimodal.md, P7 entry).
func gemma4AttendRange(cache *KVCache, pos int, global bool, spans []ImageSpan) (lo, hi int) {
	lo = cache.WindowStart(pos, global)
	hi = pos
	for _, s := range spans {
		if pos >= s.Pos && pos < s.Pos+s.Len {
			if s.Pos < lo {
				lo = s.Pos
			}
			if end := s.Pos + s.Len - 1; end > hi {
				hi = end
			}
			break
		}
	}
	return
}

// runLayersGemma4FromEmbedN is runLayersGemma4FromEmbed's batched twin: K pre-embedded rows (h, [K*HiddenDim], image positions
// already spliced with raw projected features by the caller) processed as one prefill pass instead of K sequential per-token
// calls. That is the only way to let a query inside an image/audio block see later block positions (gemma4AttendRange), which a
// strict sequential KV-append forward cannot express.
//
// It assumes a fresh cache (cache.Pos()==0 at entry; its caller prefillLogitsGemma4VLBidirectional always passes one, as the
// sequential twin prefillLogitsGemma4VL does). ids[row] is the row's real token id, used only for PLE's token-identity lookup
// (image rows substitute arch.gemma4.PadTokenID, mirroring the sequential path; see runLayersGemma4FromEmbed). gemma4Attend, the
// unchanged single-query-row kernel, is reused per row with that row's gemma4AttendRange; it already accepts an arbitrary
// [start,nKeys) range into the full cache arrays, so no new attention math is introduced.
func (m *Model) runLayersGemma4FromEmbedN(reqCtx context.Context, h []float32, ids []int, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	g4 := arch.gemma4
	be := m.be
	hidden := arch.HiddenDim
	nH := arch.NumHeads
	pleDim := g4.HiddenSizePerLayerInput
	K := len(ids)
	startPos := cache.Pos()
	if len(h) != K*hidden {
		return nil, fmt.Errorf("decoder: runLayersGemma4FromEmbedN: h len %d, want %d (%d rows x %d hidden)", len(h), K*hidden, K, hidden)
	}

	localInv := gemma4InvFreq(arch.HeadDim, arch.HeadDim, arch.RoPELocalBase)
	globalInv := gemma4InvFreq(g4.GlobalHeadDim, g4.GlobalRotaryDim, arch.RoPEGlobalBase)

	// PLE per-row inputs: (token_identity + context_aware) / √2, per row per layer.
	// context_aware depends only on the row's OWN initial embedding, so it batches
	// as one matmul at M=K exactly like the dense projections below; the rest is
	// cheap per-row embedding-table lookups + an RMSNorm, not matmuls.
	rowStride := arch.NumLayers * pleDim
	var perLayer []float32 // [K, NumLayers, pleDim], nil when pleDim==0
	if pleDim > 0 {
		perLayer = make([]float32, K*rowStride)
		ctxAware := make([]float32, K*rowStride)
		matmul(be, &m.w.PerLayerModelProj, h, ctxAware, K)
		cScale := float32(1.0 / math.Sqrt(float64(hidden)))
		for i := range ctxAware {
			ctxAware[i] *= cScale
		}
		tScale := float32(math.Sqrt(float64(pleDim)))
		inv2 := float32(1.0 / math.Sqrt2)
		for row := range K {
			tid := ids[row]
			if _, in := imageSpanRow(spans, row); in {
				tid = g4.PadTokenID
			}
			rowPL := perLayer[row*rowStride : (row+1)*rowStride]
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
	}

	// Cross-layer KV sharing is a pure function of arch, identical to the sequential path (forward_gemma4.go), so this file stays
	// structurally matched to the reference it must agree with on text-only input (the regression gate).
	firstShared := arch.NumLayers - g4.SharedKVLayers
	lastSliding, lastGlobal := -1, -1
	for i := range firstShared {
		if arch.isGlobalLayer(i) {
			lastGlobal = i
		} else {
			lastSliding = i
		}
	}
	kvSrc := func(l int) int {
		if l < firstShared {
			return l
		}
		if arch.isGlobalLayer(l) {
			return lastGlobal
		}
		return lastSliding
	}

	// Attention scratch: allocated once per call, reused across every (row,layer)
	// gemma4Attend call — sized to the widest per-layer head_dim and the widest
	// any row's attended range can ever be (startPos+K, the full batch).
	maxHd := max(g4.GlobalHeadDim, arch.HeadDim)
	nkMax := startPos + K
	g4sc := g4attnScratch{
		kh:     make([]float32, nkMax*maxHd),
		vt:     make([]float32, nkMax*maxHd),
		qstack: make([]float32, nH*maxHd),
		scores: make([]float32, nH*nkMax),
		cstack: make([]float32, nH*maxHd),
	}

	normd := make([]float32, K*hidden)
	var hq w4a8Act // the K-row normd block, quantized once for q/k/v and once for gate/up (R-13); reused per layer
	for l := range arch.NumLayers {
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		hd := arch.headDimAt(l)
		nKV := arch.kvHeadsAt(l)
		ffn := arch.ffnAt(l)
		invFreq := localInv
		if global {
			invFreq = globalInv
		}

		// --- attention sub-block (sandwich) ---
		copy(normd, h)
		for row := range K {
			normalize(arch, normd[row*hidden:(row+1)*hidden], lw.PreAttnNorm, nil, hidden)
		}

		q := make([]float32, K*nH*hd)
		hq.prepare(be, &lw.QProj, normd, K)
		matmulPre(be, &lw.QProj, &hq, normd, q, K)
		rmsNorm(q, lw.QNorm, K*nH, hd, arch.NormEps, arch.RMSAddOne)
		for row := range K {
			pos := startPos + row
			applyRoPE(q[row*nH*hd:(row+1)*nH*hd], nH, hd, pos, invFreq, 1.0)
		}

		if l < firstShared { // owns its KV
			k := make([]float32, K*nKV*hd)
			matmulPre(be, &lw.KProj, &hq, normd, k, K)
			v := make([]float32, K*nKV*hd)
			if lw.VFromK { // attention_k_eq_v: V is v_norm(k_proj output)
				copy(v, k)
			} else {
				matmulPre(be, &lw.VProj, &hq, normd, v, K)
			}
			rmsNorm(k, lw.KNorm, K*nKV, hd, arch.NormEps, arch.RMSAddOne) // K: k_norm + RoPE
			rmsNormNoWeight(v, K*nKV, hd, arch.NormEps)                   // V: scale-less v_norm, no RoPE
			// All K rows must be appended before ANY row's attention is read this
			// layer (below) — a block-interior row must see a LATER block row's
			// just-computed K/V, which a per-row append-then-attend loop could
			// never provide.
			for row := range K {
				pos := startPos + row
				kRow := k[row*nKV*hd : (row+1)*nKV*hd]
				applyRoPE(kRow, nKV, hd, pos, invFreq, 1.0)
				cache.Append(l, kRow, v[row*nKV*hd:(row+1)*nKV*hd])
			}
		}

		src := kvSrc(l)
		keys, vals := cache.Keys(src), cache.Vals(src)
		ctx := make([]float32, K*nH*hd)
		for row := range K {
			pos := startPos + row
			lo, hi := gemma4AttendRange(cache, pos, global, spans)
			gemma4Attend(q[row*nH*hd:(row+1)*nH*hd], ctx[row*nH*hd:(row+1)*nH*hd], keys, vals, nH, nKV, hd, lo, hi+1, arch.AttnScale, &g4sc)
		}

		attnOut := make([]float32, K*hidden)
		matmul(be, &lw.OProj, ctx, attnOut, K)
		for row := range K {
			normalize(arch, attnOut[row*hidden:(row+1)*hidden], lw.PostAttnNorm, nil, hidden)
		}
		for i := range h {
			h[i] += attnOut[i]
		}

		// --- FFN sub-block ---
		if lw.gemma4moe != nil {
			// Not batchable across rows by construction (the router picks different experts per token). gemma4MoEFFN is position-independent
			// (it takes no position argument), so a per-row loop over the unchanged function is the correct and only shape.
			for row := range K {
				hRow := h[row*hidden : (row+1)*hidden]
				copy(hRow, gemma4MoEFFN(be, arch, hRow, lw.gemma4moe, m.pager))
			}
		} else {
			copy(normd, h)
			for row := range K {
				normalize(arch, normd[row*hidden:(row+1)*hidden], lw.PreMLPNorm, nil, hidden)
			}
			gate := make([]float32, K*ffn)
			up := make([]float32, K*ffn)
			hq.prepare(be, &lw.GateProj, normd, K)
			matmulPre(be, &lw.GateProj, &hq, normd, gate, K)
			matmulPre(be, &lw.UpProj, &hq, normd, up, K)
			parallelElementwise(len(gate), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					gate[i] = geluTanh(gate[i]) * up[i]
				}
			})
			mlpOut := make([]float32, K*hidden)
			matmul(be, &lw.DownProj, gate, mlpOut, K)
			for row := range K {
				normalize(arch, mlpOut[row*hidden:(row+1)*hidden], lw.PostMLPNorm, nil, hidden)
			}
			for i := range h {
				h[i] += mlpOut[i]
			}

			// --- PLE branch: gate→gelu→×per-layer-embedding→proj→norm→+residual ---
			if pleDim > 0 {
				px := make([]float32, K*pleDim)
				matmul(be, &lw.PLEGate, h, px, K)
				for row := range K {
					plSeg := perLayer[row*rowStride+l*pleDim : row*rowStride+(l+1)*pleDim]
					pxRow := px[row*pleDim : (row+1)*pleDim]
					for i := range pxRow {
						pxRow[i] = geluTanh(pxRow[i]) * plSeg[i]
					}
				}
				pout := make([]float32, K*hidden)
				matmul(be, &lw.PLEProj, px, pout, K)
				for row := range K {
					normalize(arch, pout[row*hidden:(row+1)*hidden], lw.PostPLENorm, nil, hidden)
				}
				for i := range h {
					h[i] += pout[i]
				}
			}

			// --- per-layer output scalar (row-agnostic: one uniform multiply) ---
			if lw.LayerScalar != 0 {
				for i := range h {
					h[i] *= lw.LayerScalar
				}
			}
		}
		// cache.captureResidual (the draft-head hidden-state seam) is deliberately not wired here —
		// a decode-time concern; post-prefill decode resumes via the unchanged
		// per-token runLayersGemma4 path, which already carries it.
	}
	cache.advanceTo(startPos + K) // gemma4's manualPos=true: Append never auto-advances pos
	return h[(K-1)*hidden : K*hidden], nil
}
