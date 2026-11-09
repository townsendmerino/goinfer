package metal

import (
	"os"

	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// The attention step's segments, in the order a layer runs them: the input norm, the q/k/v projection, the QK-norm, Gemma 4's
// v_norm, the rotation, the KV store and attention, the output gate, and the output tail (the context's conversion, the output
// projection, any post-attention norm and the residual add). The step's block (resident.encodeAttentionResidualWith) runs them by
// the family's flags; the decode walk (resident.encodeMixerWalk) runs them by the step's operations.

// attnEnc is one attention step's encoding state: the layer, its geometry and lane, the residual and the position uniforms.
type attnEnc struct {
	l                         int
	L                         *residLayer
	g                         *attnGeom
	x, uPos, uNKeys, uQTemp   Buffer
	rp                        Buffer // the rotation's position
	nHhd, qkvRows, kOff, vOff int    // kOff and vOff are the byte offsets of k and v within the fused qkv buffer
	f16Lane                   bool
}

func (r *resident) attnEncOf(l int, x, uPos, uNKeys, uQTemp, rp Buffer) attnEnc {
	L := &r.layers[l]
	g := L.geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	if L.kvShared {
		qkvRows = nHhd // S1.4: Q only; K/V come from layer L.kvSrc's cache, which r.kc[l]/r.vc[l] alias
	}
	return attnEnc{l: l, L: L, g: g, x: x, uPos: uPos, uNKeys: uNKeys, uQTemp: uQTemp, rp: rp, nHhd: nHhd, qkvRows: qkvRows,
		kOff: nHhd * 4, vOff: (nHhd + g.kvDim) * 4, f16Lane: r.canUseF16Lane(l)}
}

// attnInputNorm is the step's input norm, of form form: a copy (post-only placement: quantize the raw residual, quant_vec, the
// symmetric int8 quantizer the context's conversion also uses), an RMSNorm or a LayerNorm, on the layer's lane.
func (r *resident) attnInputNorm(e *Encoder, a *attnEnc, form program.NormForm) {
	L, x := a.L, a.x
	if r.q4kLane {
		e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, x, L.preNorm, r.axF32, r.uH, r.uEps, r.uAddOne)
	} else if form == program.NormCopy {
		e.Dispatch(r.pQv, 256, 256, x, r.aq, r.aSc, r.uH)
	} else if a.f16Lane {
		// R1: f16 activation, no quantization — bypasses encodeNorm (which always produces the
		// int8 aq/aSc pair) since preNorm here has no bias on the plain dense path canUseF16Lane
		// admits (a biased pre-norm would need layernorm_quant's shape, not rmsnorm's).
		e.Dispatch(r.pRmsF16, tgReduceNorm, tgReduceNorm, x, L.preNorm, r.axF16, r.uH, r.uEps, r.uAddOne)
	} else if form == program.NormLayer {
		e.Dispatch(r.pLayerNorm, tgReduceNorm, tgReduceNorm, x, L.preNorm, L.preNormBias, r.aq, r.aSc, r.uH, r.uEps, r.uLNHasBias)
	} else {
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	}
}

// attnGatedQKV is a gated layer's projection. This family's softmax layer: q_proj is DOUBLE WIDTH ([query ‖ gate] per head,
// interleaved) and bias-free, so it dispatches separately via the no-bias SA kernel into dnQg, then delta_qsplit extracts Q into
// r.qkv's normal Q slot (offset 0, same as every other family) and the gate into dnAGate. K‖V is a 2-way fused (still bias-free)
// projection straight into r.qkv's K/V slots (kOff) — same buffer, same layout, just two dispatches building it instead of one.
// Both MUST land before qk_norm runs.
func (r *resident) attnGatedQKV(e *Encoder, a *attnEnc) {
	L, g := a.L, a.g
	e.DispatchTG(r.pSA, 2*a.nHhd*32, 256, r.H*2, L.dnQw, L.dnQs, r.aq, r.aSc, r.dnQg, r.uH)
	e.Dispatch(r.pDnQSplit, a.nHhd, 256, r.dnQg, r.qkv, r.dnAGate, g.uNHhd, g.uHd)
	e.DispatchTG(r.pSA, 2*g.kvDim*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv.At(a.kOff), r.uH)
}

// attnProjQKV is the fused q/k/v projection with its combined bias (zeros where absent), on the layer's lane.
func (r *resident) attnProjQKV(e *Encoder, a *attnEnc) {
	L := a.L
	if r.q4kLane {
		r.q4kGemv(e, L.qkvSegs, false, L.qkvBias, r.axF32, r.qkv, r.uH) // the combined bias, zeros where absent
	} else if a.f16Lane {
		e.DispatchTG(r.pSAf16Bias, a.qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.axF16, L.qkvBias, r.qkv, r.uH)
	} else {
		p, n := saRowsPick(r.pSABias, r.pSABiasRows, a.qkvRows, r.gemvRows.qkv)
		e.DispatchTG(p, n, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		if r.loraLayers != nil {
			// G3: compute-time LoRA — added on top of the base q/k/v projection, into the SAME
			// r.qkv slots it just wrote, before anything downstream (qk_norm/RoPE) reads them.
			// Not reached for the qGate projection (see lora.go's file comment).
			LA := &r.loraLayers[a.l]
			r.applyResidentLoRA(e, LA.q, r.aq, r.aSc, r.qkv)
			if !L.kvShared {
				r.applyResidentLoRA(e, LA.k, r.aq, r.aSc, r.qkv.At(a.kOff))
				r.applyResidentLoRA(e, LA.v, r.aq, r.aSc, r.qkv.At(a.vOff))
			}
		}
	}
}

// attnQKNorm is the per-head Q/K RMSNorm before the rotation, or with whole one statistic over all heads (Olmo 3, Olmo Hybrid):
// the same kernel with the grid collapsed to one Q block and one K block, via the constant-1 and whole-width buffers built in
// buildResident. qk_norm's block h<nH reduces [base, +hd) with base = head*hd, so nH=1, hd=nH_orig*hd spans the whole contiguous
// Q vector for h=0 and the whole K vector for h=nH=1. g.uNHhd (the K offset within the fused qkv buffer) does not depend on the
// reinterpreted reduction width. Correct only for MHA (nH==nKV); buildResident declines otherwise.
func (r *resident) attnQKNorm(e *Encoder, a *attnEnc, whole bool) {
	L, g := a.L, a.g
	qkNH, qkNKV, qkHD := r.uNH, g.uNKV, g.uHd
	if whole {
		qkNH, qkNKV, qkHD = r.uQKWholeOne, r.uQKWholeOne, r.uQKWholeHD
	}
	grid := (r.nH + g.nKV) * tgReduceAttn
	if whole {
		grid = 2 * tgReduceAttn
	}
	if L.kvShared {
		grid = r.nH * tgReduceAttn // Q heads only: the K slot holds nothing on a shared layer
	}
	e.Dispatch(r.pQKNorm, grid, tgReduceAttn, r.qkv, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, g.uNHhd, r.uEps, r.uAddOne)
}

// attnVNormNeeded reports whether layer a runs Gemma 4's scale-less v_norm: every K/V-owning Gemma 4 layer, and a K=V layer.
func (r *resident) attnVNormNeeded(a *attnEnc) bool {
	return !a.L.kvShared && (a.g.kEqV || (r.g4VNorm && !g4DropVNormForTest))
}

// attnVNorm is Gemma 4's scale-less v_norm on the V slot, every K/V-owning layer (S1.0). For K=V (Gemma 4 globals) the V slot holds
// the RAW k_proj output ([Q|K|K] fusion); otherwise the v_proj output. It is qk_norm over the V slot (qkv.At(vOff)) with nH=0 so
// every head takes the K branch at base 0+head*hd, a UNIT weight, and addOne=0 → x·rms·1. It runs AFTER qk_norm (which touched the
// K slot, not V) and BEFORE RoPE (which never touches V), so V = v_norm(raw k), un-rotated — exactly the CPU path (copy(v,k)
// pre-k_norm; then rmsNormNoWeight(v)). See TestVNorm_scaleless.
func (r *resident) attnVNorm(e *Encoder, a *attnEnc) {
	g := a.g
	e.Dispatch(r.pQKNorm, g.nKV*tgReduceAttn, tgReduceAttn, r.qkv.At(a.vOff), r.vNormUnit, r.vNormUnit, r.uZero, g.uNKV, g.uHd, r.uZero, r.uEps, r.uZero)
}

// attnRope is the rotation of Q and K, with the query's temperature scale. One merged dispatch for both Q and K (rope2, kernels.go)
// instead of two: gid<qTotal addresses Q at offset 0, gid>=qTotal addresses K at offset g.uNHhd (the fused qkv buffer's kOff, in
// elements) — V (at vOff) is untouched either way.
func (r *resident) attnRope(e *Encoder, a *attnEnc) {
	L, g := a.L, a.g
	ropeN := r.nH*g.half + g.nKV*g.half
	if L.kvShared {
		ropeN = r.nH * g.half // Q only: rope2's gid < qTotal branch
	}
	e.Dispatch(r.pRope2, ropeN, 64, r.qkv, L.invf, g.uHd, a.rp, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, a.uQTemp)
}

// attnCore stores the position's K and V and attends: the int8 cache's kernels, the split-KV attention at its depths, or the
// exact kernel.
func (r *resident) attnCore(e *Encoder, a *attnEnc) {
	l, L, g := a.l, a.L, a.g
	// S1.4: a shared layer must never write its source's cache. The seam is G2's defect (5): the store not skipped,
	// so the shared layer writes its (unprojected) K/V slots over its source's row.
	storeKV := !L.kvShared || g4KeepSharedKVStoreForTest
	if r.kvI8 {
		if storeKV {
			e.Dispatch(r.pKvI8, g.nKV, 1, r.qkv.At(a.kOff), r.qkv.At(a.vOff), r.kc[l].At(0), r.vc[l].At(0), r.ks[l].At(0), r.vs[l].At(0), g.uNKV, g.uHd, a.uPos, r.kt)
		}
		e.Dispatch(r.pAttnI8, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l].At(0), r.vc[l].At(0), r.ks[l].At(0), r.vs[l].At(0), r.ctx, r.uNH, g.uNKV, g.uHd, a.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink, r.kt)
		return
	}
	if storeKV {
		e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(a.kOff), r.qkv.At(a.vOff), r.kc[l].At(0), r.vc[l].At(0), g.uKvDim, a.uPos, r.kt)
	}
	if r.canUseAttnFA(l) {
		// nSplit here sizes the dispatch grid only, which is fixed once a command buffer is encoded. It is not written
		// into r.uAttnFANSplit here: setPos does that once per decode step, before this buffer commits (see
		// uAttnFAG/uAttnFANSplit's field comment for the race a per-layer SetU32 was). The two agree by construction:
		// both are attnFASplitFor of the key count this buffer runs at (planNKeys here, the job's own position in
		// setPos), and the executor re-encodes a pre-encoded buffer whose plan does not match the arriving job
		// (execLoop). Reading the key count at encode time instead would give the previous job's: one token late at
		// the floor, and the previous request's depth on a new request's first step, with a grid sized for the old
		// depth and a uniform for the new.
		nSplit := r.attnFASplitFor(r.planNKeys(), g.nKV)
		if os.Getenv("GOINFER_ATTNFA_DEBUG") == "1" {
			logs.Infof("[ATTNFA] l=%d nKeys=%d nSplit=%d nKV=%d G=%d hd=%d partialLen=%d",
				l, r.planNKeys(), nSplit, g.nKV, r.nH/g.nKV, g.hd, r.attnFAPartial.Len())
		}
		shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
		e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes, r.qkv, r.kc[l].At(0), r.vc[l].At(0), r.attnFAPartial,
			g.uNKV, r.uAttnFAG, a.uNKeys, r.uScale, L.uWindow, r.fanSplitBuf(), r.kt)
		e.Dispatch(r.pAttnFACombine, r.nH*g.hd, g.hd, r.attnFAPartial, r.ctx, r.uAttnFAG, g.uHd, r.fanSplitBuf())
		return
	}
	e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l].At(0), r.vc[l].At(0), r.ctx, r.uNH, g.uNKV, g.uHd, a.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink, r.kt)
}

// attnOutGate is a gated layer's ctx *= sigmoid(gate), before o-proj — matches the CPU qwen35Attention.
func (r *resident) attnOutGate(e *Encoder, a *attnEnc) {
	e.Dispatch(r.pDnAttnGate, a.nHhd, 256, r.ctx, r.dnAGate, a.g.uNHhd)
}

// attnOutForm is how the output tail ends: into the residual with the output projection's dispatch (attnOutFused, with its bias
// where attnOutFusedBias), or into the oO scratch and added after (attnOutScratch), with a post-attention norm between where
// attnOutScratchNorm.
type attnOutForm uint8

const (
	attnOutFused attnOutForm = iota
	attnOutFusedBias
	attnOutScratch
	attnOutScratchNorm
)

// attnOut is the output tail: the context's conversion for the lane, the output projection, and its form's end.
func (r *resident) attnOut(e *Encoder, a *attnEnc, form attnOutForm) {
	l, L, g, x := a.l, a.L, a.g, a.x
	switch {
	case r.q4kLane: // o-proj reads r.ctx (f32) directly: no conversion, no quantization.
	case a.f16Lane:
		e.Dispatch(r.pF32ToF16, a.nHhd, 256, r.ctx, r.cxF16)
	default:
		e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
	}
	switch {
	case form == attnOutScratch || form == attnOutScratchNorm:
		// Gemma sandwich / Olmo 3 postOnly: the sublayer OUTPUT is normed BEFORE the residual
		// add, which the fused _resid epilogue can't express — project into the (otherwise
		// dead) oO scratch, norm it, then add. Cohere parallelBlock shares the "project into oO,
		// defer the add" shape but has NO post-attn norm at all (L.postAttnNorm is never built for
		// it — see the build loop), so it skips the norm dispatch entirely.
		e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
		if r.loraLayers != nil {
			// G3: added into the RAW o-proj output, BEFORE the sandwich/postOnly norm below —
			// matching applyLoRA's CPU order (delta added to `out`, caller norms afterward).
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, r.oO)
		}
		if form == attnOutScratchNorm {
			e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
		}
		e.Dispatch(r.pRes, r.H, 256, x, r.oO)
	case form == attnOutFusedBias: // GPT-2/gpt-oss: o-proj carries an additive bias, fused with the residual add
		e.DispatchTG(r.pSABiasResid, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, x, L.oBias, g.uNHhd)
		if r.loraLayers != nil {
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, x)
		}
	case r.q4kLane:
		r.q4kGemv(e, L.oSegs, true, Buffer{}, r.ctx, x, g.uNHhd) // o-proj + residual
	case a.f16Lane:
		e.DispatchTG(r.pSAf16Resid, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cxF16, x, g.uNHhd) // o-proj + residual
	default:
		p, n := saRowsPick(r.pSAResid, r.pSAResidRows, r.H, r.gemvRows.o)
		e.DispatchTG(p, n, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, x, g.uNHhd) // o-proj + residual
		if r.loraLayers != nil {
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, x)
		}
	}
}
