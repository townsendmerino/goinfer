//go:build darwin

package metal

// D-B01 (docs/audit-metal-2026-09-30.md, docs/tasks/task-metal-audit-2026-10.md): a Gated-DeltaNet layer in the batched
// prefill pass. The projections (qkv, z, out) run as prefill GEMMs over all M rows, and in_proj_b / in_proj_a as a small
// rows kernel, on f16 activations like every other batched layer. The mixer between them runs as one dispatch per
// stage over the M rows, the recurrent stages (the conv window, the delta rule's state) as a token loop inside the
// kernel. Those row kernels are the decode kernels with their bodies copied verbatim, so given the same inputs they
// produce decode's bits, window and state included (TestDeltaNetSeqKernels_matchDecodeBitwise); what leaves the exact
// lane is only what every batched layer changes, the f16 projections. 13 dispatches per DeltaNet layer for the whole
// prompt, where the sequential path runs 12 per token.
//
// The window and state continue from whatever the resident holds, as M decode steps would: the decoder resets them for
// a fresh sequence (resetDeltaNet) and leaves them for a continuation.

// dnetPrefillOn admits Gated-DeltaNet models to the batched prefill pass. On by default; tests turn it off for the
// sequential arm (docs/tasks/task-metal-audit-2026-10.md, D-B01; docs/code-notes/metal.md#dnetPrefillOn).
var dnetPrefillOn = true

// prefillDelta is one PrefillLast call's DeltaNet scratch, shared by every DeltaNet layer (each layer's own window and
// state live on residLayer.delta).
type prefillDelta struct {
	mixedF, zF, gatedH      Buffer // half: [Mpad][convDim], [Mpad][valueDim], [Mpad][valueDim]
	bt, at, conv, headP     Buffer // f32: [M][nv] x2, [M][convDim], [M][nv][2]
	qn, kn, core, gated     Buffer // f32: [M][keyDim] x2, [M][valueDim] x2
	uM, uConvDim, uValueDim Buffer
	bufs                    []Buffer
}

func (r *resident) newPrefillDelta(M, Mpad int) *prefillDelta {
	d, dp := r.d, r.dnet
	// mixedF, zF and gatedH rely on Metal's zero-fill of a new buffer for their pad rows M..Mpad (aikit gpu.NewBufferLen*
	// contract: Metal only; CUDA's is uninitialized); see prefillScratchU16.
	s := &prefillDelta{
		mixedF: prefillScratchU16(d, Mpad*dp.convDim),
		zF:     prefillScratchU16(d, Mpad*dp.valueDim),
		gatedH: prefillScratchU16(d, Mpad*dp.valueDim),
		bt:     d.NewBufferLen(M * dp.nv), at: d.NewBufferLen(M * dp.nv),
		conv: d.NewBufferLen(M * dp.convDim), headP: d.NewBufferLen(M * dp.nv * 2),
		qn: d.NewBufferLen(M * dp.keyDim), kn: d.NewBufferLen(M * dp.keyDim),
		core: d.NewBufferLen(M * dp.valueDim), gated: d.NewBufferLen(M * dp.valueDim),
		uM: NewBufferU32(d, uint32(M)), uConvDim: NewBufferU32(d, uint32(dp.convDim)), uValueDim: NewBufferU32(d, uint32(dp.valueDim)),
	}
	s.bufs = []Buffer{s.mixedF, s.zF, s.gatedH, s.bt, s.at, s.conv, s.headP, s.qn, s.kn, s.core, s.gated, s.uM, s.uConvDim, s.uValueDim}
	return s
}

// encodePrefillDeltaMixer is encodeDeltaNetMixer over the pass's M rows: pre-norm, the projections, the mixer, and
// out_proj accumulated into the f16 residual xF. normF is the pass's norm scratch; gemm is PrefillLast's tiled GEMM.
func (r *resident) encodePrefillDeltaMixer(e *Encoder, L *residLayer, xF, normF Buffer, s *prefillDelta, M, Mpad int,
	gemm func(e *Encoder, rows, N int, bufs ...Buffer), uMpad, m0, m2, dummyBias Buffer) {
	pf, dp, D := r.pf, r.dnet, L.delta
	e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, r.uH, r.uEps, r.uAddOne)
	gemm(e, Mpad, dp.convDim, normF, D.qkvW, D.qkvS, s.mixedF, uMpad, s.uConvDim, r.uH, dummyBias, m0)
	gemm(e, Mpad, dp.valueDim, normF, D.zW, D.zS, s.zF, uMpad, s.uValueDim, r.uH, dummyBias, m0)
	e.Dispatch(r.pDnProjW8Rows, M*dp.nv*32, 32, normF, D.bW, D.bS, s.bt, r.uH, r.uDnNv, s.uM)
	e.Dispatch(r.pDnProjW8Rows, M*dp.nv*32, 32, normF, D.aW, D.aS, s.at, r.uH, r.uDnNv, s.uM)
	e.Dispatch(r.pDnConvSeq, dp.convDim, 256, s.mixedF, D.convW, D.win, s.conv, r.uDnConvDim, r.uDnK, s.uM)
	e.Dispatch(r.pDnGatesRows, M*dp.nv, 64, s.bt, s.at, D.dtBias, D.negExpA, s.headP, r.uDnNv, s.uM)
	e.Dispatch(r.pDnNormRows, M*dp.nk*tgReduceAttn, tgReduceAttn, s.conv, s.qn, s.kn, r.uDnNk, r.uDnHk, r.uDnKeyDim, r.uDnQScale,
		r.uDnConvDim, s.uM)
	e.Dispatch(r.pDnRuleSeq, dp.valueDim, 128, s.qn, s.kn, s.conv, s.headP, D.state, s.core,
		r.uDnNv, r.uDnHk, r.uDnHv, r.uDnRep, r.uDnVBase, r.uDnKeyDim, r.uDnConvDim, s.uM)
	e.Dispatch(r.pDnGNormRows, M*dp.nv*tgReduceAttn, tgReduceAttn, s.core, s.zF, D.normW, s.gated, r.uDnNv, r.uDnHv, r.uEps, s.uM)
	e.Dispatch(r.pF32ToF16, M*dp.valueDim, 256, s.gated, s.gatedH)
	gemm(e, Mpad, r.H, s.gatedH, D.outW, D.outS, xF, uMpad, r.uH, s.uValueDim, dummyBias, m2)
}
