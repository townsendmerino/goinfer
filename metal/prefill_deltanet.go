//go:build darwin

package metal

import "github.com/townsendmerino/goinfer/internal/mem"

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
		bt:     d.NewBufferLenOn(mem.LineScratch, M*dp.nv), at: d.NewBufferLenOn(mem.LineScratch, M*dp.nv),
		conv: d.NewBufferLenOn(mem.LineScratch, M*dp.convDim), headP: d.NewBufferLenOn(mem.LineScratch, M*dp.nv*2),
		qn: d.NewBufferLenOn(mem.LineScratch, M*dp.keyDim), kn: d.NewBufferLenOn(mem.LineScratch, M*dp.keyDim),
		core: d.NewBufferLenOn(mem.LineScratch, M*dp.valueDim), gated: d.NewBufferLenOn(mem.LineScratch, M*dp.valueDim),
		uM: NewBufferU32On(mem.LineScratch, d, uint32(M)), uConvDim: NewBufferU32On(mem.LineScratch, d, uint32(dp.convDim)), uValueDim: NewBufferU32On(mem.LineScratch, d, uint32(dp.valueDim)),
	}
	prefillScratchSeen.Add(int64(4 * M * (4*dp.nv + dp.convDim + 2*dp.keyDim + 2*dp.valueDim)))
	s.bufs = []Buffer{s.mixedF, s.zF, s.gatedH, s.bt, s.at, s.conv, s.headP, s.qn, s.kn, s.core, s.gated, s.uM, s.uConvDim, s.uValueDim}
	return s
}

// dnSeqOldForTest makes the prefill take the device-memory scan kernels (delta_rule_seq, delta_conv_seq): the old arm of
// KS-D3's A/B.
var dnSeqOldForTest bool

// dnSeqPipelines returns the prefill mixer's scan kernels and the rule's threadgroup size: the on-chip forms
// (delta_rule_seq_tg, 32 threads, its state rows in threadgroup memory; delta_conv_seq_reg, the window in registers) where
// the geometry fits them (hk at most 128, K at most 5), else the device-memory forms. Both forms run the same operations in
// the same order, so the output is the same bits (TestDeltaNetSeqKernels_matchDecodeBitwise).
func (r *resident) dnSeqPipelines() (rule, conv Pipeline, ruleTG int) {
	dp := r.dnet
	rule, conv, ruleTG = r.pDnRuleSeq, r.pDnConvSeq, 128
	if dnSeqOldForTest {
		return
	}
	if dp.hk <= 128 {
		rule, ruleTG = r.pDnRuleSeqTG, 32
	}
	if dp.convK <= 5 {
		conv = r.pDnConvSeqReg
	}
	return
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
	rule, conv, ruleTG := r.dnSeqPipelines()
	e.Dispatch(conv, dp.convDim, 256, s.mixedF, D.convW, D.win, s.conv, r.uDnConvDim, r.uDnK, s.uM)
	e.Dispatch(r.pDnGatesRows, M*dp.nv, 64, s.bt, s.at, D.dtBias, D.negExpA, s.headP, r.uDnNv, s.uM)
	e.Dispatch(r.pDnNormRows, M*dp.nk*tgReduceAttn, tgReduceAttn, s.conv, s.qn, s.kn, r.uDnNk, r.uDnHk, r.uDnKeyDim, r.uDnQScale,
		r.uDnConvDim, s.uM)
	e.Dispatch(rule, dp.valueDim, ruleTG, s.qn, s.kn, s.conv, s.headP, D.state, s.core,
		r.uDnNv, r.uDnHk, r.uDnHv, r.uDnRep, r.uDnVBase, r.uDnKeyDim, r.uDnConvDim, s.uM)
	e.Dispatch(r.pDnGNormRows, M*dp.nv*tgReduceAttn, tgReduceAttn, s.core, s.zF, D.normW, s.gated, r.uDnNv, r.uDnHv, r.uEps, s.uM)
	e.Dispatch(r.pF32ToF16, M*dp.valueDim, 256, s.gated, s.gatedH)
	gemm(e, Mpad, r.H, s.gatedH, D.outW, D.outS, xF, uMpad, r.uH, s.uValueDim, dummyBias, m2)
}
