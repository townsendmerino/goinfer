//go:build darwin

package metal

// KS-D8 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D8 design and gates"): a Mamba-2 layer in the batched prefill pass, in
// D-B01's shape. in_proj and out_proj run as int8 prefill GEMMs over all M rows (gemm_w8f16_*, on the int8 codes and per-row
// scales the layer already holds), on f16 activations like every other batched layer; the mixer between them runs as one
// dispatch per stage over the M rows, the recurrent stages (the conv window, the SSM state) as a token loop inside the kernel.
// Those row kernels are the decode kernels' statements per row, so given the same inputs they produce decode's bits, window
// and state included; what leaves the decode lane is only what every batched layer changes, the f16 projections. 7
// dispatches per Mamba layer for the whole prompt, where the sequential path runs 9 per token.
//
// The window and state continue from whatever the resident holds, as M decode steps would: the decoder resets them for a
// fresh sequence (resetMamba) and leaves them for a continuation.

// mambaPrefillOn admits Mamba-2 hybrids to the batched prefill pass. On by default; tests turn it off for the sequential arm.
var mambaPrefillOn = true

// prefillMamba is one PrefillLast call's Mamba-2 scratch, shared by every Mamba layer.
type prefillMamba struct {
	projF, gatedH  Buffer // half: [Mpad][projDim], [Mpad][dInner]
	conv, y, gated Buffer // f32: [M][convDim], [M][dInner] x2
	uM, uProjDim   Buffer
	bufs           []Buffer
}

func (r *resident) newPrefillMamba(M, Mpad int) *prefillMamba {
	d, p := r.d, r.mamba
	// projF's and gatedH's pad rows M..Mpad rely on Metal's zero-fill of a new buffer, as prefillDelta's do.
	s := &prefillMamba{
		projF: prefillScratchU16(d, Mpad*p.projDim), gatedH: prefillScratchU16(d, Mpad*p.dInner),
		conv: d.NewBufferLen(M * p.convDim), y: d.NewBufferLen(M * p.dInner), gated: d.NewBufferLen(M * p.dInner),
		uM: NewBufferU32(d, uint32(M)), uProjDim: NewBufferU32(d, uint32(p.projDim)),
	}
	prefillScratchSeen.Add(int64(4 * M * (p.convDim + 2*p.dInner)))
	s.bufs = []Buffer{s.projF, s.gatedH, s.conv, s.y, s.gated, s.uM, s.uProjDim}
	return s
}

// prefillMambaBytes is newPrefillMamba's allocation for M rows (Mpad = M rounded up to 8), for prefillScratchBytes.
func (r *resident) prefillMambaBytes(m, mp int64) int64 {
	p := r.mamba
	return 2*mp*int64(p.projDim+p.dInner) + 4*m*int64(p.convDim+2*p.dInner)
}

// encodePrefillMambaMixer is encodeMambaMixer over the pass's M rows: pre-norm, in_proj, the conv → SSM → gated-norm chain,
// and out_proj accumulated into the f16 residual xF. normF is the pass's norm scratch; gemmW8 is PrefillLast's tiled GEMM on
// the int8 tiles.
func (r *resident) encodePrefillMambaMixer(e *Encoder, L *residLayer, xF, normF Buffer, s *prefillMamba, M, Mpad int,
	gemmW8 func(e *Encoder, rows, N int, bufs ...Buffer), uMpad, m0, m2, dummyBias Buffer) {
	pf, p, mb := r.pf, r.mamba, L.mamba
	e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, r.uH, r.uEps, r.uAddOne)
	gemmW8(e, Mpad, p.projDim, normF, mb.inW, mb.inS, s.projF, uMpad, s.uProjDim, r.uH, dummyBias, m0)
	e.Dispatch(r.pMbConvSeq, p.convDim, 64, s.projF, mb.convW, mb.convB, mb.win, s.conv, r.uMbConvDim, r.uMbK, r.uMbDInner,
		s.uProjDim, s.uM)
	e.Dispatch(r.pMbSSMSeq, p.nHeads*p.headDim, 64, s.conv, s.projF, mb.headP, mb.ssm, s.y,
		r.uMbNH, r.uMbHD, r.uMbDS, r.uMbRep, r.uMbGSize, r.uMbDInner, r.uMbDtBase, r.uMbConvDim, s.uProjDim, s.uM)
	e.Dispatch(r.pMbGNormRows, M*p.normGroups*64, 64, s.y, s.projF, mb.normW, s.gated, r.uMbGroupSize, r.uMbEps, r.uMbZero,
		r.uMbNormGroups, r.uMbDInner, s.uProjDim)
	e.Dispatch(r.pF32ToF16, M*p.dInner, 256, s.gated, s.gatedH)
	gemmW8(e, Mpad, r.H, s.gatedH, mb.outW, mb.outS, xF, uMpad, r.uH, r.uMbDInner, dummyBias, m2)
}
