package cuda

import (
	"github.com/townsendmerino/aikit/gpu"
)

// dnPrefillBufs are the M-row buffers a batched prefill of a Gated-DeltaNet model uses
// (docs/tasks/task-cuda-deltanet-prefill-2026-09.md). Rows are packed at each projection's own width.
type dnPrefillBufs struct {
	mix, bt, at, z Buffer // the four input projections, [M × convDim | nv | nv | valueDim]
	conv           Buffer // the conv output [q|k|v] per row, [M × convDim]
	qn, kn         Buffer // l2-normalized q and k per row, [M × keyDim]
	headP          Buffer // (beta, gt) per row and value head, [M × nv × 2]
	core           Buffer // the recurrence output per row, [M × valueDim]
	g              Buffer // the gated recurrence output, [M × valueDim]
	gq, gsc        Buffer // g quantized for out_proj (int8 rows + one scale per row)
	qg, gate       Buffer // softmax layers with attn_output_gate: the double-width q projection and its gate half
}

// prefillDeltaNetRows is deltaNetMixer over the M rows of a batched prefill, folding each row's mixer output into
// that row of the residual xB. The projections (norm+quant, in_proj_qkv / b / a / z, and out_proj) run once over all
// M rows. The recurrence — the conv ring, the gates, the q/k norm, the delta rule and the gated norm — runs through
// the row-batched twins of decode's five kernels (deltanet.cu), which do each row's arithmetic exactly as decode does
// and walk the stateful ones (the conv ring, the delta rule) in row order, so the layer's conv window and matrix state
// advance exactly as M decode steps would advance them.
func (r *cudaResident) prefillDeltaNetRows(Ly *cudaLayer, xB, aqB, aScB Buffer, M int, b dnPrefillBufs) error {
	dp := r.dnet
	nullBias := ArgNull()
	t := r.profTic()
	if e := r.bRmsB(xB, Ly.preNorm, r.hidden, aqB, aScB, M); e != nil { // decode: r.rms
		return e
	}
	for _, p := range []struct {
		w   cudaWQ
		dst Buffer
	}{{Ly.dnQKV, b.mix}, {Ly.dnB, b.bt}, {Ly.dnA, b.at}, {Ly.dnZ, b.z}} {
		if e := r.bGemvB(p.w, aqB, aScB, nullBias, p.dst, M, 0); e != nil {
			return e
		}
	}
	r.profToc(gemvCat, t)
	t = r.profTic()
	// The recurrence, five launches for all M rows. The conv ring and the delta rule walk the rows in order inside
	// each thread; the gates and both norms are row-independent. Bit-identical to five launches per row.
	if e := r.launch(r.dnConvRows, g1cfg(dp.convDim, 256),
		Arg(b.mix), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(b.conv),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK)), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	if e := r.launch(r.dnGatesRows, g1cfg(M*dp.nv, 64),
		Arg(b.bt), Arg(b.at), Arg(Ly.dnDtBias), Arg(Ly.dnNegExpA), Arg(b.headP),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	if e := r.launch(r.dnNormRows, LaunchConfig{GridX: uint32(dp.nk), GridY: uint32(M), GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4},
		Arg(b.conv), Arg(b.qn), Arg(b.kn),
		gpu.ArgValue(int32(dp.nk)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(dp.qScale)); e != nil {
		return e
	}
	if dp.hk == 128 && dp.hv == 128 {
		// Every released model's geometry: the state row stays in registers across the rows (the measured cost of the generic
		// scan: docs/tasks/task-cuda-deltanet-prefill-2026-09.md).
		if e := r.launch(r.dnRuleRows128, LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1},
			Arg(b.qn), Arg(b.kn), Arg(b.conv), Arg(b.headP), Arg(Ly.dnState), Arg(b.core),
			gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2*dp.keyDim)),
			gpu.ArgValue(int32(dp.keyDim)), gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
	} else if e := r.launch(r.dnRuleRows, g1cfg(dp.valueDim, 128),
		Arg(b.qn), Arg(b.kn), Arg(b.conv), Arg(b.headP), Arg(Ly.dnState), Arg(b.core),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2*dp.keyDim)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	if e := r.launch(r.dnGNormRows, LaunchConfig{GridX: uint32(dp.nv), GridY: uint32(M), GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		Arg(b.core), Arg(b.z), Arg(Ly.dnNormW), Arg(b.g),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hv)), gpu.ArgValue(r.eps)); e != nil {
		return e
	}
	r.profToc(recCat, t)
	t = r.profTic()
	// Quantize the M gated rows (decode: fQ per row), then out_proj accumulating into the residual.
	if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(b.g), gpu.ArgValue(int32(dp.valueDim)), Arg(b.gq), Arg(b.gsc), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	err := r.bGemvB(Ly.dnOut, b.gq, b.gsc, nullBias, xB, M, 1)
	r.profToc(gemvCat, t)
	return err
}
