//go:build darwin

package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"math"
	"strconv"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// Gated-DeltaNet mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8) — the resident-runner wiring on top
// of deltanet_kernels.go's gated recurrence chain. Ported from cuda/resident.go's deltaNetMixer;
// read that file's comments first, this mirrors its structure as closely as Metal's per-layer,
// no-graph-capture encodeLayer/encodeAttention shape allows.
//
// dnetParams carries the model-level geometry (uniform across the linear layers) — the same
// fields as cuda's dnetParams, computed once from decoder.Model.Qwen35ResidentParams().
type dnetParams struct {
	convK      int
	hk, hv     int
	nk, nv     int
	rep        int
	keyDim     int
	valueDim   int
	convDim    int
	stateElems int
	qScale     float32
}

// deltaNetLayer is one Gated-DeltaNet mixer layer's weights + persistent state. win/state are
// PERSISTENT (survive across tokens; zeroed at build and by Reset — see resetDeltaNet), unlike
// the KV cache buffers every other layer type uses, which are simply overwritten per position.
type deltaNetLayer struct {
	qkvW, qkvS Buffer // in_proj_qkv, model quant (int4/int8/f32->int4 via mk)
	zW, zS     Buffer // in_proj_z
	outW, outS Buffer // out_proj
	bW, bS     Buffer // in_proj_b (int8 — see the file-level note below)
	aW, aS     Buffer // in_proj_a
	convW      Buffer // [convDim*K]
	dtBias     Buffer // [nv]
	negExpA    Buffer // [nv] PRECOMPUTED -exp(A_log)
	normW      Buffer // [hv] gated-RMSNorm weight
	win        Buffer // PERSISTENT: [(K-1)*convDim] causal-conv ring
	state      Buffer // PERSISTENT: [nv*hv*hk], transposed [hv,hk] layout — see deltanet_kernels.go
}

// buildDeltaNet compiles the mixer's pipelines and derives its model-level geometry. Returns
// (nil, nil) for a dense model (no qwen35 family) — the same "ok=false -> not this family" shape
// buildMoE uses, so buildResident's caller can no-op it uniformly.
func buildDeltaNet(d *Device, m *decoder.Model, pipe func(string) Pipeline) (*dnetParams, error) {
	convKernel, keyHeadDim, valueHeadDim, numKeyHeads, numValueHeads, _, ok := m.Qwen35ResidentParams()
	if !ok {
		return nil, nil
	}
	if recurrentInt8Load(m) && !dnetInt8DeclineOffForTest {
		// Metal's native int8 path covers no recurrent family (w8Eligible), so int8 projections would be re-quantized to
		// int4 without notice: decline, and the CPU runs the model at the int8 it was loaded with (G44, as buildMamba).
		return nil, fmt.Errorf("metal: a Gated-DeltaNet model runs resident at int4 only; its int8 projections would be re-quantized to int4 here, so it runs on the CPU")
	}
	keyDim, valueDim := numKeyHeads*keyHeadDim, numValueHeads*valueHeadDim
	dnet := &dnetParams{
		convK: convKernel, hk: keyHeadDim, hv: valueHeadDim, nk: numKeyHeads, nv: numValueHeads,
		rep: numValueHeads / numKeyHeads, keyDim: keyDim, valueDim: valueDim,
		convDim: 2*keyDim + valueDim, stateElems: numValueHeads * valueHeadDim * keyHeadDim,
		qScale: float32(1 / math.Sqrt(float64(keyHeadDim))),
	}
	_ = pipe // pipelines are compiled by buildResident's shared `pipe` closure at the call site (see model.go)
	return dnet, nil
}

// recurrentInt8Load reports whether m was loaded with int8 projections (int8, int8int8, or int4mix's attention), which a
// recurrent family's build would re-quantize to int4: Metal's native int8 path covers no recurrent family (w8Eligible).
func recurrentInt8Load(m *decoder.Model) bool {
	switch m.Quant() {
	case "int8", "int8int8", "int4mix":
		return true
	}
	return false
}

// dnetInt8DeclineOffForTest removes buildDeltaNet's int8 decline: G44's planted defect.
var dnetInt8DeclineOffForTest = false

// buildDeltaNetLayer uploads one Gated-DeltaNet layer's weights and allocates its persistent
// state, zeroed. inB/inA are kept f32 on the CPU (deltaNetWeights' own comment: they feed the
// write/decay gates, where the recurrence is most precision-sensitive) but quantized to int8
// here, matching cuda/backend.go's build — "the one place this port is knowingly coarser than the
// reference," recorded there, not re-litigated here. They are TINY relative to the model (a few
// dozen rows), so this rides Metal's existing int8 GEMV path (pGemvW8) rather than adding a
// dedicated f32 GEMV kernel for two small projections.
func buildDeltaNetLayer(d *Device, m *decoder.Model, l int, dnet *dnetParams, mk func(*linalg.WeightMat) (Buffer, Buffer)) (*deltaNetLayer, error) {
	inQKV, inZ, outProj, inB, inA, convW, dtBias, negExpA, normW := m.Qwen35DeltaWeights(l)
	// Name the missing tensor (as cuda/backend.go's identical check does): an empty slice becomes a 0-byte device upload that
	// fails as "invalid length" with no hint which of the four small tensors was missing.
	// History: docs/code-notes/metal.md#buildDeltaNetLayer.
	for _, chk := range []struct {
		name string
		n    int
	}{{"conv_w", len(convW)}, {"dt_bias", len(dtBias)}, {"neg_exp_a", len(negExpA)}, {"norm_w", len(normW)}} {
		if chk.n == 0 {
			return nil, errDeltaEmpty(l, chk.name)
		}
	}
	L := &deltaNetLayer{}
	L.qkvW, L.qkvS = mk(inQKV)
	L.zW, L.zS = mk(inZ)
	L.outW, L.outS = mk(outProj)
	bWM := linalg.QuantizeInt8(inB, dnet.nv, len(inB)/dnet.nv, false)
	aWM := linalg.QuantizeInt8(inA, dnet.nv, len(inA)/dnet.nv, false)
	var e1, e2 error
	if L.bW, L.bS, e1 = int8BufOn(mem.LineDense, d, &bWM); e1 != nil {
		return nil, e1
	}
	if L.aW, L.aS, e2 = int8BufOn(mem.LineDense, d, &aWM); e2 != nil {
		return nil, e2
	}
	L.convW = NewBufferFloatsOn(mem.LineDense, d, convW)
	L.dtBias = NewBufferFloatsOn(mem.LineDense, d, dtBias)
	L.negExpA = NewBufferFloatsOn(mem.LineDense, d, negExpA)
	L.normW = NewBufferFloatsOn(mem.LineDense, d, normW)
	// PERSISTENT, zeroed — matching newDeltaState (decoder/deltanet.go) and cuda's up(make(...)).
	L.win = NewBufferFloatsOn(mem.LineState, d, make([]float32, (dnet.convK-1)*dnet.convDim))
	L.state = NewBufferFloatsOn(mem.LineState, d, make([]float32, dnet.stateElems))
	return L, nil
}

func errDeltaEmpty(l int, name string) error {
	return &deltaEmptyError{layer: l, name: name}
}

type deltaEmptyError struct {
	layer int
	name  string
}

func (e *deltaEmptyError) Error() string {
	return "metal: layer " + strconv.Itoa(e.layer) + ": DeltaNet " + e.name +
		" is empty — the loader did not populate it for this container"
}

// encodeDeltaNetMixer runs one Gated-DeltaNet layer's sequence mixer and folds its output into
// the residual — the recurrent replacement for the whole attention sub-block (norm through
// o-proj). Mirrors cuda/resident.go's deltaNetMixer dispatch-for-dispatch.
func (r *resident) encodeDeltaNetMixer(e *Encoder, L *residLayer) {
	dp := r.dnet
	D := L.delta
	r.encodeNorm(e, r.x, L.preNorm, L.preNormBias, r.aq, r.aSc)
	r.gemvExt(e, gemvExtKCoal, dp.convDim, r.H, D.qkvW, D.qkvS, r.aq, r.aSc, r.dnMixed, r.uH)
	e.Dispatch(r.pGemvW8, dp.nv*32, 32, r.aq, r.aSc, D.bW, D.bS, r.dnBt, r.uH)
	e.Dispatch(r.pGemvW8, dp.nv*32, 32, r.aq, r.aSc, D.aW, D.aS, r.dnAt, r.uH)
	r.gemvExt(e, gemvExtKCoal, dp.valueDim, r.H, D.zW, D.zS, r.aq, r.aSc, r.dnZOut, r.uH)
	e.Dispatch(r.pDnConv, dp.convDim, 256, r.dnMixed, D.convW, D.win, r.dnConvOut, r.uDnConvDim, r.uDnK)
	e.Dispatch(r.pDnGates, dp.nv, 64, r.dnBt, r.dnAt, D.dtBias, D.negExpA, r.dnHeadP, r.uDnNv)
	e.Dispatch(r.pDnNorm, dp.nk*tgReduceAttn, tgReduceAttn, r.dnConvOut, r.dnQn, r.dnKn, r.uDnNk, r.uDnHk, r.uDnKeyDim, r.uDnQScale)
	e.Dispatch(r.pDnRule, dp.valueDim, 128, r.dnQn, r.dnKn, r.dnConvOut, r.dnHeadP, D.state, r.dnCore,
		r.uDnNv, r.uDnHk, r.uDnHv, r.uDnRep, r.uDnVBase)
	e.Dispatch(r.pDnGNorm, dp.nv*tgReduceAttn, tgReduceAttn, r.dnCore, r.dnZOut, D.normW, r.dnGated, r.uDnNv, r.uDnHv, r.uEps)
	// Quantize the gated output, then out_proj straight into the residual (accum via _resid),
	// exactly as the ordinary attention path's o-proj does. Dispatch width is r.H*32 — one
	// (packed) simdgroup per OUTPUT row (hidden, same output width the o-proj GEMV has), not per
	// input row; the shared-memory size and the trailing uniform are the INPUT width (dp.valueDim),
	// matching how encodeAttention's o-proj call sizes both against qDim (its own input width).
	e.Dispatch(r.pQv, 256, 256, r.dnGated, r.dnGq, r.dnGSc, r.uDnValueDim)
	e.DispatchTG(r.pSAResid, r.H*32, 256, dp.valueDim*2, D.outW, D.outS, r.dnGq, r.dnGSc, r.x, r.uDnValueDim)
}

// resetDeltaNet zeroes every DeltaNet layer's causal-conv ring and recurrent matrix state. Unlike KV positions, which the
// next sequence overwrites, this state accumulates, so a fresh Generate on the same resident must re-zero it or it keeps
// decaying stale state from the prior sequence. No-op when r.dnet is nil (every other family).
func (r *resident) resetDeltaNet() {
	if r.dnet == nil {
		return
	}
	// Cleared in place, the same prefix the copy from two zeroed slices covered, without building the slices.
	nWin, nSt := (r.dnet.convK-1)*r.dnet.convDim, r.dnet.stateElems
	for i := range r.layers {
		L := &r.layers[i]
		if L.delta == nil {
			continue
		}
		win, st := L.delta.win.Floats(), L.delta.state.Floats()
		clear(win[:min(nWin, len(win))])
		clear(st[:min(nSt, len(st))])
	}
}
