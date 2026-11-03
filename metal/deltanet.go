//go:build darwin

package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/program"
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
	// betaScale multiplies the write gate's sigmoid: 2 for NegEigval (the gate in [0, 2)), else 1. oNormEps is the gated
	// output norm's own epsilon, 0 when it is the model's (program.DeltaNetOpParams).
	betaScale float32
	oNormEps  float32
}

// deltaNetOpParams is the parameters every OpDeltaNet layer of m's program shares, or an error if two layers disagree.
func deltaNetOpParams(m *decoder.Model) (program.DeltaNetOpParams, error) {
	var p program.DeltaNetOpParams
	seen := false
	for l, ops := range m.Program().Layers {
		for i := range ops {
			if ops[i].Kind != program.OpDeltaNet {
				continue
			}
			if seen && ops[i].Delta != p {
				return p, fmt.Errorf("metal: layer %d's DeltaNet parameters %+v differ from an earlier layer's %+v; the block holds one set", l, ops[i].Delta, p)
			}
			p, seen = ops[i].Delta, true
		}
	}
	return p, nil
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
	// slotWin and slotState are each KV slot's own win and state, slot 0's first; win and state are the bound slot's
	// (useKVSlot). Empty on a one-slot build.
	slotWin, slotState []Buffer
}

// allocRecurrentSlots gives every recurrent layer, Gated-DeltaNet and Mamba-2, its own conv window and state for each of n KV
// slots, zeroed. Slot 0's are the pair the layer was built with. The decoder asks for more than one slot only where
// program.Metal.RecurrentSlots lists every recurrent kind the model has, and metalSlotBytes prices each further slot's pairs.
func (r *resident) allocRecurrentSlots(n int) {
	r.allocDeltaNetSlots(n)
	r.allocMambaSlots(n)
}

// allocDeltaNetSlots is allocRecurrentSlots for the Gated-DeltaNet layers.
func (r *resident) allocDeltaNetSlots(n int) {
	if r.dnet == nil || n < 2 {
		return
	}
	for i := range r.layers {
		D := r.layers[i].delta
		if D == nil {
			continue
		}
		D.slotWin, D.slotState = []Buffer{D.win}, []Buffer{D.state}
		for range n - 1 {
			D.slotWin = append(D.slotWin, NewBufferFloatsOn(mem.LineState, r.d, make([]float32, D.win.Len())))
			D.slotState = append(D.slotState, NewBufferFloatsOn(mem.LineState, r.d, make([]float32, D.state.Len())))
		}
	}
}

// resetDeltaNetSlot zeroes every Gated-DeltaNet layer's conv window and state for KV slot s: the slot's own pair, or the
// layer's one pair on a one-slot build. The executor must be stopped first, as for resetDeltaNet.
func (r *resident) resetDeltaNetSlot(s int) {
	if r.dnet == nil {
		return
	}
	for i := range r.layers {
		D := r.layers[i].delta
		if D == nil {
			continue
		}
		win, state := D.win, D.state
		if s < len(D.slotWin) {
			win, state = D.slotWin[s], D.slotState[s]
		}
		clear(win.Floats())
		clear(state.Floats())
	}
}

// metalRecurrentSlotBytes is one KV slot's recurrent state on m, from the list its layers allocate from (decoder.Model.StateSpecs).
func metalRecurrentSlotBytes(m *decoder.Model) int64 {
	_, nL, _, _, _, _, _ := m.Dims()
	var n int64
	for l := range nL {
		for _, s := range m.StateSpecs(l) {
			n += 4 * int64(s.Floats)
		}
	}
	return n
}

// stateBuffersOn allocates layer l's state from its specs, zeroed: its window and its matrix state.
func stateBuffersOn(d *Device, m *decoder.Model, l int) (win, state Buffer, err error) {
	specs := m.StateSpecs(l)
	if len(specs) != 2 || specs[0].Name != "window" || specs[1].Name != "state" {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: layer %d's recurrent state is %v, and the block holds a window and a state", l, specs)
	}
	return NewBufferFloatsOn(mem.LineState, d, make([]float32, specs[0].Floats)), NewBufferFloatsOn(mem.LineState, d, make([]float32, specs[1].Floats)), nil
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
		qScale: float32(1 / math.Sqrt(float64(keyHeadDim))), betaScale: 1,
	}
	op, err := deltaNetOpParams(m)
	if err != nil {
		return nil, err
	}
	if op.NegEigval && !dnetNegEigvalOffForTest {
		dnet.betaScale = 2
	}
	if op.ONormEps != 0 && op.ONormEps != op.Eps && !dnetONormEpsOffForTest {
		dnet.oNormEps = float32(op.ONormEps)
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

// dnetNegEigvalOffForTest and dnetONormEpsOffForTest drop the DeltaNet's NegEigval and ONormEps at the build: OM-1's planted
// defects (docs/tasks/task-olmo-hybrid-metal-webgpu-2026-11.md).
var dnetNegEigvalOffForTest, dnetONormEpsOffForTest = false, false

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
	var err error
	if L.win, L.state, err = stateBuffersOn(d, m, l); err != nil {
		return nil, err
	}
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
	r.encodeDeltaNetMixerOn(e, L, r.x, L.delta.win, L.delta.state)
}

// encodeDeltaNetMixerOn is encodeDeltaNetMixer on residual x, against the conv window win and state state: the multi-sequence
// step passes each row's residual and its own slot's pair.
func (r *resident) encodeDeltaNetMixerOn(e *Encoder, L *residLayer, x, win, state Buffer) {
	dp := r.dnet
	D := L.delta
	r.encodeNorm(e, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
	r.gemvExt(e, gemvExtKCoal, dp.convDim, r.H, D.qkvW, D.qkvS, r.aq, r.aSc, r.dnMixed, r.uH)
	e.Dispatch(r.pGemvW8, dp.nv*32, 32, r.aq, r.aSc, D.bW, D.bS, r.dnBt, r.uH)
	e.Dispatch(r.pGemvW8, dp.nv*32, 32, r.aq, r.aSc, D.aW, D.aS, r.dnAt, r.uH)
	r.gemvExt(e, gemvExtKCoal, dp.valueDim, r.H, D.zW, D.zS, r.aq, r.aSc, r.dnZOut, r.uH)
	e.Dispatch(r.pDnConv, dp.convDim, 256, r.dnMixed, D.convW, win, r.dnConvOut, r.uDnConvDim, r.uDnK)
	e.Dispatch(r.pDnGates, dp.nv, 64, r.dnBt, r.dnAt, D.dtBias, D.negExpA, r.dnHeadP, r.uDnNv, r.uDnBetaScale)
	e.Dispatch(r.pDnNorm, dp.nk*tgReduceAttn, tgReduceAttn, r.dnConvOut, r.dnQn, r.dnKn, r.uDnNk, r.uDnHk, r.uDnKeyDim, r.uDnQScale)
	e.Dispatch(r.pDnRule, dp.valueDim, 128, r.dnQn, r.dnKn, r.dnConvOut, r.dnHeadP, state, r.dnCore,
		r.uDnNv, r.uDnHk, r.uDnHv, r.uDnRep, r.uDnVBase)
	e.Dispatch(r.pDnGNorm, dp.nv*tgReduceAttn, tgReduceAttn, r.dnCore, r.dnZOut, D.normW, r.dnGated, r.uDnNv, r.uDnHv, r.uDnONormEps)
	// Quantize the gated output, then out_proj straight into the residual (accum via _resid),
	// exactly as the ordinary attention path's o-proj does. Dispatch width is r.H*32 — one
	// (packed) simdgroup per OUTPUT row (hidden, same output width the o-proj GEMV has), not per
	// input row; the shared-memory size and the trailing uniform are the INPUT width (dp.valueDim),
	// matching how encodeAttention's o-proj call sizes both against qDim (its own input width).
	e.Dispatch(r.pQv, 256, 256, r.dnGated, r.dnGq, r.dnGSc, r.uDnValueDim)
	e.DispatchTG(r.pSAResid, r.H*32, 256, dp.valueDim*2, D.outW, D.outS, r.dnGq, r.dnGSc, x, r.uDnValueDim)
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
