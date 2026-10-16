//go:build darwin

package metal

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// The Mamba-2 mixer (Nemotron-H) on the resident runner, on top of mamba_kernels_msl.go's conv → SSM → gated-norm chain:
// the recurrent replacement for a layer's attention sub-block, as encodeDeltaNetMixer is for the Gated-DeltaNet hybrids.
// in_proj and out_proj run W8A8 (int8 weights from the CPU's f32, the pGemvW8 path DeltaNet's two small projections take),
// as WebGPU's does (docs/ssm-int8-quality.md); the conv window and the SSM state are persistent, zeroed at build and by
// resetRecurrent.

// mambaSrcPlantForTest is a planted defect in the library source every resident compiles, {from, to}, set by tests in this
// package only (OP2b's gate M: the Mamba-2 kernels broken in the Metal build while the CPU keeps the right math).
var mambaSrcPlantForTest [2]string

// mambaResetSkipForTest makes resetMamba leave the state alone: gate S's planted defect, set by tests in this package only.
var mambaResetSkipForTest bool

// mambaParams is the mixer's model-level geometry, uniform across the Mamba layers.
type mambaParams struct {
	nHeads, headDim, dState, nGroups, dConv, normGroups int
	dInner, gSize, convDim, projDim, repeat             int
}

// mambaLayer is one Mamba-2 layer's weights and persistent state.
type mambaLayer struct {
	inW, inS, outW, outS Buffer // in_proj [projDim, hidden], out_proj [hidden, dInner]: int8 with per-row scales
	convW, convB         Buffer // [convDim*K], [convDim]
	headP                Buffer // [nHeads*3]: -exp(aLog) (f64 on the host), dtBias, D
	normW                Buffer // [dInner]
	win                  Buffer // PERSISTENT: [(K-1)*convDim] conv window, oldest first
	ssm                  Buffer // PERSISTENT: [nHeads*headDim*dState]
}

// buildMamba returns the model's Mamba-2 geometry, or nil for a model with no Mamba layer. Granite-4.0-H declines: its
// embedding and residual multipliers are not folded on Metal.
func buildMamba(m *decoder.Model) (*mambaParams, error) {
	if _, _, _, _, _, _, _, _, _, ok := m.GraniteResidentParams(); ok {
		return nil, fmt.Errorf("metal: Granite-4.0-H's embedding and residual multipliers are not folded on Metal; it runs on the CPU")
	}
	nh, hd, ds, ng, k, normG, _, ok := m.NemotronResidentParams()
	if !ok {
		return nil, nil
	}
	if !m.ResidentProjsInt4() {
		// Metal's native int8 path covers no recurrent family (w8Eligible), so int8 projections would be re-quantized to
		// int4 without notice: decline, and the CPU runs the model at the int8 it was loaded with
		// (docs/tasks/task-forward-architecture-2026-10.md §4.7, gate M's amendment).
		return nil, fmt.Errorf("metal: Nemotron-H runs resident at int4 only; its int8 projections would be re-quantized to int4 here, so it runs on the CPU")
	}
	if normG < 1 || ng < 1 || nh%ng != 0 || (nh*hd)%normG != 0 || k < 2 || k > 8 || (nh*hd)%8 != 0 {
		return nil, fmt.Errorf("metal: Mamba-2 geometry out of the kernels' range (nHeads %d, headDim %d, nGroups %d, normGroups %d, dConv %d; "+
			"dInner must be a multiple of 8, out_proj's int8 GEMV reads K four bytes at a time)", nh, hd, ng, normG, k)
	}
	p := &mambaParams{nHeads: nh, headDim: hd, dState: ds, nGroups: ng, dConv: k, normGroups: normG}
	p.dInner, p.gSize = nh*hd, ng*ds
	p.convDim, p.repeat = p.dInner+2*p.gSize, nh/ng
	p.projDim = 2*p.dInner + 2*p.gSize + nh
	return p, nil
}

// buildMambaLayer uploads layer l's Mamba-2 weights and allocates its state, zeroed.
func buildMambaLayer(d *Device, m *decoder.Model, l, hidden int, p *mambaParams) (*mambaLayer, error) {
	inProj, convW, convB, aLog, dW, dtBias, normW, outProj := m.GraniteMambaWeights(l)
	// The projections as the load holds them: int8 already on a quantized load, else quantized here from the f32, the same
	// codes either way.
	var inWM, outWM linalg.WeightMat
	if in, out := m.MambaProjInt8(l); in != nil {
		inWM, outWM = *in, *out
	} else {
		if len(inProj) != p.projDim*hidden || len(outProj) != hidden*p.dInner {
			return nil, fmt.Errorf("metal: layer %d: Mamba-2 projections have %d and %d values, want %d and %d", l, len(inProj), len(outProj), p.projDim*hidden, hidden*p.dInner)
		}
		inWM, outWM = linalg.QuantizeInt8(inProj, p.projDim, hidden, false), linalg.QuantizeInt8(outProj, hidden, p.dInner, false)
	}
	if inWM.Rows() != p.projDim || inWM.Cols() != hidden || outWM.Rows() != hidden || outWM.Cols() != p.dInner {
		return nil, fmt.Errorf("metal: layer %d: Mamba-2 projections are %d×%d and %d×%d, want %d×%d and %d×%d", l,
			inWM.Rows(), inWM.Cols(), outWM.Rows(), outWM.Cols(), p.projDim, hidden, hidden, p.dInner)
	}
	for _, c := range []struct {
		name    string
		got, of int
	}{{"conv_w", len(convW), p.convDim * p.dConv}, {"conv_b", len(convB), p.convDim},
		{"a_log", len(aLog), p.nHeads}, {"d", len(dW), p.nHeads}, {"dt_bias", len(dtBias), p.nHeads}, {"norm_w", len(normW), p.dInner}} {
		if c.got != c.of {
			return nil, fmt.Errorf("metal: layer %d: Mamba-2 %s has %d values, want %d", l, c.name, c.got, c.of)
		}
	}
	L := &mambaLayer{}
	var err error
	if L.inW, L.inS, err = int8Buf(d, &inWM); err != nil {
		return nil, err
	}
	if L.outW, L.outS, err = int8Buf(d, &outWM); err != nil {
		return nil, err
	}
	headP := make([]float32, p.nHeads*3)
	for h := range p.nHeads {
		headP[h*3] = float32(-math.Exp(float64(aLog[h])))
		headP[h*3+1], headP[h*3+2] = dtBias[h], dW[h]
	}
	L.convW, L.convB = NewBufferFloats(d, convW), NewBufferFloats(d, convB)
	L.headP, L.normW = NewBufferFloats(d, headP), NewBufferFloats(d, normW)
	L.win = NewBufferFloats(d, make([]float32, (p.dConv-1)*p.convDim))
	L.ssm = NewBufferFloats(d, make([]float32, p.nHeads*p.headDim*p.dState))
	return L, nil
}

// buildMambaScratch compiles the mixer's pipelines and allocates its per-token scratch and uniforms.
func (r *resident) buildMambaScratch(d *Device, pipe func(string) Pipeline, eps float32) {
	p := r.mamba
	r.pMbConv, r.pMbSSM, r.pMbGNorm = pipe("mamba_conv"), pipe("mamba_ssm"), pipe("mamba_gnorm")
	r.mbProj, r.mbConv = d.NewBufferLen(p.projDim), d.NewBufferLen(p.convDim)
	r.mbY, r.mbGated = d.NewBufferLen(p.dInner), d.NewBufferLen(p.dInner)
	r.mbGq, r.mbGSc = d.NewBufferBytes(p.dInner), d.NewBufferLen(1)
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	r.uMbConvDim, r.uMbK, r.uMbDInner = u(p.convDim), u(p.dConv), u(p.dInner)
	r.uMbNH, r.uMbHD, r.uMbDS, r.uMbRep, r.uMbGSize = u(p.nHeads), u(p.headDim), u(p.dState), u(p.repeat), u(p.gSize)
	r.uMbDtBase, r.uMbGroupSize, r.uMbZero = u(p.dInner+p.convDim), u(p.dInner/p.normGroups), u(0)
	r.uMbEps = NewBufferFloats(d, []float32{eps})
}

// encodeMambaMixer runs one Mamba-2 layer's mixer and folds its output into the residual x: the norm, in_proj, the conv →
// SSM → gated-norm chain over the in_proj output [z | xBC | dt], then out_proj and the residual add.
func (r *resident) encodeMambaMixer(e *Encoder, L *residLayer, x Buffer) {
	p, M := r.mamba, L.mamba
	r.encodeNorm(e, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
	e.Dispatch(r.pGemvW8, p.projDim*32, 32, r.aq, r.aSc, M.inW, M.inS, r.mbProj, r.uH)
	e.Dispatch(r.pMbConv, p.convDim, 64, r.mbProj, M.convW, M.convB, M.win, r.mbConv, r.uMbConvDim, r.uMbK, r.uMbDInner)
	e.Dispatch(r.pMbSSM, p.nHeads*p.headDim, 64, r.mbConv, r.mbProj, M.headP, M.ssm, r.mbY,
		r.uMbNH, r.uMbHD, r.uMbDS, r.uMbRep, r.uMbGSize, r.uMbDInner, r.uMbDtBase)
	e.Dispatch(r.pMbGNorm, p.normGroups*64, 64, r.mbY, r.mbProj, M.normW, r.mbGated, r.uMbGroupSize, r.uMbEps, r.uMbZero)
	e.Dispatch(r.pQv, 256, 256, r.mbGated, r.mbGq, r.mbGSc, r.uMbDInner)
	e.Dispatch(r.pGemvW8, r.H*32, 32, r.mbGq, r.mbGSc, M.outW, M.outS, r.dO, r.uMbDInner)
	e.Dispatch(r.pRes, r.H, 256, x, r.dO)
}

// resetMamba zeroes every Mamba layer's conv window and SSM state, which a fresh sequence must not inherit. No-op without
// Mamba layers.
func (r *resident) resetMamba() {
	if r.mamba == nil || mambaResetSkipForTest {
		return
	}
	for i := range r.layers {
		if M := r.layers[i].mamba; M != nil {
			clear(M.win.Floats())
			clear(M.ssm.Floats())
		}
	}
}

// orZeros is v, or n zeros when v is empty: a bias buffer for a family without that bias (an empty upload is an error).
func orZeros(v []float32, n int) []float32 {
	if len(v) == 0 {
		return make([]float32, n)
	}
	return v
}

// resetRecurrent zeroes every recurrent mixer's state (DeltaNet and Mamba-2), as a fresh sequence needs.
func (r *resident) resetRecurrent() {
	r.resetDeltaNet()
	r.resetMamba()
}
