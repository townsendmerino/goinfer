//go:build cuda

package cuda

import (
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// The Mamba-2 mixer (Nemotron-H) and the non-gated squared-ReLU MLP on the resident runner (docs/tasks/task-forward-architecture-2026-10.md §4.7, OP2b on
// CUDA, step 2b). A Nemotron-H block carries ONE operation (decoder.Model.ResidentLayerOps): the Mamba-2 mixer, attention, or the MLP, each behind
// its own pre-norm and added to the residual. The mixer is mamba.cu's conv → SSM → gated-norm chain around two W8A8 projections; the MLP is the up
// GEMV, nongated_act.cu's relu2_quant, and the down GEMV. Nothing here is reachable end to end until the recurrent guard sites read Mamba too
// (step 2c) and the backend declares the features.

// mambaConvShiftForTest shifts mamba_conv's input read by that many elements: OP2b gate M's planted defect, a conv that sees the wrong channels while the CPU
// keeps the right math. Set by tests only.
var mambaConvShiftForTest int

// mambaResetSkipForTest makes resetState leave the Mamba state alone: gate S's planted defect, set by tests only.
var mambaResetSkipForTest bool

// mambaInt8DeclineOffForTest removes buildMambaParams's int4-only decline: the planted defect of the int8 test, set by tests only.
var mambaInt8DeclineOffForTest bool

// mambaParams is the mixer's model-level geometry, uniform across the Mamba layers.
type mambaParams struct {
	nHeads, headDim, dState, nGroups, dConv, normGroups int
	dInner, gSize, convDim, projDim, repeat             int
}

// buildMambaParams returns the model's Mamba-2 geometry, or nil for a model with no Mamba layer. Granite-4.0-H and an int8 load decline by name:
// Granite's embedding and residual multipliers are not folded here, and int8 projections would be re-quantized to int4 without notice.
func buildMambaParams(m *decoder.Model) (*mambaParams, error) {
	if _, _, _, _, _, _, _, _, _, ok := m.GraniteResidentParams(); ok {
		return nil, fmt.Errorf("Granite-4.0-H's embedding and residual multipliers are not folded on CUDA; it runs on the CPU")
	}
	nh, hd, ds, ng, k, normG, _, ok := m.NemotronResidentParams()
	if !ok {
		return nil, nil
	}
	if (!m.ResidentProjsInt4() || recurrentInt8Load(m)) && !mambaInt8DeclineOffForTest {
		return nil, fmt.Errorf("Nemotron-H runs resident at int4 only; its int8 projections would be re-quantized to int4 here, so it runs on the CPU")
	}
	if normG < 1 || ng < 1 || nh%ng != 0 || (nh*hd)%normG != 0 || k < 2 || k > 8 || (nh*hd)%8 != 0 {
		return nil, fmt.Errorf("Mamba-2 geometry out of the kernels' range (nHeads %d, headDim %d, nGroups %d, normGroups %d, dConv %d; "+
			"dInner must be a multiple of 8, out_proj's int8 GEMV reads K four bytes at a time)", nh, hd, ng, normG, k)
	}
	p := &mambaParams{nHeads: nh, headDim: hd, dState: ds, nGroups: ng, dConv: k, normGroups: normG}
	p.dInner, p.gSize = nh*hd, ng*ds
	p.convDim, p.repeat = p.dInner+2*p.gSize, nh/ng
	p.projDim = 2*p.dInner + 2*p.gSize + nh
	return p, nil
}

// recurrentInt8Load reports whether m was loaded with int8 projections (int8, int8int8, or int4mix's attention), which a Nemotron-H build would run at
// int4 against the CPU's int8.
func recurrentInt8Load(m *decoder.Model) bool {
	switch m.Quant() {
	case "int8", "int8int8", "int4mix":
		return true
	}
	return false
}

// hostMamba is one Mamba-2 layer's packed weights and small tables, before upload.
type hostMamba struct {
	in, out                    hostW
	convW, convB, headP, normW []float32
}

// packMambaLayer packs layer l's Mamba-2 weights: in_proj [projDim, hidden] and out_proj [hidden, dInner] as per-row int8 (the codes the load holds, or
// quantized here from the f32, the same codes either way), and headP = {-exp(aLog) in f64, dtBias, D} per head.
func packMambaLayer(m *decoder.Model, l, hidden int, p *mambaParams) (hostMamba, error) {
	inProj, convW, convB, aLog, dW, dtBias, normW, outProj := m.GraniteMambaWeights(l)
	var inWM, outWM linalg.WeightMat
	if in, out := m.MambaProjInt8(l); in != nil {
		inWM, outWM = *in, *out
	} else {
		if len(inProj) != p.projDim*hidden || len(outProj) != hidden*p.dInner {
			return hostMamba{}, fmt.Errorf("layer %d: Mamba-2 projections have %d and %d values, want %d and %d", l, len(inProj), len(outProj), p.projDim*hidden, hidden*p.dInner)
		}
		inWM, outWM = linalg.WrapF32(inProj, p.projDim, hidden), linalg.WrapF32(outProj, hidden, p.dInner)
	}
	if inWM.Rows() != p.projDim || inWM.Cols() != hidden || outWM.Rows() != hidden || outWM.Cols() != p.dInner {
		return hostMamba{}, fmt.Errorf("layer %d: Mamba-2 projections are %d×%d and %d×%d, want %d×%d and %d×%d", l,
			inWM.Rows(), inWM.Cols(), outWM.Rows(), outWM.Cols(), p.projDim, hidden, hidden, p.dInner)
	}
	for _, c := range []struct {
		name    string
		got, of int
	}{{"conv_w", len(convW), p.convDim * p.dConv}, {"conv_b", len(convB), p.convDim},
		{"a_log", len(aLog), p.nHeads}, {"d", len(dW), p.nHeads}, {"dt_bias", len(dtBias), p.nHeads}, {"norm_w", len(normW), p.dInner}} {
		if c.got != c.of {
			return hostMamba{}, fmt.Errorf("layer %d: Mamba-2 %s has %d values, want %d", l, c.name, c.got, c.of)
		}
	}
	var hm hostMamba
	var err error
	if hm.in, err = packWeight(&inWM); err != nil {
		return hostMamba{}, fmt.Errorf("layer %d in_proj: %w", l, err)
	}
	if hm.out, err = packWeight(&outWM); err != nil {
		return hostMamba{}, fmt.Errorf("layer %d out_proj: %w", l, err)
	}
	if hm.in.kind != "int8" || hm.out.kind != "int8" {
		return hostMamba{}, fmt.Errorf("layer %d: Mamba-2 projections pack as %q/%q, want int8 (gemv_w8a8)", l, hm.in.kind, hm.out.kind)
	}
	hm.headP = make([]float32, p.nHeads*3)
	for h := range p.nHeads {
		hm.headP[h*3] = float32(-math.Exp(float64(aLog[h])))
		hm.headP[h*3+1], hm.headP[h*3+2] = dtBias[h], dW[h]
	}
	hm.convW, hm.convB, hm.normW = convW, convB, normW
	return hm, nil
}

// noKV reports whether the layer holds no attention K/V: the recurrent mixers and a feed-forward-only block.
func (L *cudaLayer) noKV() bool {
	return L.isDeltaNet || L.mixer == decoder.MixerMamba || L.mixer == decoder.MixerNone
}

// loadMambaKernels compiles mamba.ptx and nongated_act.ptx and binds the pipelines, and allocates the mixer's per-token scratch. needNonGated is true when
// any layer is a feed-forward-only block.
func (r *cudaResident) loadMambaKernels(p *mambaParams, needNonGated bool) error {
	mod, err := r.dev.CompileLibrary(mambaPTX)
	if err != nil {
		return fmt.Errorf("cuda: JIT mamba.ptx: %w", err)
	}
	var lerr error
	load := func(dst *Pipeline, m Library, name string) {
		pl, pe := r.dev.NewComputePipeline(m, name)
		if pe != nil {
			lerr = fmt.Errorf("cuda: mamba kernel %q: %w", name, pe)
			return
		}
		*dst = pl
	}
	load(&r.mbConvP, mod, "mamba_conv")
	load(&r.mbSSMP, mod, "mamba_ssm")
	load(&r.mbGNormP, mod, "mamba_gnorm")
	if needNonGated {
		ng, ne := r.dev.CompileLibrary(nongatedActPTX)
		if ne != nil {
			return fmt.Errorf("cuda: JIT nongated_act.ptx: %w", ne)
		}
		load(&r.relu2QuantP, ng, "relu2_quant")
	}
	if lerr != nil {
		return lerr
	}
	r.mbProj, r.mbConv = r.af(p.projDim), r.af(p.convDim)
	r.mbY, r.mbGated = r.af(p.dInner), r.af(p.dInner)
	r.mbGq, r.mbGSc = r.ai(p.dInner/4), r.af(1)
	return nil
}

// mambaMixer runs one Mamba-2 layer's mixer and folds its output into the residual: the pre-norm and in_proj, the conv → SSM → gated-norm chain over the
// in_proj output [z | xBC | dt], then out_proj accumulating into x. The conv window and the SSM state are updated in place; the recurrence advances one
// token per call.
func (r *cudaResident) mambaMixer(Ly *cudaLayer) error {
	p := r.mamba
	nullBias := ArgNull()
	if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(Ly.mbIn, r.aq, r.aSc, nullBias, r.mbProj, 0); e != nil {
		return e
	}
	if e := r.launch(r.mbConvP, g1cfg(p.convDim, 256),
		Arg(r.mbProj), Arg(Ly.mbConvW), Arg(Ly.mbConvB), Arg(Ly.mbWin), Arg(r.mbConv),
		gpu.ArgValue(int32(p.convDim)), gpu.ArgValue(int32(p.dConv)), gpu.ArgValue(int32(p.dInner+mambaConvShiftForTest))); e != nil {
		return e
	}
	if e := r.launch(r.mbSSMP, g1cfg(p.nHeads*p.headDim, 64),
		Arg(r.mbConv), Arg(r.mbProj), Arg(Ly.mbHeadP), Arg(Ly.mbSSM), Arg(r.mbY),
		gpu.ArgValue(int32(p.nHeads)), gpu.ArgValue(int32(p.headDim)), gpu.ArgValue(int32(p.dState)), gpu.ArgValue(int32(p.repeat)),
		gpu.ArgValue(int32(p.gSize)), gpu.ArgValue(int32(p.dInner)), gpu.ArgValue(int32(p.dInner+p.convDim))); e != nil {
		return e
	}
	if e := r.launch(r.mbGNormP, LaunchConfig{GridX: uint32(p.normGroups), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		Arg(r.mbY), Arg(r.mbProj), Arg(Ly.mbNormW), Arg(r.mbGated),
		gpu.ArgValue(int32(p.dInner/p.normGroups)), gpu.ArgValue(r.eps), gpu.ArgValue(int32(0))); e != nil {
		return e
	}
	if e := r.launch(r.fQ, onecfg(256, 256*4), Arg(r.mbGated), gpu.ArgValue(int32(p.dInner)), Arg(r.mbGq), Arg(r.mbGSc)); e != nil {
		return e
	}
	return r.doG(Ly.mbOut, r.mbGq, r.mbGSc, nullBias, r.x, 1)
}

// nonGatedMLP runs a feed-forward-only block: the pre-norm and up GEMV, relu2_quant (squared ReLU and the symmetric int8 quantization the down GEMV reads),
// and the down GEMV accumulating into x.
func (r *cudaResident) nonGatedMLP(Ly *cudaLayer) error {
	nullBias := ArgNull()
	if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(Ly.u, r.aq, r.aSc, nullBias, r.uO, 0); e != nil {
		return e
	}
	if e := r.launch(r.relu2QuantP, onecfg(glueQuantThreads, glueQuantThreads*4),
		Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.ffnI)), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
		return e
	}
	return r.doG(Ly.d, r.dq, r.dSc, nullBias, r.x, 1)
}

// resetMambaState zeroes every Mamba layer's conv window and SSM state, which a fresh sequence must not inherit. No-op without Mamba layers.
func (r *cudaResident) resetMambaState() error {
	if r.mamba == nil || mambaResetSkipForTest {
		return nil
	}
	p := r.mamba
	winZ := make([]float32, (p.dConv-1)*p.convDim)
	ssmZ := make([]float32, p.nHeads*p.headDim*p.dState)
	for i := range r.layers {
		L := &r.layers[i]
		if L.mixer != decoder.MixerMamba {
			continue
		}
		if e := gpu.Upload(L.mbWin, winZ); e != nil {
			return e
		}
		if e := gpu.Upload(L.mbSSM, ssmZ); e != nil {
			return e
		}
	}
	return nil
}

// recurrent reports whether the resident carries per-sequence recurrent state (Gated-DeltaNet or Mamba-2) that a KV slot does not hold: the batched
// decode step, the prefix snapshot and the KV copy decline it, a verify pass runs one token at a time, and a fast-lever prefill is off.
func (r *cudaResident) recurrent() bool { return r.dnet != nil || r.mamba != nil }

// batchedVerifyOK reports whether ForwardN may run its rows as one weight-stationary pass: batched kernels exist and no recurrent state would have to
// advance token by token. VerifyPath reports the same answer.
func (r *cudaResident) batchedVerifyOK() bool { return r.prefillReady && !r.recurrent() }

// mambaStateBytes is the device memory the Mamba layers' conv window and SSM state take, each buffer at the driver's allocation quantum. The pre-build
// context plan prices it beside the weights; the state is allocated at upload, before the build's own fit check reads free VRAM.
func mambaStateBytes(p *mambaParams, layers int) int64 {
	if p == nil || layers <= 0 {
		return 0
	}
	win := int64((p.dConv-1)*p.convDim) * 4
	ssm := int64(p.nHeads*p.headDim*p.dState) * 4
	return int64(layers) * (win + allocRoundSlack(win) + ssm + allocRoundSlack(ssm))
}
