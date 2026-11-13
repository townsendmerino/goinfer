package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// BackendModelView is the narrow, read-only subsystem interface a GPU resident
// builder or placement planner uses to inspect a loaded Model's architecture,
// global geometry, and per-layer resident view without coupling to the 200+
// methods on *Model.
//
// Callers obtain it via m.BackendView(). Individual *Resident methods on *Model
// delegate through this view for backwards compatibility.
type BackendModelView interface {
	Dims() (hidden, nLayers, nH, nKV, hd, inter, vocab int)
	NumLayers() int
	Spec() ResidentSpec
	Weights() *Weights

	NormEps() float32
	AttnScale() float32
	AttnTempScale(pos int) float32
	AttnTempParams() (beta, origMaxPos float64)
	EmbedScale() float64
	EmbedLen() int

	SlidingWindow() int
	HasQKNorm() bool
	QKNormWhole() bool
	PartialRotary() bool
	RotaryDim() int
	RopeInvFreq() []float32
	PairwiseRoPE() bool
	MRopeSection() []int
	MRopeInterleaved() bool
	MRopeAxis(half int) []int

	RMSAddOne() bool
	LayerNorm() bool
	SandwichNorm() bool
	PostOnlyNorm() bool
	ParallelBlock() bool
	NonGatedMLP() bool
	GatedAct() int
	OutBias() bool
	LearnedPos() bool
	PosEmbed() *linalg.WeightMat

	FinalLogitSoftcap() float32
	AttnLogitSoftcap() float32
	LogitScale() (inv float32, ok bool)
	GptOssAct() (alpha, limit float32, ok bool)
	HasAttnSinks() bool

	IsGemma4() bool
	HasGemma4MoE() bool
	Gemma4PLEDim() int
	MoEExpertKind() string
	ProjsInt4() bool
	LoadAbort() <-chan struct{}

	Layer(i int) ResidentLayerView
}

// ResidentLayerView exposes one decoder layer's resident operations, geometry,
// RoPE schedule, and family-specific mixer/FFN weights to a backend builder.
type ResidentLayerView interface {
	Index() int
	Ops() (mixer ResidentMixer, ffn bool)

	HeadDim() int
	KVHeads() int
	RotaryDim() int
	KVSrc() int
	IsLocal() bool
	RopeGlobal() bool
	RopeInvFreq() []float32
	RopeMscale() float64
	VFromK() bool

	Gemma4DenseScalar() float32
	Gemma4MoEBundle() (Gemma4MoEResidentBundle, bool)

	AttnSinks() []float32
	GptOssExpertBias() []float32
	GptOssExpertDownBias() []float32

	MLAWeights() (qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj []float32)

	GraniteMamba() bool
	GraniteMambaWeights() (inProj, convW, convB, aLog, dW, dtBias, normW, outProj []float32)
	MambaProjInt8() (in, out *linalg.WeightMat)
	NemotronBlockKind() uint8

	Qwen35Linear() bool
	Qwen35DeltaWeights() (inQKV, inZ, outProj *linalg.WeightMat, inB, inA, convW, dtBias, negExpA, normW []float32)
	Qwen35AttnWeights() (qProj, kProj, vProj, oProj *linalg.WeightMat, qNorm, kNorm []float32)
}

type backendModelView struct {
	m *Model
}

type residentLayerView struct {
	m *Model
	i int
}

// BackendView returns the read-only BackendModelView for m.
func (m *Model) BackendView() BackendModelView {
	return backendModelView{m: m}
}

func (v backendModelView) Dims() (hidden, nLayers, nH, nKV, hd, inter, vocab int) {
	a := v.m.w.arch
	return a.HiddenDim, a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.VocabSize
}

func (v backendModelView) NumLayers() int              { return v.m.w.arch.NumLayers }
func (v backendModelView) Spec() ResidentSpec          { return v.m.ResidentSpec() }
func (v backendModelView) Weights() *Weights           { return v.m.w }
func (v backendModelView) NormEps() float32            { return v.m.NormEps() }
func (v backendModelView) AttnScale() float32          { return v.m.AttnScale() }
func (v backendModelView) EmbedScale() float64         { return v.m.w.arch.EmbedScale }
func (v backendModelView) SlidingWindow() int          { return v.m.w.arch.SlidingWindow }
func (v backendModelView) HasQKNorm() bool             { return v.m.w.arch.QKNorm }
func (v backendModelView) QKNormWhole() bool           { return v.m.w.arch.QKNormWhole }
func (v backendModelView) PairwiseRoPE() bool          { return v.m.w.arch.pairwiseRoPE() }
func (v backendModelView) MRopeSection() []int         { return v.m.w.arch.MRopeSection }
func (v backendModelView) MRopeInterleaved() bool      { return v.m.w.arch.MRopeInterleaved }
func (v backendModelView) RMSAddOne() bool             { return v.m.w.arch.RMSAddOne }
func (v backendModelView) LayerNorm() bool             { return v.m.w.arch.Norm == NormLayer }
func (v backendModelView) SandwichNorm() bool          { return v.m.w.arch.NormPlacement == NormSandwich4 }
func (v backendModelView) PostOnlyNorm() bool          { return v.m.w.arch.NormPlacement == NormPostOnly }
func (v backendModelView) ParallelBlock() bool         { return v.m.w.arch.NormPlacement == NormParallel }
func (v backendModelView) NonGatedMLP() bool           { return v.m.w.arch.NonGatedMLP }
func (v backendModelView) GatedAct() int               { return int(v.m.w.arch.Act) }
func (v backendModelView) OutBias() bool               { return v.m.w.arch.OutBias }
func (v backendModelView) LearnedPos() bool            { return v.m.w.arch.LearnedPosEmbed }
func (v backendModelView) PosEmbed() *linalg.WeightMat { return &v.m.w.PosEmbed }
func (v backendModelView) FinalLogitSoftcap() float32  { return float32(v.m.w.arch.FinalLogitSoftcap) }
func (v backendModelView) AttnLogitSoftcap() float32   { return float32(v.m.w.arch.AttnLogitSoftcap) }
func (v backendModelView) HasAttnSinks() bool          { return v.m.w.arch.hasAttnSinks() }
func (v backendModelView) IsGemma4() bool              { return v.m.w.arch.gemma4 != nil }
func (v backendModelView) ProjsInt4() bool             { return v.m.residentProjsInt4() }
func (v backendModelView) LoadAbort() <-chan struct{}  { return v.m.loadAbort }

func (v backendModelView) AttnTempScale(pos int) float32 {
	a := v.m.w.arch
	if a.AttnTempBeta == 0 {
		return 1
	}
	return float32(1 + a.AttnTempBeta*math.Log1p(math.Floor(float64(pos)/a.AttnTempOrigMaxPos)))
}

func (v backendModelView) AttnTempParams() (beta, origMaxPos float64) {
	return v.m.w.arch.AttnTempBeta, v.m.w.arch.AttnTempOrigMaxPos
}

func (v backendModelView) EmbedLen() int {
	n := v.m.w.arch.HiddenDim
	if g4 := v.m.w.arch.gemma4; g4 != nil {
		n += v.m.w.arch.NumLayers * g4.HiddenSizePerLayerInput
	}
	return n
}

func (v backendModelView) PartialRotary() bool {
	return v.m.w.arch.RotaryDim != 0 && v.m.w.arch.RotaryDim < v.m.w.arch.HeadDim
}

func (v backendModelView) RotaryDim() int {
	if v.m.w.arch.RotaryDim != 0 && v.m.w.arch.RotaryDim < v.m.w.arch.HeadDim {
		return v.m.w.arch.RotaryDim
	}
	return v.m.w.arch.HeadDim
}

func (v backendModelView) RopeInvFreq() []float32 {
	inv := v.m.w.arch.ropeInvFreq(0)
	out := make([]float32, len(inv))
	for i, val := range inv {
		out[i] = float32(val)
	}
	return out
}

func (v backendModelView) MRopeAxis(half int) []int {
	sec := v.m.w.arch.MRopeSection
	if len(sec) != 3 || half <= 0 {
		return nil
	}
	comp := mropeComponent
	if v.m.w.arch.MRopeInterleaved {
		comp = mropeComponentInterleaved
	}
	axis := make([]int, half)
	for d := range axis {
		axis[d] = comp(d, sec)
	}
	return axis
}

func (v backendModelView) LogitScale() (inv float32, ok bool) {
	if v.m.w.arch.LogitScale == 0 || v.m.w.arch.LogitScale == 1 {
		return 1, false
	}
	return float32(1 / v.m.w.arch.LogitScale), true
}

func (v backendModelView) GptOssAct() (alpha, limit float32, ok bool) {
	gp := v.m.w.arch.gptoss
	if gp == nil {
		return 0, 0, false
	}
	return float32(gp.SwigluAlpha), float32(gp.SwigluLimit), true
}

func (v backendModelView) HasGemma4MoE() bool {
	if v.m.w.arch.gemma4 == nil {
		return false
	}
	for l := range v.m.w.Layers {
		if v.m.w.Layers[l].gemma4moe != nil {
			return true
		}
	}
	return false
}

func (v backendModelView) Gemma4PLEDim() int {
	if g4 := v.m.w.arch.gemma4; g4 != nil {
		return g4.HiddenSizePerLayerInput
	}
	return 0
}

func (v backendModelView) MoEExpertKind() string {
	for i := range v.m.w.Layers {
		lw := &v.m.w.Layers[i]
		if gm := lw.gemma4moe; gm != nil && len(gm.expertsGateUp) > 0 {
			return gm.expertsGateUp[0].Kind()
		}
		if len(lw.Experts) > 0 {
			e := &lw.Experts[0]
			for _, w := range []*linalg.WeightMat{&e.Gate, &e.Up, &e.Down} {
				if w.Rows() > 0 {
					return w.Kind()
				}
			}
		}
	}
	return ""
}

func (v backendModelView) Layer(i int) ResidentLayerView {
	return residentLayerView{m: v.m, i: i}
}

func (lv residentLayerView) Index() int { return lv.i }

func (lv residentLayerView) Ops() (mixer ResidentMixer, ffn bool) {
	a := lv.m.w.arch
	i := lv.i
	switch {
	case a.nemotron != nil:
		switch a.nemotron.blockKind[i] {
		case nemoMamba:
			return MixerMamba, false
		case nemoAttn:
			return MixerAttention, false
		default: // nemoMLP, nemoMoE
			return MixerNone, true
		}
	case a.qwen35 != nil && a.isLinearLayer(i):
		return MixerDeltaNet, true
	case a.granite != nil && a.isMambaLayer(i):
		return MixerMamba, true
	case a.mla != nil:
		return MixerMLA, true
	}
	return MixerAttention, true
}

func (lv residentLayerView) HeadDim() int        { return lv.m.w.arch.headDimAt(lv.i) }
func (lv residentLayerView) KVHeads() int        { return lv.m.w.arch.kvHeadsAt(lv.i) }
func (lv residentLayerView) KVSrc() int          { return lv.m.w.arch.gemma4KVSrcAt(lv.i) }
func (lv residentLayerView) RopeGlobal() bool    { return lv.m.w.arch.isGlobalLayer(lv.i) }
func (lv residentLayerView) RopeMscale() float64 { return lv.m.w.arch.ropeMscale(lv.i) }

func (lv residentLayerView) RotaryDim() int {
	if lv.m.w.arch.gemma4 != nil {
		return lv.m.w.arch.headDimAt(lv.i)
	}
	return lv.m.BackendView().RotaryDim()
}

func (lv residentLayerView) IsLocal() bool {
	return lv.m.w.arch.SlidingWindow > 0 && !lv.m.w.arch.isGlobalLayer(lv.i)
}

func (lv residentLayerView) RopeInvFreq() []float32 {
	a := lv.m.w.arch
	i := lv.i
	if a.gemma4 != nil {
		hd, rot, base := a.HeadDim, a.HeadDim, a.RoPELocalBase
		if a.isGlobalLayer(i) {
			hd, rot, base = a.gemma4.GlobalHeadDim, a.gemma4.GlobalRotaryDim, a.RoPEGlobalBase
		}
		f := gemma4InvFreq(hd, rot, base)
		out := make([]float32, len(f))
		for j, val := range f {
			out[j] = float32(val)
		}
		return out
	}
	return lv.m.RopeInvFreqLayer(i)
}

func (lv residentLayerView) VFromK() bool {
	i := lv.i
	if lv.m.w.arch.gemma4 == nil {
		return false
	}
	if lv.m.w.arch.gemma4.KVShared && lv.m.w.arch.isGlobalLayer(i) {
		return true
	}
	return i >= 0 && i < len(lv.m.w.Layers) && lv.m.w.Layers[i].VFromK
}

func (lv residentLayerView) Gemma4DenseScalar() float32 {
	i := lv.i
	if lv.m.w.arch.gemma4 == nil || i < 0 || i >= len(lv.m.w.Layers) {
		return 0
	}
	return lv.m.w.Layers[i].LayerScalar
}

func (lv residentLayerView) Gemma4MoEBundle() (b Gemma4MoEResidentBundle, ok bool) {
	m := lv.m
	l := lv.i
	if m.w.arch.gemma4 == nil || l < 0 || l >= len(m.w.Layers) {
		return b, false
	}
	gm := m.w.Layers[l].gemma4moe
	if gm == nil {
		return b, false
	}
	proj, has := gm.routerProj.F32()
	if !has {
		logs.Warnf("[resident] gemma4 MoE layer %d: router is %q, not f32 — declining", l, gm.routerProj.Kind())
		return b, false
	}
	if len(gm.routerScale) != m.w.arch.HiddenDim {
		return b, false
	}
	H := m.w.arch.HiddenDim
	root := float32(math.Pow(float64(H), -0.5))
	scaled := make([]float32, len(proj))
	for e := 0; e < gm.nE; e++ {
		for i := range H {
			scaled[e*H+i] = proj[e*H+i] * gm.routerScale[i] * root
		}
	}
	gu := make([]*linalg.WeightMat, gm.nE)
	dn := make([]*linalg.WeightMat, gm.nE)
	for e := 0; e < gm.nE; e++ {
		gu[e], dn[e] = &gm.expertsGateUp[e], &gm.expertsDown[e]
	}
	return Gemma4MoEResidentBundle{
		PreFFNNorm: gm.preFFNNorm, PostFFNNorm1: gm.postFFNNorm1, PreFFNNorm2: gm.preFFNNorm2,
		PostFFNNorm2: gm.postFFNNorm2, PostFFNNorm: gm.postFFNNorm,
		MlpGate: &gm.mlpGate, MlpUp: &gm.mlpUp, MlpDown: &gm.mlpDown,
		RouterProjScaled: scaled, PerExpertScale: gm.perExpertScale,
		ExpertsGateUp: gu, ExpertsDown: dn,
		LayerScalar: gm.layerScalar, DenseInter: gm.denseInter, MoeInter: gm.moeInter,
		NE: gm.nE, TopK: gm.topK,
	}, true
}

func (lv residentLayerView) AttnSinks() []float32 {
	l := lv.i
	if !lv.m.w.arch.hasAttnSinks() || l < 0 || l >= len(lv.m.w.Layers) {
		return nil
	}
	return lv.m.w.Layers[l].AttnSinks
}

func (lv residentLayerView) GptOssExpertBias() []float32 {
	m := lv.m
	l := lv.i
	if m.w.arch.gptoss == nil || m.w.arch.MoE == nil || l < 0 || l >= len(m.w.Layers) {
		return nil
	}
	lw := &m.w.Layers[l]
	if len(lw.Experts) == 0 || lw.Experts[0].GateBias == nil {
		return nil
	}
	inter := m.w.arch.MoE.IntermediateDim
	out := make([]float32, len(lw.Experts)*2*inter)
	for e := range lw.Experts {
		copy(out[e*2*inter:], lw.Experts[e].GateBias)
		copy(out[e*2*inter+inter:], lw.Experts[e].UpBias)
	}
	return out
}

func (lv residentLayerView) GptOssExpertDownBias() []float32 {
	m := lv.m
	l := lv.i
	if m.w.arch.gptoss == nil || m.w.arch.MoE == nil || l < 0 || l >= len(m.w.Layers) {
		return nil
	}
	lw := &m.w.Layers[l]
	if len(lw.Experts) == 0 || lw.Experts[0].DownBias == nil {
		return nil
	}
	hidden := m.w.arch.HiddenDim
	out := make([]float32, len(lw.Experts)*hidden)
	for e := range lw.Experts {
		copy(out[e*hidden:], lw.Experts[e].DownBias)
	}
	return out
}

func (lv residentLayerView) MLAWeights() (qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj []float32) {
	w := lv.m.w.Layers[lv.i].mla
	if w == nil {
		return
	}
	return w.qAProj, w.qALayernorm, w.qBProj, w.qProj, w.kvAProj, w.kvALayernorm, w.kvBProj, w.oProj
}

func (lv residentLayerView) GraniteMamba() bool { return lv.m.w.arch.isMambaLayer(lv.i) }

func (lv residentLayerView) GraniteMambaWeights() (inProj, convW, convB, aLog, dW, dtBias, normW, outProj []float32) {
	w := lv.m.w.Layers[lv.i].mamba
	if w == nil {
		return
	}
	return w.inProj, w.convW, w.convB, w.aLog, w.d, w.dtBias, w.normW, w.outProj
}

func (lv residentLayerView) MambaProjInt8() (in, out *linalg.WeightMat) {
	w := lv.m.w.Layers[lv.i].mamba
	if w == nil || w.inQ.Rows() == 0 {
		return nil, nil
	}
	return &w.inQ, &w.outQ
}

func (lv residentLayerView) NemotronBlockKind() uint8 {
	return lv.m.w.arch.nemotron.blockKind[lv.i]
}

func (lv residentLayerView) Qwen35Linear() bool {
	return lv.m.w.arch.isLinearLayer(lv.i)
}

func (lv residentLayerView) Qwen35DeltaWeights() (inQKV, inZ, outProj *linalg.WeightMat, inB, inA, convW, dtBias, negExpA, normW []float32) {
	w := lv.m.w.Layers[lv.i].delta
	if w == nil {
		return
	}
	return &w.inProjQKV, &w.inProjZ, &w.outProj, w.inProjB, w.inProjA, w.convW, w.dtBias, w.negExpA, w.normW
}

func (lv residentLayerView) Qwen35AttnWeights() (qProj, kProj, vProj, oProj *linalg.WeightMat, qNorm, kNorm []float32) {
	a := lv.m.w.Layers[lv.i].qattn
	if a == nil {
		return
	}
	return &a.qProj, &a.kProj, &a.vProj, &a.oProj, a.qNorm, a.kNorm
}
