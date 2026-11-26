package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// BackendModelView is a read-only view over a loaded Model's architecture,
// global geometry, and per-layer resident metadata, obtained via m.BackendView().
type BackendModelView struct {
	m *Model
}

// ResidentLayerView is a read-only view of one decoder layer's resident
// operations, geometry, RoPE schedule, and family-specific mixer/FFN weights.
type ResidentLayerView struct {
	m *Model
	i int
}

// BackendView returns the read-only BackendModelView for m.
func (m *Model) BackendView() BackendModelView {
	return BackendModelView{m: m}
}

func (v BackendModelView) Dims() ResidentDims {
	a := v.m.w.Arch
	return ResidentDims{Hidden: a.HiddenDim, Layers: a.NumLayers, Heads: a.NumHeads, KVHeads: a.NumKVHeads, HeadDim: a.HeadDim,
		Inter: a.IntermediateDim, Vocab: a.VocabSize}
}

func (v BackendModelView) NumLayers() int              { return v.m.w.Arch.NumLayers }
func (v BackendModelView) Spec() ResidentSpec          { return v.m.ResidentSpec() }
func (v BackendModelView) Weights() *darch.Weights     { return v.m.w }
func (v BackendModelView) NormEps() float32            { return v.m.NormEps() }
func (v BackendModelView) AttnScale() float32          { return v.m.AttnScale() }
func (v BackendModelView) EmbedScale() float64         { return v.m.w.Arch.EmbedScale }
func (v BackendModelView) SlidingWindow() int          { return v.m.w.Arch.SlidingWindow }
func (v BackendModelView) HasQKNorm() bool             { return v.m.w.Arch.QKNorm }
func (v BackendModelView) QKNormWhole() bool           { return v.m.w.Arch.QKNormWhole }
func (v BackendModelView) PairwiseRoPE() bool          { return v.m.w.Arch.PairwiseRoPE() }
func (v BackendModelView) MRopeSection() []int         { return v.m.w.Arch.MRopeSection }
func (v BackendModelView) MRopeInterleaved() bool      { return v.m.w.Arch.MRopeInterleaved }
func (v BackendModelView) RMSAddOne() bool             { return v.m.w.Arch.RMSAddOne }
func (v BackendModelView) LayerNorm() bool             { return v.m.w.Arch.Norm == darch.NormLayer }
func (v BackendModelView) SandwichNorm() bool          { return v.m.w.Arch.NormPlacement == darch.NormSandwich4 }
func (v BackendModelView) PostOnlyNorm() bool          { return v.m.w.Arch.NormPlacement == darch.NormPostOnly }
func (v BackendModelView) ParallelBlock() bool         { return v.m.w.Arch.NormPlacement == darch.NormParallel }
func (v BackendModelView) NonGatedMLP() bool           { return v.m.w.Arch.NonGatedMLP }
func (v BackendModelView) GatedAct() int               { return int(v.m.w.Arch.Act) }
func (v BackendModelView) OutBias() bool               { return v.m.w.Arch.OutBias }
func (v BackendModelView) LearnedPos() bool            { return v.m.w.Arch.LearnedPosEmbed }
func (v BackendModelView) PosEmbed() *linalg.WeightMat { return &v.m.w.PosEmbed }
func (v BackendModelView) FinalLogitSoftcap() float32  { return float32(v.m.w.Arch.FinalLogitSoftcap) }
func (v BackendModelView) AttnLogitSoftcap() float32   { return float32(v.m.w.Arch.AttnLogitSoftcap) }
func (v BackendModelView) HasAttnSinks() bool          { return v.m.w.Arch.HasAttnSinks() }
func (v BackendModelView) IsGemma4() bool              { return v.m.w.Arch.Gemma4 != nil }
func (v BackendModelView) ProjsInt4() bool             { return v.m.residentProjsInt4() }
func (v BackendModelView) LoadAbort() <-chan struct{}  { return v.m.opts.LoadAbort }

func (v BackendModelView) MoE() (ResidentMoESpec, bool) {
	if mo := v.m.ResidentSpec().MoE; mo != nil {
		return *mo, true
	}
	return ResidentMoESpec{}, false
}

func (v BackendModelView) MLA() (ResidentMLASpec, bool) {
	if p := v.m.ResidentSpec().MLA; p != nil {
		return *p, true
	}
	return ResidentMLASpec{}, false
}

func (v BackendModelView) Qwen35() (ResidentQwen35Spec, bool) {
	if g := v.m.ResidentSpec().Qwen35; g != nil {
		return *g, true
	}
	return ResidentQwen35Spec{}, false
}

func (v BackendModelView) Nemotron() (ResidentNemotronSpec, bool) {
	if np := v.m.ResidentSpec().Nemotron; np != nil {
		return *np, true
	}
	return ResidentNemotronSpec{}, false
}

func (v BackendModelView) Granite() (ResidentGraniteSpec, bool) {
	if g := v.m.ResidentSpec().Granite; g != nil {
		return *g, true
	}
	return ResidentGraniteSpec{}, false
}

func (v BackendModelView) AttnTempScale(pos int) float32 {
	a := v.m.w.Arch
	if a.AttnTempBeta == 0 {
		return 1
	}
	return float32(1 + a.AttnTempBeta*math.Log1p(math.Floor(float64(pos)/a.AttnTempOrigMaxPos)))
}

func (v BackendModelView) AttnTempParams() (beta, origMaxPos float64) {
	return v.m.w.Arch.AttnTempBeta, v.m.w.Arch.AttnTempOrigMaxPos
}

func (v BackendModelView) EmbedLen() int {
	n := v.m.w.Arch.HiddenDim
	if g4 := v.m.w.Arch.Gemma4; g4 != nil {
		n += v.m.w.Arch.NumLayers * g4.HiddenSizePerLayerInput
	}
	return n
}

func (v BackendModelView) PartialRotary() bool {
	return v.m.w.Arch.RotaryDim != 0 && v.m.w.Arch.RotaryDim < v.m.w.Arch.HeadDim
}

func (v BackendModelView) RotaryDim() int {
	if v.m.w.Arch.RotaryDim != 0 && v.m.w.Arch.RotaryDim < v.m.w.Arch.HeadDim {
		return v.m.w.Arch.RotaryDim
	}
	return v.m.w.Arch.HeadDim
}

func (v BackendModelView) RopeInvFreq() []float32 {
	inv := v.m.w.Arch.RoPEInvFreq(0)
	out := make([]float32, len(inv))
	for i, val := range inv {
		out[i] = float32(val)
	}
	return out
}

func (v BackendModelView) MRopeAxis(half int) []int {
	sec := v.m.w.Arch.MRopeSection
	if len(sec) != 3 || half <= 0 {
		return nil
	}
	comp := rope.MRoPEComponent
	if v.m.w.Arch.MRopeInterleaved {
		comp = rope.MRoPEComponentInterleaved
	}
	axis := make([]int, half)
	for d := range axis {
		axis[d] = comp(d, sec)
	}
	return axis
}

func (v BackendModelView) LogitScale() (inv float32, ok bool) {
	if v.m.w.Arch.LogitScale == 0 || v.m.w.Arch.LogitScale == 1 {
		return 1, false
	}
	return float32(1 / v.m.w.Arch.LogitScale), true
}

func (v BackendModelView) GptOssAct() (alpha, limit float32, ok bool) {
	gp := v.m.w.Arch.GPTOSS
	if gp == nil {
		return 0, 0, false
	}
	return float32(gp.SwigluAlpha), float32(gp.SwigluLimit), true
}

func (v BackendModelView) HasGemma4MoE() bool {
	if v.m.w.Arch.Gemma4 == nil {
		return false
	}
	for l := range v.m.w.Layers {
		if v.m.w.Layers[l].Gemma4MoE != nil {
			return true
		}
	}
	return false
}

func (v BackendModelView) Gemma4PLEDim() int {
	if g4 := v.m.w.Arch.Gemma4; g4 != nil {
		return g4.HiddenSizePerLayerInput
	}
	return 0
}

func (v BackendModelView) MoEExpertKind() string {
	for i := range v.m.w.Layers {
		lw := &v.m.w.Layers[i]
		if gm := lw.Gemma4MoE; gm != nil && len(gm.ExpertsGateUp) > 0 {
			return gm.ExpertsGateUp[0].Kind().String()
		}
		if len(lw.Experts) > 0 {
			e := &lw.Experts[0]
			for _, w := range []*linalg.WeightMat{&e.Gate, &e.Up, &e.Down} {
				if w.Rows() > 0 {
					return w.Kind().String()
				}
			}
		}
	}
	return ""
}

func (v BackendModelView) Layer(i int) ResidentLayerView {
	return ResidentLayerView{m: v.m, i: i}
}

func (lv ResidentLayerView) Index() int { return lv.i }

func (lv ResidentLayerView) Ops() (mixer ResidentMixer, ffn bool) {
	a := lv.m.w.Arch
	i := lv.i
	switch {
	case a.Nemotron != nil:
		switch a.Nemotron.BlockKind[i] {
		case archparts.NemoMamba:
			return MixerMamba, false
		case archparts.NemoAttn:
			return MixerAttention, false
		default: // nemoMLP, nemoMoE
			return MixerNone, true
		}
	case a.Qwen35 != nil && a.IsLinearLayer(i):
		return MixerDeltaNet, true
	case a.Granite != nil && a.IsMambaLayer(i):
		return MixerMamba, true
	case a.MLA != nil:
		return MixerMLA, true
	}
	return MixerAttention, true
}

func (lv ResidentLayerView) HeadDim() int        { return lv.m.w.Arch.HeadDimAt(lv.i) }
func (lv ResidentLayerView) KVHeads() int        { return lv.m.w.Arch.KVHeadsAt(lv.i) }
func (lv ResidentLayerView) KVSrc() int          { return lv.m.w.Arch.Gemma4KVSrcAt(lv.i) }
func (lv ResidentLayerView) RopeGlobal() bool    { return lv.m.w.Arch.IsGlobalLayer(lv.i) }
func (lv ResidentLayerView) RopeMscale() float64 { return lv.m.w.Arch.RoPEMscale(lv.i) }

func (lv ResidentLayerView) RotaryDim() int {
	if lv.m.w.Arch.Gemma4 != nil {
		return lv.m.w.Arch.HeadDimAt(lv.i)
	}
	return lv.m.BackendView().RotaryDim()
}

func (lv ResidentLayerView) IsLocal() bool {
	return lv.m.w.Arch.SlidingWindow > 0 && !lv.m.w.Arch.IsGlobalLayer(lv.i)
}

func (lv ResidentLayerView) RopeInvFreq() []float32 {
	a := lv.m.w.Arch
	i := lv.i
	if a.Gemma4 != nil {
		hd, rot, base := a.HeadDim, a.HeadDim, a.RoPELocalBase
		if a.IsGlobalLayer(i) {
			hd, rot, base = a.Gemma4.GlobalHeadDim, a.Gemma4.GlobalRotaryDim, a.RoPEGlobalBase
		}
		f := archparts.Gemma4InvFreq(hd, rot, base)
		out := make([]float32, len(f))
		for j, val := range f {
			out[j] = float32(val)
		}
		return out
	}
	return lv.m.RopeInvFreqLayer(i)
}

func (lv ResidentLayerView) VFromK() bool {
	i := lv.i
	if lv.m.w.Arch.Gemma4 == nil {
		return false
	}
	if lv.m.w.Arch.Gemma4.KVShared && lv.m.w.Arch.IsGlobalLayer(i) {
		return true
	}
	return i >= 0 && i < len(lv.m.w.Layers) && lv.m.w.Layers[i].VFromK
}

func (lv ResidentLayerView) Gemma4DenseScalar() float32 {
	i := lv.i
	if lv.m.w.Arch.Gemma4 == nil || i < 0 || i >= len(lv.m.w.Layers) {
		return 0
	}
	return lv.m.w.Layers[i].LayerScalar
}

func (lv ResidentLayerView) Gemma4MoEBundle() (b Gemma4MoEResidentBundle, ok bool) {
	m := lv.m
	l := lv.i
	if m.w.Arch.Gemma4 == nil || l < 0 || l >= len(m.w.Layers) {
		return b, false
	}
	gm := m.w.Layers[l].Gemma4MoE
	if gm == nil {
		return b, false
	}
	proj, has := gm.RouterProj.F32()
	if !has {
		logs.Warnf("[resident] gemma4 MoE layer %d: router is %q, not f32 — declining", l, gm.RouterProj.Kind())
		return b, false
	}
	if len(gm.RouterScale) != m.w.Arch.HiddenDim {
		return b, false
	}
	H := m.w.Arch.HiddenDim
	root := float32(math.Pow(float64(H), -0.5))
	scaled := make([]float32, len(proj))
	for e := 0; e < gm.NE; e++ {
		for i := range H {
			scaled[e*H+i] = proj[e*H+i] * gm.RouterScale[i] * root
		}
	}
	gu := make([]*linalg.WeightMat, gm.NE)
	dn := make([]*linalg.WeightMat, gm.NE)
	for e := 0; e < gm.NE; e++ {
		gu[e], dn[e] = &gm.ExpertsGateUp[e], &gm.ExpertsDown[e]
	}
	return Gemma4MoEResidentBundle{
		PreFFNNorm: gm.PreFFNNorm, PostFFNNorm1: gm.PostFFNNorm1, PreFFNNorm2: gm.PreFFNNorm2,
		PostFFNNorm2: gm.PostFFNNorm2, PostFFNNorm: gm.PostFFNNorm,
		MlpGate: &gm.MlpGate, MlpUp: &gm.MlpUp, MlpDown: &gm.MlpDown,
		RouterProjScaled: scaled, PerExpertScale: gm.PerExpertScale,
		ExpertsGateUp: gu, ExpertsDown: dn,
		LayerScalar: gm.LayerScalar, DenseInter: gm.DenseInter, MoeInter: gm.MoeInter,
		NE: gm.NE, TopK: gm.TopK,
	}, true
}

func (lv ResidentLayerView) AttnSinks() []float32 {
	l := lv.i
	if !lv.m.w.Arch.HasAttnSinks() || l < 0 || l >= len(lv.m.w.Layers) {
		return nil
	}
	return lv.m.w.Layers[l].AttnSinks
}

func (lv ResidentLayerView) GptOssExpertBias() []float32 {
	m := lv.m
	l := lv.i
	if m.w.Arch.GPTOSS == nil || m.w.Arch.MoE == nil || l < 0 || l >= len(m.w.Layers) {
		return nil
	}
	lw := &m.w.Layers[l]
	if len(lw.Experts) == 0 || lw.Experts[0].GateBias == nil {
		return nil
	}
	inter := m.w.Arch.MoE.IntermediateDim
	out := make([]float32, len(lw.Experts)*2*inter)
	for e := range lw.Experts {
		copy(out[e*2*inter:], lw.Experts[e].GateBias)
		copy(out[e*2*inter+inter:], lw.Experts[e].UpBias)
	}
	return out
}

func (lv ResidentLayerView) GptOssExpertDownBias() []float32 {
	m := lv.m
	l := lv.i
	if m.w.Arch.GPTOSS == nil || m.w.Arch.MoE == nil || l < 0 || l >= len(m.w.Layers) {
		return nil
	}
	lw := &m.w.Layers[l]
	if len(lw.Experts) == 0 || lw.Experts[0].DownBias == nil {
		return nil
	}
	hidden := m.w.Arch.HiddenDim
	out := make([]float32, len(lw.Experts)*hidden)
	for e := range lw.Experts {
		copy(out[e*hidden:], lw.Experts[e].DownBias)
	}
	return out
}

func (lv ResidentLayerView) MLAWeights() (qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj []float32) {
	w := lv.m.w.Layers[lv.i].MLA
	if w == nil {
		return
	}
	return w.QAProj, w.QALayernorm, w.QBProj, w.QProj, w.KVAProj, w.KVALayernorm, w.KVBProj, w.OProj
}

func (lv ResidentLayerView) GraniteMamba() bool { return lv.m.w.Arch.IsMambaLayer(lv.i) }

func (lv ResidentLayerView) GraniteMambaWeights() (inProj, convW, convB, aLog, dW, dtBias, normW, outProj []float32) {
	w := lv.m.w.Layers[lv.i].Mamba
	if w == nil {
		return
	}
	return w.InProj, w.ConvW, w.ConvB, w.ALog, w.D, w.DtBias, w.NormW, w.OutProj
}

func (lv ResidentLayerView) MambaProjInt8() (in, out *linalg.WeightMat) {
	w := lv.m.w.Layers[lv.i].Mamba
	if w == nil || w.InQ.Rows() == 0 {
		return nil, nil
	}
	return &w.InQ, &w.OutQ
}

func (lv ResidentLayerView) NemotronBlockKind() uint8 {
	return lv.m.w.Arch.Nemotron.BlockKind[lv.i]
}

func (lv ResidentLayerView) Qwen35Linear() bool {
	return lv.m.w.Arch.IsLinearLayer(lv.i)
}

func (lv ResidentLayerView) Qwen35DeltaWeights() (inQKV, inZ, outProj *linalg.WeightMat, inB, inA, convW, dtBias, negExpA, normW []float32) {
	w := lv.m.w.Layers[lv.i].Delta
	if w == nil {
		return
	}
	return &w.InProjQKV, &w.InProjZ, &w.OutProj, w.InProjB, w.InProjA, w.ConvW, w.DtBias, w.NegExpA, w.NormW
}

func (lv ResidentLayerView) Qwen35AttnWeights() (qProj, kProj, vProj, oProj *linalg.WeightMat, qNorm, kNorm []float32) {
	a := lv.m.w.Layers[lv.i].QAttn
	if a == nil {
		return
	}
	return &a.QProj, &a.KProj, &a.VProj, &a.OProj, a.QNorm, a.KNorm
}
