package arch

import (
	"slices"

	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// mlaOpParamsAt is layer l's opMLA parameters.
func (a *Architecture) mlaOpParamsAt(l int) mlaOpParams {
	return mlaOpParams{InvFreq: a.RoPEInvFreq(l), Mscale: a.RoPEMscale(l), Interleave: a.MLA.RoPEInterleave, Eps: a.NormEps,
		Scale: a.AttnScale, HeadWise: a.MLA.GateGranularity == "head_wise"}
}

// EffectiveEmbedScale is the factor the embedding stage multiplies by: Granite's embedding_multiplier, else EmbedScale.
func (a *Architecture) EffectiveEmbedScale() float64 {
	if a.Granite != nil {
		return float64(a.Granite.EmbMul)
	}
	return a.EmbedScale
}

func (a *Architecture) NormForm() normForm {
	if a.Norm == NormLayer {
		return normLayer
	}
	return normRMS
}

// LayerOps is one generic layer: the placement decides the order of norms, sublayers and residual adds.
func (a *Architecture) LayerOps(l int, hasExperts bool) []op {
	placement := a.normPlacementAt(l)
	postOnly := placement == NormPostOnly
	postNorm := placement == NormSandwich4 || postOnly
	input := func(slot normSlot) op {
		if postOnly {
			return op{Kind: opNorm, In: regH, Out: regX, Norm: normParams{Slot: slot, Form: normCopy}}
		}
		return op{Kind: opNorm, In: regH, Out: regX, Norm: normParams{Slot: slot, Form: a.NormForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true}}
	}
	post := func(slot normSlot) op {
		return op{Kind: opNorm, In: regA, Out: regA, Norm: normParams{Slot: slot, Form: a.NormForm(), Eps: a.NormEps, AddOne: a.RMSAddOne}}
	}
	ffn := func(out opReg) op {
		f := ffnParams{Form: ffnGated, Act: a.Act}
		switch {
		case a.GPTOSS != nil:
			f.Form, f.MoE, f.Alpha, f.Limit = ffnGptOss, *a.MoE, a.GPTOSS.SwigluAlpha, a.GPTOSS.SwigluLimit
		case a.MoE != nil && hasExperts:
			f.Form, f.MoE = ffnMoE, *a.MoE
		case a.NonGatedMLP:
			f.Form = ffnNonGated
		}
		return op{Kind: opFFN, In: regX, Out: out, FFN: f}
	}

	if a.Llama4 != nil {
		return a.llama4LayerOps(l, input, ffn)
	}
	if a.Inkling2 != nil {
		return a.inkling2LayerOps(l)
	}
	if a.Gemma4 != nil {
		return a.gemma4LayerOps(l, hasExperts)
	}
	if a.Granite != nil {
		return a.graniteLayerOps(l, input, ffn)
	}
	if a.Nemotron != nil {
		return a.nemotronLayerOps(l, input, ffn)
	}
	if a.LFM2 != nil && a.IsConvLayer(l) {
		return []op{input(slotPreAttn), {Kind: opShortConv, In: regX, Out: regA}, {Kind: opResidual}, input(slotPreMLP), ffn(regA), {Kind: opResidual}}
	}
	if a.MLA != nil {
		mixer := op{Kind: opMLA, In: regX, Out: regA, MLA: a.mlaOpParamsAt(l)}
		if a.KDA != nil && a.IsLinearLayer(l) { // Bailing Hybrid: KDA on the linear layers, MLA on the rest
			mixer = op{Kind: opKDA, In: regX, Out: regA, KDA: kdaOpParams{Eps: a.NormEps, LowerBound: a.KDA.LowerBound}}
		}
		return []op{input(slotPreAttn), mixer, {Kind: opResidual}, input(slotPreMLP), ffn(regA), {Kind: opResidual}}
	}
	ops := []op{input(slotPreAttn)}
	ops = append(ops, a.mixerOps(l)...)
	if placement == NormParallel {
		// One input norm feeds both sublayers, and their outputs meet in one residual add.
		return append(ops, ffn(regB), op{Kind: opResidual, Res: residualParams{Two: true}})
	}
	if postNorm {
		ops = append(ops, post(slotPostAttn))
	}
	ops = append(ops, op{Kind: opResidual}, input(slotPreMLP), ffn(regA))
	if postNorm {
		ops = append(ops, post(slotPostMLP))
	}
	return append(ops, op{Kind: opResidual})
}

// mixerOps is a generic-placement layer's first sublayer, regX → regA: qwen3.5's family puts a Gated DeltaNet on its linear
// layers and its gated attention on the rest (Olmo Hybrid keeps the generic attention, in the batched kernel's order);
// every other family the generic attention.
func (a *Architecture) mixerOps(l int) []op {
	g := a.Qwen35
	switch {
	case g == nil:
		return a.AttentionOps(l)
	case a.IsLinearLayer(l):
		return []op{{Kind: opDeltaNet, In: regX, Out: regA, Delta: deltaNetOpParams{Eps: a.NormEps, ONormEps: g.ONormEps, NegEigval: g.NegEigval}}}
	case g.PlainFullAttn:
		ops := a.AttentionOps(l)
		for i := range ops {
			if ops[i].Kind == opAttention {
				ops[i].Attn.Order = attnBatchedAcc64
			}
		}
		return ops
	}
	return []op{{Kind: opGatedAttn, In: regX, Out: regA, Gated: gatedAttnParams{
		QKNorm: qkNormParams{Form: qkRMS, Eps: a.NormEps, AddOne: a.RMSAddOne},
		// The rotary is the half-split form whatever ropeInterleave says, as it always was for this family.
		Rope: ropeParams{Kernel: ropeAtKernel, InvFreq: a.RoPEInvFreq(l), Mscale: a.RoPEMscale(l), MRopeSection: a.MRopeSection, MRopeInterleaved: a.MRopeInterleaved},
		Attn: attentionParams{Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnQuery},
	}}}
}

// AttentionOps is the generic attention sublayer, regX → regA.
func (a *Architecture) AttentionOps(l int) []op {
	ops := []op{{Kind: opProjQKV, In: regX, Proj: projParams{Bias: a.QKVBias}}}
	if a.QKNorm {
		ops = append(ops, op{Kind: opQKNorm, QKNorm: qkNormParams{Form: qkRMS, Whole: a.QKNormWhole, Eps: a.NormEps, AddOne: a.RMSAddOne}})
	}
	if !a.LearnedPosEmbed && !a.IsNoPELayer(l) && len(a.RoPEInvFreq(l)) > 0 { // an empty table rotates nothing (Nemotron-H)
		ops = append(ops, op{Kind: opRope, Rope: ropeParams{
			Kernel: ropeAtKernel, InvFreq: a.RoPEInvFreq(l), Mscale: a.RoPEMscale(l), Interleave: a.RoPEInterleave,
			MRopeSection: a.MRopeSection, MRopeInterleaved: a.MRopeInterleaved,
		}})
	}
	if a.AttnTempBeta != 0 {
		ops = append(ops, op{Kind: opAttnTemp, Temp: attnTempParams{Beta: a.AttnTempBeta, OrigMaxPos: a.AttnTempOrigMaxPos}})
	}
	window := 0
	if !a.IsGlobalLayer(l) {
		window = a.SlidingWindow
	}
	ops = append(ops, op{Kind: opAttention, Attn: attentionParams{Window: window, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: a.AttnOrder()}})
	if a.hasAttnOutputGate() {
		ops = append(ops, op{Kind: opOutGate, In: regX, Gate: outGateParams{Sigmoid: a.AttnGate == GateSigmoid}})
	}
	return append(ops, op{Kind: opProjO, Out: regA, Proj: projParams{Bias: a.OutBias}})
}

// Recurrent reports whether a layer carries state mutated in place per token (a conv window, an SSM state, a linear-attention
// state), which no positional rewind restores: any recurrent operation in its layers. Speculative rollback and the hybrid image
// prefill read it before any cache exists.
func (a *Architecture) Recurrent() bool {
	for l := range a.NumLayers {
		for _, o := range a.LayerOps(l, false) {
			if o.Kind.Recurrent() {
				return true
			}
		}
	}
	return false
}

// GenericKindsOnly reports whether every layer of a is made of genericKinds, so the generic machinery around the layers applies:
// LoRA, and the dense layer pager (with the fit guard's retry that mirrors it).
func (a *Architecture) GenericKindsOnly() bool {
	for l := range a.NumLayers {
		for _, o := range a.LayerOps(l, false) {
			if !slices.Contains(genericKinds, o.Kind) {
				return false
			}
		}
	}
	return true
}

// AttnOrder is the attention kernel this family's CPU forward computes in.
func (a *Architecture) AttnOrder() attnOrder {
	switch {
	case a.Llama4 != nil:
		return attnQuery
	case a.MLA != nil:
		return attnMLA
	case a.LFM2 != nil, a.Granite != nil, a.Nemotron != nil, a.Qwen35 != nil:
		return attnQuery
	case a.Gemma4 != nil:
		return attnGemma4
	case a.Inkling2 != nil:
		return attnInkling2
	}
	return attnBatchedAcc64
}

// gemma4LayerOps is a Gemma 4 layer: sandwich norms around its attention; then, on a 26B MoE layer, the parallel dense ‖ MoE
// block, which returns the whole layer output (scalar included); on a dense layer the sandwiched GeGLU, the PLE branch (when the
// model has PLE) and the layer scalar.
func (a *Architecture) gemma4LayerOps(l int, moeLayer bool) []op {
	g := a.Gemma4
	norm := func(slot normSlot, in, out opReg) op {
		return op{Kind: opNorm, In: in, Out: out, Norm: normParams{Slot: slot, Form: a.NormForm(), Eps: a.NormEps, AddOne: a.RMSAddOne}}
	}
	global := a.IsGlobalLayer(l)
	attn := gemma4AttnParams{Eps: a.NormEps, AddOne: a.RMSAddOne, Mscale: 1, Scale: a.AttnScale, KVSrc: a.Gemma4KVSrcAt(l),
		OwnsKV: l < a.NumLayers-g.SharedKVLayers, InvFreq: archparts.Gemma4InvFreq(a.HeadDim, a.HeadDim, a.RoPELocalBase)}
	if global {
		attn.InvFreq = archparts.Gemma4InvFreq(g.GlobalHeadDim, g.GlobalRotaryDim, a.RoPEGlobalBase)
	} else {
		attn.Window = a.SlidingWindow
	}
	ops := []op{norm(slotPreAttn, regH, regX), {Kind: opGemma4Attn, In: regX, Out: regA, G4: attn}, norm(slotPostAttn, regA, regA), {Kind: opResidual}}
	ne := normEpsParams{Eps: a.NormEps, AddOne: a.RMSAddOne}
	if moeLayer {
		return append(ops, op{Kind: opGemma4MoE, Norms: ne})
	}
	ops = append(ops, norm(slotPreMLP, regH, regX), op{Kind: opFFN, In: regX, Out: regA, FFN: ffnParams{Form: ffnGemma4Dense, Act: a.Act}},
		norm(slotPostMLP, regA, regA), op{Kind: opResidual})
	if g.HiddenSizePerLayerInput > 0 {
		ops = append(ops, op{Kind: opPLE, Norms: ne})
	}
	return append(ops, op{Kind: opLayerScale})
}

// graniteLayerOps is a Granite-4.0-H layer: pre-norm, a Mamba-2 mixer or GQA attention, each sublayer's output scaled by
// residual_multiplier before its add, and a MoE FFN.
func (a *Architecture) graniteLayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	resid := op{Kind: opResidual, Res: residualParams{Scale: float64(a.Granite.ResidMul)}}
	ops := []op{input(slotPreAttn)}
	if a.IsMambaLayer(l) {
		ops = append(ops, op{Kind: opMamba2, In: regX, Out: regA, Mamba: mamba2OpParams{Eps: a.NormEps, NormGroups: 1}})
	} else {
		ops = append(ops, a.AttentionOps(l)...)
	}
	ops = append(ops, resid)
	return append(ops, input(slotPreMLP), ffn(regA), resid)
}

// nemotronLayerOps is a Nemotron-H layer: one operation per block, pre-norm with a residual add. A mamba block is Mamba-2 with
// per-group gated norms, an attention block is NoPE GQA through attendQuery, an mlp block is a non-gated relu² FFN, and a moe
// block is Nemotron 3 Nano's relu² MoE.
func (a *Architecture) nemotronLayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	np := a.Nemotron
	ops := []op{input(slotPreAttn)}
	switch np.BlockKind[l] {
	case archparts.NemoMamba:
		ops = append(ops, op{Kind: opMamba2, In: regX, Out: regA, Mamba: mamba2OpParams{Eps: a.NormEps, NormGroups: np.NGroups}})
	case archparts.NemoAttn:
		ops = append(ops, a.AttentionOps(l)...)
	case archparts.NemoMLP:
		f := ffn(regA)
		f.FFN = ffnParams{Form: ffnNonGated, Act: a.Act}
		ops = append(ops, f)
	case archparts.NemoMoE:
		f := ffn(regA)
		f.FFN = ffnParams{Form: ffnNemotronMoE, Act: a.Act, MoE: *a.MoE}
		ops = append(ops, f)
	}
	return append(ops, op{Kind: opResidual})
}

// Mamba2Shapes is the Mamba-2 geometry of a Granite or Nemotron-H layer, with gated-norm group count g.
func (a *Architecture) Mamba2Shapes(g int) archparts.Mamba2Params {
	switch {
	case a.Granite != nil:
		p := a.Granite
		return archparts.Mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	case a.Nemotron != nil:
		p := a.Nemotron
		return archparts.Mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	}
	return archparts.Mamba2Params{}
}

// llama4LayerOps is a Llama 4 layer: pre-norm, and attention that varies by layer. A RoPE layer rotates q and k (mlaRope's
// interleaved form, no mscale), then normalises them without weights, and attends within its chunk; a NoPE layer scales the
// query by the attention temperature instead. The FFN is dense or Llama 4's MoE (moe_layers).
func (a *Architecture) llama4LayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	lp := a.Llama4
	ops := []op{input(slotPreAttn), {Kind: opProjQKV, In: regX}}
	chunk := 0
	if lp.UseRope[l] {
		ops = append(ops, op{Kind: opRope, Rope: ropeParams{Kernel: ropeMLAKernel, InvFreq: a.RoPEInvFreq(l), Mscale: 1, Interleave: true}})
		if lp.UseQKNorm {
			ops = append(ops, op{Kind: opQKNorm, QKNorm: qkNormParams{Form: qkL2, Eps: a.NormEps}})
		}
		chunk = lp.ChunkSize
	} else if lp.AttnTemp {
		ops = append(ops, op{Kind: opAttnTemp, Temp: attnTempParams{Beta: lp.AttnScale, OrigMaxPos: lp.FloorScale, PosOffset: 1}})
	}
	ops = append(ops,
		op{Kind: opAttention, Attn: attentionParams{Chunk: chunk, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnQuery}},
		op{Kind: opProjO, Out: regA},
		op{Kind: opResidual}, input(slotPreMLP))
	f := ffn(regA)
	if lp.IsMoE[l] {
		f.FFN = ffnParams{Form: ffnLlama4MoE, Act: a.Act, MoE: *a.MoE}
	}
	return append(ops, f, op{Kind: opResidual})
}

// inkling2LayerOps is one Inkling 2 layer visit: a single opInkling2 on the residual, reading the visit's physical layer,
// with the routing on its FFN.
func (a *Architecture) inkling2LayerOps(l int) []op {
	p := a.Inkling2
	v := p.Visits()[l]
	swa := p.SWA[v.Phys]
	ink := program.Inkling2Params{Phys: v.Phys, Pass: v.Pass, Boundary: v.Boundary, SWA: swa, Eps: a.NormEps,
		QKClip: p.QKClip, RelClip: p.RelClip, LogFloor: p.LogFloor, LogAlpha: p.LogAlpha, ConvK: p.ConvK,
		Extent: p.RelExtent, SituBeta: p.SituBeta, SituUp: p.SituLinearBeta}
	if swa {
		ink.Window, ink.Extent = a.SlidingWindow, a.SlidingWindow
	}
	return []op{{Kind: opInkling2, FFN: ffnParams{Form: ffnMoE, MoE: *a.MoE}, Ink: ink}}
}
