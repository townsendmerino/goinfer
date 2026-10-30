package decoder

import "slices"

// mlaOpParamsAt is layer l's opMLA parameters.
func (a *Architecture) mlaOpParamsAt(l int) mlaOpParams {
	return mlaOpParams{InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), Interleave: a.mla.ropeInterleave, Eps: a.NormEps,
		Scale: a.AttnScale, HeadWise: a.mla.GateGranularity == "head_wise"}
}

// buildProgram builds a's program. experts reports whether layer l carries MoE experts: a MoE family's leading dense layers
// (first_k_dense_replace) do not, and only the weights say so.
func buildProgram(a *Architecture, experts func(l int) bool) *layerProgram {
	p := &layerProgram{
		Embed: embedParams{Scale: a.embedScale(), LearnedPos: a.LearnedPosEmbed, PLE: a.gemma4 != nil && a.gemma4.HiddenSizePerLayerInput > 0},
		Head: headParams{
			Norm: normParams{Slot: slotFinal, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true},
			Tied: a.TiedLMHead, Softcap: a.FinalLogitSoftcap, LogitScale: a.LogitScale,
		},
	}
	p.Layers = make([][]op, a.NumLayers)
	for l := range a.NumLayers {
		p.Layers[l] = a.layerOps(l, experts != nil && experts(l))
	}
	return p
}

// embedScale is the factor the embedding stage multiplies by: Granite's embedding_multiplier (1 under GOINFER_SSM_NOMUL), else
// EmbedScale.
func (a *Architecture) embedScale() float64 {
	if a.granite != nil {
		if ssmNoMul {
			return 1
		}
		return float64(a.granite.EmbMul)
	}
	return a.EmbedScale
}

func (a *Architecture) normForm() normForm {
	if a.Norm == NormLayer {
		return normLayer
	}
	return normRMS
}

// layerOps is one generic layer: the placement decides the order of norms, sublayers and residual adds.
func (a *Architecture) layerOps(l int, hasExperts bool) []op {
	placement := a.normPlacementAt(l)
	postOnly := placement == NormPostOnly
	postNorm := placement == NormSandwich4 || postOnly
	input := func(slot normSlot) op {
		if postOnly {
			return op{Kind: opNorm, In: regH, Out: regX, Norm: normParams{Slot: slot, Form: normCopy}}
		}
		return op{Kind: opNorm, In: regH, Out: regX, Norm: normParams{Slot: slot, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true}}
	}
	post := func(slot normSlot) op {
		return op{Kind: opNorm, In: regA, Out: regA, Norm: normParams{Slot: slot, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne}}
	}
	ffn := func(out opReg) op {
		f := ffnParams{Form: ffnGated, Act: a.Act}
		switch {
		case a.gptoss != nil:
			f.Form, f.MoE, f.Alpha, f.Limit = ffnGptOss, *a.MoE, a.gptoss.SwigluAlpha, a.gptoss.SwigluLimit
		case a.MoE != nil && hasExperts:
			f.Form, f.MoE = ffnMoE, *a.MoE
		case a.NonGatedMLP:
			f.Form = ffnNonGated
		}
		return op{Kind: opFFN, In: regX, Out: out, FFN: f}
	}

	if a.llama4 != nil {
		return a.llama4LayerOps(l, input, ffn)
	}
	if a.gemma4 != nil {
		return a.gemma4LayerOps(l, hasExperts)
	}
	if a.granite != nil {
		return a.graniteLayerOps(l, input, ffn)
	}
	if a.nemotron != nil {
		return a.nemotronLayerOps(l, input, ffn)
	}
	if a.lfm2 != nil && a.isConvLayer(l) {
		return []op{input(slotPreAttn), {Kind: opShortConv, In: regX, Out: regA}, {Kind: opResidual}, input(slotPreMLP), ffn(regA), {Kind: opResidual}}
	}
	if a.mla != nil {
		mixer := op{Kind: opMLA, In: regX, Out: regA, MLA: a.mlaOpParamsAt(l)}
		if a.kda != nil && a.isLinearLayer(l) { // Bailing Hybrid: KDA on the linear layers, MLA on the rest
			mixer = op{Kind: opKDA, In: regX, Out: regA, KDA: kdaOpParams{Eps: a.NormEps, LowerBound: a.kda.LowerBound}}
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
	g := a.qwen35
	switch {
	case g == nil:
		return a.attentionOps(l)
	case a.isLinearLayer(l):
		return []op{{Kind: opDeltaNet, In: regX, Out: regA, Delta: deltaNetOpParams{Eps: a.NormEps, ONormEps: g.ONormEps, NegEigval: g.NegEigval}}}
	case g.PlainFullAttn:
		ops := a.attentionOps(l)
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
		Rope: ropeParams{Kernel: ropeAtKernel, InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), MRopeSection: a.MRopeSection, MRopeInterleaved: a.MRopeInterleaved},
		Attn: attentionParams{Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnQuery},
	}}}
}

// attentionOps is the generic attention sublayer, regX → regA.
func (a *Architecture) attentionOps(l int) []op {
	ops := []op{{Kind: opProjQKV, In: regX, Proj: projParams{Bias: a.QKVBias}}}
	if a.QKNorm {
		ops = append(ops, op{Kind: opQKNorm, QKNorm: qkNormParams{Form: qkRMS, Whole: a.QKNormWhole, Eps: a.NormEps, AddOne: a.RMSAddOne}})
	}
	if !a.LearnedPosEmbed && !a.isNoPELayer(l) && len(a.ropeInvFreq(l)) > 0 { // an empty table rotates nothing (Nemotron-H)
		ops = append(ops, op{Kind: opRope, Rope: ropeParams{
			Kernel: ropeAtKernel, InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), Interleave: a.ropeInterleave,
			MRopeSection: a.MRopeSection, MRopeInterleaved: a.MRopeInterleaved,
		}})
	}
	if a.AttnTempBeta != 0 {
		ops = append(ops, op{Kind: opAttnTemp, Temp: attnTempParams{Beta: a.AttnTempBeta, OrigMaxPos: a.AttnTempOrigMaxPos}})
	}
	window := 0
	if !a.isGlobalLayer(l) {
		window = a.SlidingWindow
	}
	ops = append(ops, op{Kind: opAttention, Attn: attentionParams{Window: window, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: a.attnOrder()}})
	if a.hasAttnOutputGate() {
		ops = append(ops, op{Kind: opOutGate, In: regX, Gate: outGateParams{Sigmoid: a.AttnGate == GateSigmoid}})
	}
	return append(ops, op{Kind: opProjO, Out: regA, Proj: projParams{Bias: a.OutBias}})
}

// recurrent reports whether a layer carries state mutated in place per token (a conv window, an SSM state, a linear-attention
// state), which no positional rewind restores: any recurrent operation in its layers. Speculative rollback and the hybrid image
// prefill read it before any cache exists.
func (a *Architecture) recurrent() bool {
	for l := range a.NumLayers {
		for _, o := range a.layerOps(l, false) {
			if o.Kind.Recurrent() {
				return true
			}
		}
	}
	return false
}

// genericKindsOnly reports whether every layer of a is made of genericKinds, so the generic machinery around the layers applies:
// LoRA, and the dense layer pager (with the fit guard's retry that mirrors it).
func (a *Architecture) genericKindsOnly() bool {
	for l := range a.NumLayers {
		for _, o := range a.layerOps(l, false) {
			if !slices.Contains(genericKinds, o.Kind) {
				return false
			}
		}
	}
	return true
}

// attnOrder is the attention kernel this family's CPU forward computes in.
func (a *Architecture) attnOrder() attnOrder {
	switch {
	case a.llama4 != nil:
		return attnQuery
	case a.mla != nil:
		return attnMLA
	case a.lfm2 != nil, a.granite != nil, a.nemotron != nil, a.qwen35 != nil:
		return attnQuery
	case a.gemma4 != nil:
		return attnGemma4
	}
	return attnBatchedAcc64
}

// gemma4LayerOps is a Gemma 4 layer: sandwich norms around its attention; then, on a 26B MoE layer, the parallel dense ‖ MoE
// block, which returns the whole layer output (scalar included); on a dense layer the sandwiched GeGLU, the PLE branch (when the
// model has PLE) and the layer scalar.
func (a *Architecture) gemma4LayerOps(l int, moeLayer bool) []op {
	g := a.gemma4
	norm := func(slot normSlot, in, out opReg) op {
		return op{Kind: opNorm, In: in, Out: out, Norm: normParams{Slot: slot, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne}}
	}
	global := a.isGlobalLayer(l)
	attn := gemma4AttnParams{Eps: a.NormEps, AddOne: a.RMSAddOne, Mscale: 1, Scale: a.AttnScale, KVSrc: a.gemma4KVSrcAt(l),
		OwnsKV: l < a.NumLayers-g.SharedKVLayers, InvFreq: gemma4InvFreq(a.HeadDim, a.HeadDim, a.RoPELocalBase)}
	if global {
		attn.InvFreq = gemma4InvFreq(g.GlobalHeadDim, g.GlobalRotaryDim, a.RoPEGlobalBase)
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
// residual_multiplier before its add, and a MoE FFN. The isolation seams GOINFER_SSM_NOMUL (multipliers 1) and
// GOINFER_SSM_SKIPFFN (no FFN) shape the program, as they shaped the loop it replaced.
func (a *Architecture) graniteLayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	resid := op{Kind: opResidual, Res: residualParams{Scale: float64(a.granite.ResidMul)}}
	if ssmNoMul {
		resid.Res.Scale = 1
	}
	ops := []op{input(slotPreAttn)}
	if a.isMambaLayer(l) {
		ops = append(ops, op{Kind: opMamba2, In: regX, Out: regA, Mamba: mamba2OpParams{Eps: a.NormEps, NormGroups: 1}})
	} else {
		ops = append(ops, a.attentionOps(l)...)
	}
	ops = append(ops, resid)
	if ssmSkipFFN {
		return ops
	}
	return append(ops, input(slotPreMLP), ffn(regA), resid)
}

// nemotronLayerOps is a Nemotron-H layer: one operation per block, pre-norm with a residual add. A mamba block is Mamba-2 with
// per-group gated norms, an attention block is NoPE GQA through attendQuery, an mlp block is a non-gated relu² FFN, and a moe
// block is Nemotron 3 Nano's relu² MoE.
func (a *Architecture) nemotronLayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	np := a.nemotron
	ops := []op{input(slotPreAttn)}
	switch np.blockKind[l] {
	case nemoMamba:
		ops = append(ops, op{Kind: opMamba2, In: regX, Out: regA, Mamba: mamba2OpParams{Eps: a.NormEps, NormGroups: np.NGroups}})
	case nemoAttn:
		ops = append(ops, a.attentionOps(l)...)
	case nemoMLP:
		f := ffn(regA)
		f.FFN = ffnParams{Form: ffnNonGated, Act: a.Act}
		ops = append(ops, f)
	case nemoMoE:
		f := ffn(regA)
		f.FFN = ffnParams{Form: ffnNemotronMoE, Act: a.Act, MoE: *a.MoE}
		ops = append(ops, f)
	}
	return append(ops, op{Kind: opResidual})
}

// mamba2Shapes is the Mamba-2 geometry of a Granite or Nemotron-H layer, with gated-norm group count g.
func (a *Architecture) mamba2Shapes(g int) mamba2Params {
	switch {
	case a.granite != nil:
		p := a.granite
		return mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	case a.nemotron != nil:
		p := a.nemotron
		return mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	}
	return mamba2Params{}
}

// llama4LayerOps is a Llama 4 layer: pre-norm, and attention that varies by layer. A RoPE layer rotates q and k (mlaRope's
// interleaved form, no mscale), then normalises them without weights, and attends within its chunk; a NoPE layer scales the
// query by the attention temperature instead. The FFN is dense or Llama 4's MoE (moe_layers).
func (a *Architecture) llama4LayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	lp := a.llama4
	ops := []op{input(slotPreAttn), {Kind: opProjQKV, In: regX}}
	chunk := 0
	if lp.useRope[l] {
		ops = append(ops, op{Kind: opRope, Rope: ropeParams{Kernel: ropeMLAKernel, InvFreq: a.ropeInvFreq(l), Mscale: 1, Interleave: true}})
		if lp.useQKNorm {
			ops = append(ops, op{Kind: opQKNorm, QKNorm: qkNormParams{Form: qkL2, Eps: a.NormEps}})
		}
		chunk = lp.chunkSize
	} else if lp.attnTemp {
		ops = append(ops, op{Kind: opAttnTemp, Temp: attnTempParams{Beta: lp.attnScale, OrigMaxPos: lp.floorScale, PosOffset: 1}})
	}
	ops = append(ops,
		op{Kind: opAttention, Attn: attentionParams{Chunk: chunk, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnQuery}},
		op{Kind: opProjO, Out: regA},
		op{Kind: opResidual}, input(slotPreMLP))
	f := ffn(regA)
	if lp.isMoE[l] {
		f.FFN = ffnParams{Form: ffnLlama4MoE, Act: a.Act, MoE: *a.MoE}
	}
	return append(ops, f, op{Kind: opResidual})
}

// attnParamsFor is the attention parameters the loops that do not walk a program read from the Architecture: the cache's window
// on a local layer (the window the cache was built with), every key on a global one.
func (a *Architecture) attnParamsFor(cache *KVCache, global bool) attentionParams {
	p := attentionParams{Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnBatchedAcc64}
	if !global && cache.window > 0 {
		p.Window = cache.window
	}
	return p
}

// attnParamsQueryFor is attnParamsFor for attendQuery's callers, with Llama 4's chunk on its RoPE layers.
func (a *Architecture) attnParamsQueryFor(cache *KVCache, layer int, global bool) attentionParams {
	p := a.attnParamsFor(cache, global)
	p.Order = attnQuery
	if l4 := a.llama4; l4 != nil && l4.chunkSize > 0 && layer < len(l4.useRope) && l4.useRope[layer] {
		p.Chunk = l4.chunkSize
	}
	return p
}
