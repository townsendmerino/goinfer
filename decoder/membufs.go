package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/program"

// layerShapeOf is layer l's widths for its operations' buffers (program.Buffers), read where the walker's implementations of those
// operations read them (Model.walkToken).
func layerShapeOf(a *Architecture, l int, ops []op) program.LayerShape {
	s := program.LayerShape{KVHeads: a.kvHeadsAt(l), HeadDim: a.headDimAt(l), Hidden: a.HiddenDim, Heads: a.headsAt(l), FFN: a.ffnAt(l)}
	for _, o := range ops {
		switch o.Kind {
		case opMLA:
			p := a.mla
			s.MLALatent = p.KVLoRARank + p.QKRopeHeadDim
			s.MLA.QLoRA, s.MLA.Rank, s.MLA.QKHead, s.MLA.VHead = p.QLoRARank, p.KVLoRARank, p.QKNopeHeadDim+p.QKRopeHeadDim, p.VHeadDim
		case opDeltaNet:
			p, d := a.qwen35, &s.DeltaNet
			d.KeyHeads, d.KeyHeadDim, d.ValueHeads, d.ValueHeadDim, d.ConvKernel = p.NumKeyHeads, p.KeyHeadDim, p.NumValueHeads, p.ValueHeadDim, p.ConvKernel
		case opMamba2:
			p, d := a.mamba2Shapes(o.Mamba.NormGroups), &s.Mamba2
			d.Heads, d.HeadDim, d.DState, d.ConvDim, d.ConvKernel = p.NHeads, p.HeadDim, p.DState, p.mixer().ConvDim(), p.DConv
		case opKDA:
			p, d := a.kda, &s.KDA
			d.Heads, d.HeadDim, d.ConvKernel = p.NumHeads, p.HeadDim, p.ConvKernel
		case opShortConv:
			s.ShortConv.Dim, s.ShortConv.LCache = a.lfm2.ConvDim, a.lfm2.ConvLCache
		case opPLE:
			s.PerLayerEmbed = a.gemma4.HiddenSizePerLayerInput
		case opGemma4MoE:
			s.Gemma4MoE.Experts, s.Gemma4MoE.TopK, s.Gemma4MoE.ExpertFFN = a.MoE.NumExperts, a.MoE.TopK, a.MoE.IntermediateDim
		}
	}
	return s
}

// LayerBuffers is what layer l of the loaded model declares beyond its weights (program.Buffers): the list a backend sizes its KV pools
// from and the plan prices. It is read once per model where the model keeps a fit memo; a caller must not modify it.
func (m *Model) LayerBuffers(l int) []program.Buf {
	a := m.w.arch
	memo := m.fitOf
	if memo == nil {
		return layerBuffers(a, l)
	}
	memo.mu.Lock()
	defer memo.mu.Unlock()
	if memo.bufs == nil {
		memo.bufs = make([][]program.Buf, a.NumLayers)
		for i := range a.NumLayers {
			memo.bufs[i] = layerBuffers(a, i)
		}
	}
	return memo.bufs[l]
}

// layerBuffers is what layer l's operations declare beyond their weights. The FFN's form declares none, so the layer is read without
// its experts, and nothing here builds a whole program (a fit probe on a backend with no operation table builds none).
func layerBuffers(a *Architecture, l int) []program.Buf {
	ops := a.layerOps(l, false)
	return program.LayerBuffers(ops, layerShapeOf(a, l, ops))
}
