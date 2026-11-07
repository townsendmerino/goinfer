package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/program"

// layerShapeOf is layer l's widths for its operations' buffers (program.Buffers), read where the walker's implementations of those
// operations read them (Model.walkToken).
func layerShapeOf(a *Architecture, l int, ops []op) program.LayerShape {
	s := program.LayerShape{KVHeads: a.kvHeadsAt(l), HeadDim: a.headDimAt(l)}
	for _, o := range ops {
		switch o.Kind {
		case opMLA:
			s.MLALatent = a.mla.KVLoRARank + a.mla.QKRopeHeadDim
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
		}
	}
	return s
}

// layerBuffers is what layer l's operations declare beyond their weights. The FFN's form declares none, so the layer is read without
// its experts, and nothing here builds a whole program (a fit probe on a backend with no operation table builds none).
func layerBuffers(a *Architecture, l int) []program.Buf {
	ops := a.layerOps(l, false)
	return program.LayerBuffers(ops, layerShapeOf(a, l, ops))
}
