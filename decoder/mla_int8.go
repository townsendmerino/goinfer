package decoder

import "github.com/townsendmerino/aikit/linalg"

// mlaInt8ForTest arms int8 MLA projections on an integer-quantized load.
var mlaInt8ForTest bool

// mlaProjF32ForTest names single projections ("q", "kva", "o") kept f32 while the rest go int8.
var mlaProjF32ForTest map[string]bool

// MLALayerInt8 returns layer l's int8 MLA projections, each zero-valued where the load holds that projection f32: the q-LoRA
// pair or the direct q, kv-A and o. A backend uploads these codes as they are; they are what it would make from the f32.
func (m *Model) MLALayerInt8(l int) (qA, qB, q, kvA, o *linalg.WeightMat) {
	w := m.w.Layers[l].MLA
	return &w.QAQ, &w.QBQ, &w.QQ, &w.KVAQ, &w.OQ
}
