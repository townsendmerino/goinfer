package decoder

import (
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
)

// mlaProjInt8 reports whether a load at q holds MLA's q (or q-LoRA pair), kv-A and o projections as per-row int8. No production
// load does: every load keeps all six f32, because the DeepSeek family's real-model gates fail with any of the three int8
// (docs/measurements/mem11-int8-2026-10/mla.md). A GPU backend still makes its own int8 from the f32. The int8 form stays for a
// .giw that carries it and for the tests that arm it (mlaInt8ForTest), on an integer quant only.
func mlaProjInt8(q quantMode) bool {
	switch q {
	case quantInt8, quantInt8I8, quantInt4, quantInt4Mix, quantQ4K:
		return mlaInt8ForTest
	}
	return false
}

// mlaInt8ForTest arms the int8 MLA projections on an integer-quantized load: the int8 MLA gates and the per-projection search.
var mlaInt8ForTest bool

// mlaProjF32ForTest names single projections ("q", "kva", "o") an armed load keeps f32 while the rest go int8: the
// per-projection search (docs/measurements/mem11-int8-2026-10/mla.md).
var mlaProjF32ForTest map[string]bool

// mlaMatvec is one MLA projection of x: its int8 form q when the load holds one, else the f32 f [rows, cols].
func mlaMatvec(f []float32, q *linalg.WeightMat, rows, cols int, x []float32) []float32 {
	if q.Rows() > 0 {
		return matvecWM(nil, q, x)
	}
	return hybridattn.Matvec(f, rows, cols, x)
}

// MLALayerInt8 returns layer l's int8 MLA projections, each zero-valued where the load holds that projection f32: the q-LoRA
// pair or the direct q, kv-A and o. A backend uploads these codes as they are; they are what it would make from the f32.
func (m *Model) MLALayerInt8(l int) (qA, qB, q, kvA, o *linalg.WeightMat) {
	w := m.w.Layers[l].MLA
	return &w.QAQ, &w.QBQ, &w.QQ, &w.KVAQ, &w.OQ
}
