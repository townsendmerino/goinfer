package cpuengine

import (
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
)

// mlaMatvec is one MLA projection of x: its int8 form q when the load holds one, else the f32 f [rows, cols].
func mlaMatvec(f []float32, q *linalg.WeightMat, rows, cols int, x []float32) []float32 {
	if q.Rows() > 0 {
		return MatvecWM(nil, q, x)
	}
	return hybridattn.Matvec(f, rows, cols, x)
}

// MLALayerInt8 returns layer l's int8 MLA projections, each zero-valued where the load holds that projection f32: the q-LoRA
// pair or the direct q, kv-A and o. A backend uploads these codes as they are; they are what it would make from the f32.
func (m *Engine) MLALayerInt8(l int) (qA, qB, q, kvA, o *linalg.WeightMat) {
	w := m.W.Layers[l].MLA
	return &w.QAQ, &w.QBQ, &w.QQ, &w.KVAQ, &w.OQ
}
