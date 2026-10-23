package decoder

import "github.com/townsendmerino/aikit/linalg"

// mlaProjInt8 reports whether a load at q holds MLA's q (or q-LoRA pair), kv-A and o projections as per-row int8: every integer
// quant does, and an f32, bf16 or f16 load keeps them f32 (MEM11-int8, docs/tasks/task-memory-campaign-2026-10.md). kv-B stays
// f32 at every quant: the backends slice it into W_UK and W_UV as f32, and the CPU's absorbed path reads it by row.
func mlaProjInt8(q quantMode) bool {
	switch q {
	case quantInt8, quantInt8I8, quantInt4, quantInt4Mix, quantQ4K:
		return !mlaKeepF32ForTest
	}
	return false
}

// mlaKeepF32ForTest keeps MLA's f32 projections on a quantized load: the f32 arm of MEM11-int8's fidelity bar.
var mlaKeepF32ForTest bool

// quantizeProjections replaces the q (or q-LoRA pair), kv-A and o projections with their int8 form: linalg.QuantizeInt8 of the
// f32, weight-only, the codes and scales every GPU backend makes from the f32 (B27's form for Mamba).
func (w *mlaWeights) quantizeProjections(arch *Architecture) {
	p := arch.mla
	H, hidden, qkHead := arch.NumHeads, arch.HiddenDim, p.QKNopeHeadDim+p.QKRopeHeadDim
	if p.QLoRARank > 0 {
		w.qAQ = linalg.QuantizeInt8(w.qAProj, p.QLoRARank, hidden, false)
		w.qBQ = linalg.QuantizeInt8(w.qBProj, H*qkHead, p.QLoRARank, false)
		w.qAProj, w.qBProj = nil, nil
	} else {
		w.qQ = linalg.QuantizeInt8(w.qProj, H*qkHead, hidden, false)
		w.qProj = nil
	}
	w.kvAQ = linalg.QuantizeInt8(w.kvAProj, p.KVLoRARank+p.QKRopeHeadDim, hidden, false)
	w.oQ = linalg.QuantizeInt8(w.oProj, hidden, H*p.VHeadDim, false)
	w.kvAProj, w.oProj = nil, nil
}

// mlaMatvec is one MLA projection of x: its int8 form q when the load holds one, else the f32 f [rows, cols].
func mlaMatvec(f []float32, q *linalg.WeightMat, rows, cols int, x []float32) []float32 {
	if q.Rows() > 0 {
		return matvecWM(nil, q, x)
	}
	return matvec(f, rows, cols, x)
}

// MLALayerInt8 returns layer l's int8 MLA projections, each zero-valued where the load holds that projection f32: the q-LoRA
// pair or the direct q, kv-A and o. A backend uploads these codes as they are; they are what it would make from the f32.
func (m *Model) MLALayerInt8(l int) (qA, qB, q, kvA, o *linalg.WeightMat) {
	w := m.w.Layers[l].mla
	return &w.qAQ, &w.qBQ, &w.qQ, &w.kvAQ, &w.oQ
}
