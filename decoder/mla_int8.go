package decoder

import "github.com/townsendmerino/aikit/linalg"

// mlaProjInt8 reports whether a load at q holds MLA's q (or q-LoRA pair), kv-A and o projections as per-row int8. No production
// load does: every load keeps all six f32, because the DeepSeek family's real-model gates fail with any of the three int8 (D14,
// docs/measurements/mem11-int8-2026-10/mla.md). A GPU backend still makes its own int8 from the f32. The int8 form stays for a
// .giw that carries it and for the tests that arm it (mlaInt8ForTest), on an integer quant only.
func mlaProjInt8(q quantMode) bool {
	switch q {
	case quantInt8, quantInt8I8, quantInt4, quantInt4Mix, quantQ4K:
		return mlaInt8ForTest
	}
	return false
}

// mlaInt8ForTest arms the int8 MLA projections on an integer-quantized load: MEM11-int8's gates and D14's search.
var mlaInt8ForTest bool

// mlaProjF32ForTest names single projections ("q", "kva", "o") an armed load keeps f32 while the rest go int8: D14's
// per-projection search (docs/measurements/mem11-int8-2026-10/mla.md).
var mlaProjF32ForTest map[string]bool

// quantizeProjections replaces the q (or q-LoRA pair), kv-A and o projections with their int8 form: linalg.QuantizeInt8 of the
// f32, weight-only, the codes and scales every GPU backend makes from the f32 (B27's form for Mamba).
func (w *mlaWeights) quantizeProjections(arch *Architecture) {
	p := arch.mla
	H, hidden, qkHead := arch.NumHeads, arch.HiddenDim, p.QKNopeHeadDim+p.QKRopeHeadDim
	if mlaProjF32ForTest["q"] {
		// q stays f32.
	} else if p.QLoRARank > 0 {
		w.qAQ = linalg.QuantizeInt8(w.qAProj, p.QLoRARank, hidden, false)
		w.qBQ = linalg.QuantizeInt8(w.qBProj, H*qkHead, p.QLoRARank, false)
		w.qAProj, w.qBProj = nil, nil
	} else {
		w.qQ = linalg.QuantizeInt8(w.qProj, H*qkHead, hidden, false)
		w.qProj = nil
	}
	if !mlaProjF32ForTest["kva"] {
		w.kvAQ = linalg.QuantizeInt8(w.kvAProj, p.KVLoRARank+p.QKRopeHeadDim, hidden, false)
		w.kvAProj = nil
	}
	if !mlaProjF32ForTest["o"] {
		w.oQ = linalg.QuantizeInt8(w.oProj, hidden, H*p.VHeadDim, false)
		w.oProj = nil
	}
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
