package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
)

// QuantizeProjections replaces the q (or q-LoRA pair), kv-A and o projections with their int8 form: linalg.QuantizeInt8 of the
// f32, weight-only, the codes and scales every GPU backend makes from the f32 (the form the Mamba projections take).
func (w *MLAWeights) QuantizeProjections(p *MLAParams, H, hidden int, skipF32 map[string]bool) {
	qkHead := p.QKNopeHeadDim + p.QKRopeHeadDim
	if skipF32["q"] {
		// q stays f32.
	} else if p.QLoRARank > 0 {
		w.QAQ = linalg.QuantizeInt8(w.QAProj, p.QLoRARank, hidden, false)
		w.QBQ = linalg.QuantizeInt8(w.QBProj, H*qkHead, p.QLoRARank, false)
		w.QAProj, w.QBProj = nil, nil
	} else {
		w.QQ = linalg.QuantizeInt8(w.QProj, H*qkHead, hidden, false)
		w.QProj = nil
	}
	if !skipF32["kva"] {
		w.KVAQ = linalg.QuantizeInt8(w.KVAProj, p.KVLoRARank+p.QKRopeHeadDim, hidden, false)
		w.KVAProj = nil
	}
	if !skipF32["o"] {
		w.OQ = linalg.QuantizeInt8(w.OProj, hidden, H*p.VHeadDim, false)
		w.OProj = nil
	}
}
