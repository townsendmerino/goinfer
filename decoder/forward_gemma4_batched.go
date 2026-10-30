package decoder

import (
	"context"
	"fmt"
)

// runLayersGemma4FromEmbedN runs K pre-embedded rows (h, [K*HiddenDim], image positions already spliced with raw projected
// features by the caller) through the batched walker as one prefill pass, and returns the last row after the head's norm (the
// caller applies the LM head with logitsFromNormed). One pass is the only way to let a query inside an image or audio block see
// later block positions (gemma4BatchRange), which a sequential KV-append forward cannot express.
//
// spans are absolute positions in a prompt that starts at position 0; a pass may start after earlier passes of the same prefill
// (prefillLogitsGemma4VLBidirectionalSpans), and a row's image membership is read at its absolute position. ids[row] is the row's
// real token id, used only for PLE's token-identity lookup (an image row takes arch.gemma4.PadTokenID, as the per-token path does).
func (m *Model) runLayersGemma4FromEmbedN(reqCtx context.Context, h []float32, ids []int, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	hidden := m.w.arch.HiddenDim
	K := len(ids)
	if len(h) != K*hidden {
		return nil, fmt.Errorf("decoder: runLayersGemma4FromEmbedN: h len %d, want %d (%d rows x %d hidden)", len(h), K*hidden, K, hidden)
	}
	out, err := m.walkBatch(reqCtx, m.program(), h, cache, false, K, nil, &batchRows{ids: ids, spans: spans})
	if err != nil {
		return nil, err
	}
	return out[(K-1)*hidden:], nil
}
