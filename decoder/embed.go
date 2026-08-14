package decoder

import "fmt"

// Decoder-as-embedder seam (docs/completed/task-decoder-as-embedder.md).
//
// qwen3-embedding and embeddinggemma are causal decoders used as EMBEDDERS: the sentence-
// transformers stack around them is just `Transformer → Pooling → Normalize`. The only thing the
// decoder itself must expose for that is the transformer's output — the final hidden state — which
// until now was unreachable: forward() runs the stack and immediately consumes the hidden state
// through the LM head (logitsFromHidden normalizes it IN PLACE, then projects to vocab), and
// ForwardCapture hands back per-LAYER residuals, which are pre-final-norm and so are NOT what
// pooling wants.

// HiddenLast runs ids through the layer stack causally in one fresh KV cache and returns the final
// hidden state of the LAST token, AFTER the model's final norm — exactly HF's
// `last_hidden_state[:, -1, :]`, which is what sentence-transformers' last-token pooling reads.
//
// It deliberately stops before the LM head: an embedder never needs the logits, and the head is the
// single most expensive matmul in a forward (vocab×hidden — for a tied 151k-row Qwen3 head, far
// more work than the rest of the token combined).
//
// Padding note (the classic last-token-pooling silent-wrong): this takes ONE sequence and pools the
// last element of it, so "last token" is always the last REAL token. There is no padded batch here
// in which the last slot could be a pad — callers that want a batch must call this per sequence.
//
// The returned slice is a fresh copy the caller owns: the hidden state aliases the per-token decode
// scratch, which the next token would overwrite.
//
// Generic decode path only. The families with their own runLayers (gemma4/qwen35/granite/nemotron/
// mla/llama4) return an error rather than a silently wrong vector — the same contract, and the same
// guard, as ForwardCapture.
func (m *Model) HiddenLast(ids []int) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("decoder.HiddenLast: empty token sequence")
	}
	a := m.w.arch
	if a.gemma4 != nil || a.qwen35 != nil || a.granite != nil || a.nemotron != nil || a.mla != nil || a.llama4 != nil {
		return nil, fmt.Errorf("decoder.HiddenLast: hidden-state seam not wired for arch %q (own runLayers)", a.Name)
	}
	for i, id := range ids {
		if id < 0 || id >= a.VocabSize {
			return nil, fmt.Errorf("decoder.HiddenLast: token %d at index %d out of vocab [0,%d)", id, i, a.VocabSize)
		}
	}
	cache := m.NewCache(len(ids))
	var h []float32
	for _, id := range ids {
		var err error
		if h, err = m.runLayers(id, cache); err != nil {
			return nil, err
		}
	}
	// Copy before normalizing: h aliases the decode scratch and normalize mutates in place (the
	// same in-place norm logitsFromHidden does on its way to the head).
	out := append([]float32(nil), h[:a.HiddenDim]...)
	normalize(a, out, m.w.FinalNorm, m.w.FinalNormBias, a.HiddenDim)
	return out, nil
}
