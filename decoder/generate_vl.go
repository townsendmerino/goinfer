package decoder

import "context"

// GenerateVL streams a continuation for a multimodal (vision-language) prompt.
// `ids` are the text token ids with a run of `imgLen` image-placeholder ids
// starting at `imgPos`; `features` ([imgLen*HiddenDim]) are the projected vision
// embeddings (the projector's output) that replace those placeholders' token
// embeddings. It prefills once through the bidirectional image-block mask
// (prefillLogitsVL) — image tokens attend mutually, text attends causally, and
// tokens decoded after the image attend causally too — then decodes up to
// maxTokens exactly like Generate (same Sampler, LogitProcessor, and stop rule).
//
// Stateless and CPU-only by design: a fresh cache, no warm-KV prefix reuse (a
// multimodal turn opts out of the session cache), no GPU residency. The caller
// owns the returned channel: range over it to consume tokens, then check
// Generation.Err for a terminal error.
func (m *Model) GenerateVL(ctx context.Context, ids []int, features []float32, imgPos, imgLen, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	cache := m.NewCache(len(ids) + maxTokens)
	go func() {
		defer close(out)
		// Prefill: text embeddings with the projected vision features spliced in at
		// the placeholder run, under the bidirectional image-block mask. The last
		// position's logits seed the decode.
		logits, err := m.prefillLogitsVL(ids, features, imgPos, imgLen, cache)
		if err != nil {
			g.err = err
			return
		}
		sampler := NewSampler(sp)
		sampler.Observe(ids...) // repetition penalties see the whole prompt
		var generated []int
		for range maxTokens {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			default:
			}
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			info, err := sampler.SampleWithInfo(logits)
			if err != nil {
				g.err = err
				return
			}
			next := info.ID
			if m.isStop(next, sp) {
				break
			}
			if sp.Logprobs {
				g.Logprobs = append(g.Logprobs, info)
			}
			out <- next
			generated = append(generated, next)
			// Decode steps after the image block are ordinary causal forwards (the
			// image-block mask only governs the image positions, which are now in KV).
			if logits, err = m.forward(next, cache); err != nil {
				g.err = err
				return
			}
		}
	}()
	return out, g
}
