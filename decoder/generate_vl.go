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
	// Not generateInto's resident path: this entry point may drive the resident KV on its
	// own schedule, so the recorded id list stops being true the moment it does. Forget it —
	// the next turn cold-prefills, which is slow, not wrong (resident_reuse.go).
	m.residentForgetIDs()
	out := make(chan int)
	g := &Generation{}
	cache := m.NewCache(len(ids) + maxTokens)
	go func() {
		defer close(out)
		// Prefill: text embeddings with the projected vision features spliced in at
		// the placeholder run, under the bidirectional image-block mask. The last
		// position's logits seed the decode.
		logits, err := m.prefillLogitsVL(ctx, ids, features, imgPos, imgLen, cache)
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
			// Select on ctx.Done so a consumer that stops ranging can't wedge this
			// goroutine forever on a bare send (holding the KV cache) — M8.
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			case out <- next:
			}
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

// GenerateQwenVL streams a continuation for a Qwen2.5-VL multimodal prompt. Like
// GenerateVL but for the Qwen family: the merged vision features (the ViT+merger
// output, [imgLen*HiddenDim]) replace the <image> run at [imgPos,imgPos+imgLen),
// and rotary positions are m-RoPE — computed from the image grid(s) (gridTHW, t/h/w
// in patch units; merge = spatial_merge_size; imageToken = the placeholder id).
// Image tokens attend CAUSALLY (no bidirectional block — Qwen's bidirectionality is
// in the ViT). Decode past the prompt resumes scalar positions at the block max + 1
// (handled by the cache's m-RoPE delta). Stateless + CPU-only, like GenerateVL.
func (m *Model) GenerateQwenVL(ctx context.Context, ids []int, features []float32, imgPos, imgLen int, gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	// Not generateInto's resident path: this entry point may drive the resident KV on its
	// own schedule, so the recorded id list stops being true the moment it does. Forget it —
	// the next turn cold-prefills, which is slow, not wrong (resident_reuse.go).
	m.residentForgetIDs()
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		mropePos, err := mropePositions(ids, imageToken, gridTHW, merge)
		if err != nil {
			g.err = err
			return
		}
		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsQwenVL(ctx, ids, features, imgPos, imgLen, mropePos, cache)
		if err != nil {
			g.err = err
			return
		}
		sampler := NewSampler(sp)
		sampler.Observe(ids...)
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
			// Select on ctx.Done so a consumer that stops ranging can't wedge this
			// goroutine forever on a bare send (holding the KV cache) — M8.
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			case out <- next:
			}
			generated = append(generated, next)
			if logits, err = m.forward(next, cache); err != nil { // decode m-RoPE via cache.mropeDelta
				g.err = err
				return
			}
		}
	}()
	return out, g
}
