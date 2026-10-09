package decoder

import (
	"context"
	"fmt"
)

// Soft-token audio prefill and generation for Qwen3-ASR (docs/tasks/task-multimodal-support-2026-10.md, S14.3). The audio encoder's projected embeddings replace the input
// embeddings of the run of <|audio_pad|> placeholders; everything else is an ordinary causal text prefill and decode. That is the whole difference from the image paths: no
// bidirectional block (GenerateVL's Gemma 3 mask), no m-RoPE grid (GenerateQwenVL), the positions are 0..n-1 as in text. CPU only: a GPU-resident decoder is not claimed, so a
// model loaded on a GPU backend still decodes this turn on the CPU weights.

// prefillLogitsAudio prefills ids with feats ([n][HiddenDim], the projected audio embeddings) written over the embeddings of ids[pos : pos+n], and returns the last position's logits.
func (m *Model) prefillLogitsAudio(ctx context.Context, ids []int, feats []float32, pos, n int, cache *KVCache) ([]float32, error) {
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: audio prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if pos < 0 || n <= 0 || pos+n > len(ids) {
		return nil, fmt.Errorf("decoder: audio run [%d,%d) out of range for %d tokens", pos, pos+n, len(ids))
	}
	if len(feats) != n*hidden {
		return nil, fmt.Errorf("decoder: audio features len %d, want %d (%d tokens x %d)", len(feats), n*hidden, n, hidden)
	}
	h := m.embedN(ids)
	copy(h[pos*hidden:(pos+n)*hidden], feats) // raw projector output: no embed scale, as transformers' masked_scatter into inputs_embeds
	hN, err := m.runLayersFromEmbedN(ctx, h, cache, m.cpuFastAttention())
	if err != nil {
		return nil, err
	}
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

// GenerateAudio streams a continuation for a prompt whose ids hold a run of audioLen placeholders at audioPos, replaced by the embeddings features returns. features is invoked once.
// The prompt is prefilled causally (prefillLogitsAudio) and decoded on the CPU like Generate, with the same Sampler, LogitProcessor and stop rule.
func (m *Model) GenerateAudio(ctx context.Context, ids []int, audioPos, audioLen int, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsAudio(ctx, ids, feats, audioPos, audioLen, cache)
		if err != nil {
			g.err = err
			return
		}
		g.Budget = maxTokens
		sampler := NewSampler(sp)
		sampler.Observe(ids...)
		m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) { return m.forward(next, cache) })
	}()
	return out, g
}
