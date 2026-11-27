package decoder

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
)

func (m *Model) prefillLogitsAudio(ctx context.Context, ids []int, feats []float32, pos, n int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillLogitsAudio(ctx, &e, ids, feats, pos, n, unwrapCache(cache))
}

// GenerateAudio streams a continuation for a prompt whose ids hold a run of audioLen placeholders at audioPos, replaced by
// the embeddings features returns.
func (m *Model) GenerateAudio(ctx context.Context, ids []int, audioPos, audioLen int, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	m.runGen(ctx, out, g, nil, func(ctx context.Context) {
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
	})
	return out, g
}
