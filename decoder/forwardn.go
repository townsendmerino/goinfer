package decoder

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
)

// ExactPrefill reports whether this model runs its prefill with bit-exact f64-accumulated attention.
func (m *Model) ExactPrefill() bool { return m.opts.ExactPrefill }

func (m *Model) canBatchN(K int) bool {
	e := m.cpuEngine()
	return e.CanBatchN(K)
}

func (m *Model) hasRecurrentState() bool {
	e := m.cpuEngine()
	return e.HasRecurrentState()
}

// specRollbackSafe reports whether speculative decode's rollback (KVCache.TruncateTo after a partial accept) correctly restores
// this model's state.
func (m *Model) specRollbackSafe() bool { return m.specRollbackSafeVia(nil, 0) }

// specRollbackSafeVia is specRollbackSafe for a loop that verifies through fold, a resident's recurrent fold path, in passes of up
// to width rows (BlockSpec; nil for a loop that rolls back with TruncateTo or ForwardN).
func (m *Model) specRollbackSafeVia(fold ResidentRuntimeSubsystem, width int) bool {
	a := m.w.Arch
	if m.hasRecurrentState() && (fold == nil || width < 1 || width > fold.RecurrentSpecRows() || RecurrentSpecArenaBytes(a, 1) == 0) {
		return false
	}
	if a.SlidingWindow > 0 {
		return false
	}
	return true
}

func (m *Model) embedN(ids []int) []float32 {
	e := m.cpuEngine()
	return e.EmbedN(ids)
}

func (m *Model) prefillLogits(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.PrefillLogits(ctx, prompt, unwrapCache(cache))
}

func (m *Model) qwen35BatchN(K int, cache *KVCache) bool {
	e := m.cpuEngine()
	return e.Qwen35BatchN(K, unwrapCache(cache))
}

func (m *Model) runLayersQwen35N(reqCtx context.Context, h []float32, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersQwen35N(reqCtx, h, unwrapCache(cache))
}

func (m *Model) prefillLogitsVL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageEmbeds, cache, false)
}

func (m *Model) prefillLogitsVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillLogitsVLSpans(ctx, &e, ids, spans, imageEmbeds, unwrapCache(cache), causal)
}

func (m *Model) prefillHiddenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillHiddenVLSpans(ctx, &e, ids, spans, imageEmbeds, unwrapCache(cache), causal)
}

func (m *Model) prefillLogitsQwenVL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageFeats, mropePos, cache)
}

func (m *Model) prefillLogitsQwenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillLogitsQwenVLSpans(ctx, &e, qwen35VLPerToken, ids, spans, imageFeats, mropePos, unwrapCache(cache))
}
