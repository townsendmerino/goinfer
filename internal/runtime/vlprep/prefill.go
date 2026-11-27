package vlprep

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// PrefillSpliced runs ids through eng.PrefillPassesN with each pass's placeholder rows replaced by raw feats,
// returning every position's post-norm hidden rows when all is true, else the final pass's rows.
func PrefillSpliced(ctx context.Context, eng *cpuengine.Engine, ids []int, spans []ImageSpan, feats []float32, cache *cpuengine.Cache, all bool) ([]float32, error) {
	hidden := eng.W.Arch.HiddenDim
	var out []float32
	var keep func(lo int, rows []float32)
	if all {
		out = make([]float32, len(ids)*hidden)
		keep = func(lo int, rows []float32) { copy(out[lo*hidden:], rows) }
	}
	h, err := eng.PrefillPassesN(ctx, len(ids), cache, func(dst []float32, lo, hi int) {
		eng.EmbedNInto(dst, ids[lo:hi])
		SpliceRange(dst, spans, feats, hidden, lo, hi)
	}, keep)
	if all {
		return out, err
	}
	return h, err
}

// PrefillVLRows validates spans, arms bidirectional image blocks when !causal, and runs PrefillSpliced.
func PrefillVLRows(ctx context.Context, eng *cpuengine.Engine, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *cpuengine.Cache, causal, all bool) ([]float32, error) {
	if !eng.CanBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: multimodal prefill needs the batched path (canBatchN false)")
	}
	if err := CheckSpans(spans, len(ids), imageEmbeds, eng.W.Arch.HiddenDim); err != nil {
		return nil, err
	}
	if !causal {
		cache.SetImageBlocks(Blocks(spans))
	}
	return PrefillSpliced(ctx, eng, ids, spans, imageEmbeds, cache, all)
}

// PrefillHiddenLFM2VL prefills an LFM2-VL multimodal prompt sequentially through the per-token walker so conv windows
// observe every position in order, returning all post-final-norm hidden rows.
func PrefillHiddenLFM2VL(ctx context.Context, eng *cpuengine.Engine, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *cpuengine.Cache) ([]float32, error) {
	hidden := eng.W.Arch.HiddenDim
	if err := CheckSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := eng.EmbedN(ids)
	Splice(h, spans, imageEmbeds, hidden)
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := h[i*hidden : (i+1)*hidden]
		out, err := eng.RunLayersFromEmbed(row, cache)
		if err != nil {
			return nil, err
		}
		eng.FinalNorm(out)
	}
	return h, nil
}

// PrefillHiddenVLSpans prefills a multimodal prompt and returns every position's post-norm hidden rows.
func PrefillHiddenVLSpans(ctx context.Context, eng *cpuengine.Engine, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *cpuengine.Cache, causal bool) ([]float32, error) {
	if causal && eng.W.Arch.LFM2 != nil {
		return PrefillHiddenLFM2VL(ctx, eng, ids, spans, imageEmbeds, cache)
	}
	return PrefillVLRows(ctx, eng, ids, spans, imageEmbeds, cache, causal, true)
}

// PrefillLogitsVLSpans prefills a multimodal prompt and returns the last position's logits.
func PrefillLogitsVLSpans(ctx context.Context, eng *cpuengine.Engine, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *cpuengine.Cache, causal bool) ([]float32, error) {
	var hN []float32
	var err error
	if causal && eng.W.Arch.LFM2 != nil {
		hN, err = PrefillHiddenLFM2VL(ctx, eng, ids, spans, imageEmbeds, cache)
	} else {
		hN, err = PrefillVLRows(ctx, eng, ids, spans, imageEmbeds, cache, causal, false)
	}
	if err != nil {
		return nil, err
	}
	hidden := eng.W.Arch.HiddenDim
	return eng.LMHeadNInto(hN[len(hN)-hidden:], 1, cache.HeadLogits(eng.W.Arch.VocabSize)), nil
}

// PrefillLogitsQwenVLSpans prefills a Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL multimodal prompt under 3D m-RoPE positions
// and returns the last position's logits.
func PrefillLogitsQwenVLSpans(ctx context.Context, eng *cpuengine.Engine, forcePerToken bool, ids []int, spans []ImageSpan, imageFeats []float32, mropePos [][3]int, cache *cpuengine.Cache) ([]float32, error) {
	if eng.W.Arch.Qwen35 != nil {
		return PrefillQwen35VL(ctx, eng, forcePerToken, ids, imageFeats, spans, mropePos, cache)
	}
	if !eng.CanBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: Qwen2.5-VL prefill needs the batched path (canBatchN false)")
	}
	hidden := eng.W.Arch.HiddenDim
	if err := CheckSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	cache.MRopePos = mropePos
	cache.MRopeDelta = rope.MRoPEDelta(mropePos, len(ids))
	hN, err := PrefillSpliced(ctx, eng, ids, spans, imageFeats, cache, false)
	if err != nil {
		return nil, err
	}
	return eng.LMHeadNInto(hN[len(hN)-hidden:], 1, cache.HeadLogits(eng.W.Arch.VocabSize)), nil
}

// PrefillLogitsAudio prefills a causal soft-token audio prompt (ids[pos : pos+n] replaced by raw feats) and returns the
// last position's logits.
func PrefillLogitsAudio(ctx context.Context, eng *cpuengine.Engine, ids []int, feats []float32, pos, n int, cache *cpuengine.Cache) ([]float32, error) {
	if !eng.CanBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: audio prefill needs the batched path (canBatchN false)")
	}
	hidden := eng.W.Arch.HiddenDim
	if pos < 0 || n <= 0 || pos+n > len(ids) {
		return nil, fmt.Errorf("decoder: audio run [%d,%d) out of range for %d tokens", pos, pos+n, len(ids))
	}
	if len(feats) != n*hidden {
		return nil, fmt.Errorf("decoder: audio features len %d, want %d (%d tokens x %d)", len(feats), n*hidden, n, hidden)
	}
	hN, err := PrefillSpliced(ctx, eng, ids, []ImageSpan{{Pos: pos, Len: n}}, feats, cache, false)
	if err != nil {
		return nil, err
	}
	return eng.LMHeadN(hN[len(hN)-hidden:], 1), nil
}
