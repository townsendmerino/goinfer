package vlprep

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// PrefillQwen35VL prefills a Gated-DeltaNet hybrid (qwen3_5 / qwen3_5_moe) multimodal prompt under m-RoPE positions,
// taking the batched chunked path when eligible and !forcePerToken, or falling back to per-token recurrence.
func PrefillQwen35VL(ctx context.Context, eng *cpuengine.Engine, forcePerToken bool, ids []int, imageFeats []float32, spans []ImageSpan, mropePos [][3]int, cache *cpuengine.Cache) ([]float32, error) {
	arch := eng.W.Arch
	hidden := arch.HiddenDim
	if err := CheckSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	if err := CheckQwen35VLReady(arch); err != nil {
		return nil, err
	}
	cache.MRopePos = mropePos
	cache.MRopeDelta = rope.MRoPEDelta(mropePos, len(ids))
	if !forcePerToken && eng.Qwen35BatchNAnyPos(len(ids), cache) {
		return PrefillQwen35VLBatched(ctx, eng, ids, imageFeats, spans, cache)
	}
	return PrefillQwen35VLPerToken(ctx, eng, ids, imageFeats, spans, cache)
}

// PrefillQwen35VLBatched runs a Qwen3.5 multimodal prefill in passes of cpuengine.PrefillChunkRows rows, splicing raw
// image features into each pass while preserving sequential Gated-DeltaNet recurrence inside RunLayersQwen35N.
func PrefillQwen35VLBatched(ctx context.Context, eng *cpuengine.Engine, ids []int, imageFeats []float32, spans []ImageSpan, cache *cpuengine.Cache) ([]float32, error) {
	hidden := eng.W.Arch.HiddenDim
	if cache.Scr == nil {
		cache.Knobs, cache.Scr = eng.Knobs, cpuengine.NewDecodeScratch(eng.W.Arch, eng.Knobs)
	}
	n, C := len(ids), len(ids)
	if cache.CaptureLayers == nil && !cache.SubCapture && cache.TreeRowPos == nil && cache.TreeMask == nil {
		C = cpuengine.PrefillChunkRows
	}
	start := cache.Pos()
	var hn []float32
	for c0, i := 0, 0; c0 < n; i++ {
		c1 := min(c0+C, n)
		if i == 1 {
			cache.ReserveRows(max(start+n, cache.PlanRows) - cache.Pos())
			if cpuengine.PrefillChunkDone != nil {
				cpuengine.PrefillChunkDone(-1)
			}
		}
		h := make([]float32, (c1-c0)*hidden)
		for r, id := range ids[c0:c1] {
			eng.W.Embed.Row(id, h[r*hidden:(r+1)*hidden])
		}
		SpliceRange(h, spans, imageFeats, hidden, c0, c1)
		atomic.AddInt64(&cpuengine.PrefillPasses, 1)
		var err error
		if hn, err = eng.RunLayersQwen35N(ctx, h, cache); err != nil {
			return nil, err
		}
		if cpuengine.PrefillChunkDone != nil {
			cpuengine.PrefillChunkDone(i)
		}
		c0 = c1
	}
	return eng.LogitsFromNormed(hn[len(hn)-hidden:], cache), nil
}

// PrefillQwen35VLPerToken runs a Qwen3.5 multimodal prefill one token at a time through RunLayersFromEmbed.
func PrefillQwen35VLPerToken(ctx context.Context, eng *cpuengine.Engine, ids []int, imageFeats []float32, spans []ImageSpan, cache *cpuengine.Cache) ([]float32, error) {
	hidden := eng.W.Arch.HiddenDim
	all := make([]float32, len(ids)*hidden)
	for i := range ids {
		eng.W.Embed.Row(ids[i], all[i*hidden:(i+1)*hidden])
	}
	Splice(all, spans, imageFeats, hidden)
	var last []float32
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		h := all[i*hidden : (i+1)*hidden]
		var err error
		if last, err = eng.RunLayersFromEmbed(h, cache); err != nil {
			return nil, err
		}
	}
	return eng.LogitsFromHidden(last, cache), nil
}
