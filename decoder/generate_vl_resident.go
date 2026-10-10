package decoder

import (
	"context"
	"fmt"
)

// residentUploadPrefill pushes a CPU-computed prefill's KV (every layer) into the resident GPU
// cache — the bridge that lets GenerateVL/GenerateQwenVL's bidirectional-image-block CPU
// prefill (prefillLogitsVL/prefillLogitsQwenVL) seed a resident decode instead of running the
// whole turn on CPU (gap 0, docs/multimodal.md). cache must hold exactly the prefill this turn
// just ran — nothing more, nothing skipped.
//
// Per layer, KVCache.LayerKV (not Keys/Vals, which read empty for a sliding-window/ring layer or
// an int8-quantized CPU cache) returns the live K/V in absolute position order plus base, the
// absolute position of its first row — 0 for a global layer or a ring that has not wrapped,
// greater than 0 once a local layer's ring has wrapped past its window. UploadKV places those
// rows at base..base+n-1 in the resident cache, matching where a resident Forward would have
// written them.
func (m *Model) residentUploadPrefill(cache *KVCache) error {
	for l := 0; l < m.w.arch.NumLayers; l++ {
		if m.w.arch.gemma4KVSrcAt(l) != l {
			continue // a Gemma 4 E-model KV-shared layer owns no KV: the resident reads its source's (S1.4)
		}
		k, v, base := cache.LayerKV(l)
		if len(k) == 0 {
			continue // a layer with nothing live yet (shouldn't happen post-prefill, but not fatal)
		}
		if err := m.resident.UploadKV(l, base, k, v); err != nil {
			return fmt.Errorf("decoder: resident prefill upload layer %d (base=%d): %w", l, base, err)
		}
	}
	return nil
}

// residentImagePrefill builds the same spliced embedding rows prefillLogitsVL's CPU path would
// have built (embedN + the raw-feature splice of every span, no embed scale — matching HF's
// masked_scatter) and hands them to the resident backend's image prefill, skipping the CPU prefill
// and the UploadKV bridge entirely. One span goes to ResidentImagePrefill; several (S11) need
// ResidentImageBlocksPrefill. Any error is a DECLINE, not fatal — the caller falls through to the
// CPU-prefill+UploadKV bridge (gap 0, docs/multimodal.md).
func (m *Model) residentImagePrefill(ctx context.Context, rip ResidentImagePrefill, ids []int, imageEmbeds []float32, spans []ImageSpan) (logits []float32, gpuPos int, err error) {
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, 0, err
	}
	var ribp ResidentImageBlocksPrefill
	if len(spans) > 1 {
		var ok bool
		if ribp, ok = rip.(ResidentImageBlocksPrefill); !ok {
			return nil, 0, fmt.Errorf("decoder: this backend's resident image prefill takes one image block and the prompt has %d; using the CPU prefill and upload", len(spans))
		}
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden) // raw projected features, no embed scale
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden]
	}
	if ribp != nil {
		logits, err = ribp.PrefillImageBlocksLast(ctx, rows, 0, imageSpanBlocks(spans))
	} else {
		logits, err = rip.PrefillImageLast(ctx, rows, 0, spans[0].Pos, spans[0].Pos+spans[0].Len)
	}
	if err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}

// residentMRoPEPrefillDeep is residentImagePrefill's Qwen twin: builds the same spliced
// embedding rows prefillLogitsQwenVL's CPU path would have built (embedN + the raw-feature
// splice at [imgPos,imgPos+imgLen), no embed scale — matching HF's scatter into inputs_embeds)
// and hands them, together with mropePos (GenerateQwenVL's own mropePositions output, already
// computed for its other branches), to the resident backend's ResidentMRoPEPrefill. Any error is
// a DECLINE, not fatal — the caller falls through to the CPU-prefill+UploadKV bridge.
//
// With Qwen3-VL's DeepStack sets (S16), deep is non-nil and the resident must implement ResidentMRoPEDeepstackPrefill, or
// the call declines (and the caller falls through to the CPU prefill); nil is Qwen2.5-VL's plain m-RoPE prefill.
func (m *Model) residentMRoPEPrefillDeep(ctx context.Context, rmp ResidentMRoPEPrefill, ids []int, imageFeats []float32, spans []ImageSpan, mropePos [][3]int, deep [][]float32) (logits []float32, gpuPos int, err error) {
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, 0, err
	}
	if len(mropePos) != len(ids) {
		return nil, 0, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	if deep != nil && len(spans) > 1 { // the resident DeepStack prefill injects one run (S11)
		return nil, 0, fmt.Errorf("decoder: the resident DeepStack prefill takes one image and the prompt has %d; using the CPU prefill and upload", len(spans))
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageFeats, hidden) // raw merged features, no embed scale
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden]
	}
	if deep != nil {
		rdp, ok := rmp.(ResidentMRoPEDeepstackPrefill)
		if !ok {
			return nil, 0, fmt.Errorf("decoder: the resident has no DeepStack prefill")
		}
		logits, err = rdp.PrefillMRoPEDeepstackLast(ctx, rows, 0, mropePos, deep, spans[0].Pos, spans[0].Len)
	} else {
		logits, err = rmp.PrefillMRoPELast(ctx, rows, 0, mropePos)
	}
	if err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}

// residentCausalImagePrefill is residentImagePrefill for a causal image family (Pixtral, S10): the same spliced rows,
// through the backend's plain batched pass (Prefiller.PrefillLast), which is exactly a causal prefill. The resident's
// recorded ids are forgotten first: the pass overwrites its cache from position 0 even when it then declines.
func (m *Model) residentCausalImagePrefill(ctx context.Context, pf Prefiller, ids []int, imageEmbeds []float32, spans []ImageSpan) (logits []float32, gpuPos int, err error) {
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, 0, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden)
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden]
	}
	m.residentForgetIDs()
	if logits, err = pf.PrefillLast(ctx, rows, 0); err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}
