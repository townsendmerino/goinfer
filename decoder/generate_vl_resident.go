package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
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
//
// A ring that has wrapped uploads no rows below its base, so the slot serves only a prefix whose window starts at or past it: kvmgr.Slots.SetFloor, on the shared path
// every resident backend takes.
func (m *Model) residentUploadPrefill(cache *KVCache) (err error) {
	floor := 0
	for l := 0; l < m.w.Arch.NumLayers; l++ {
		if r := cache.c.Rings[l]; r != nil && m.w.Arch.Gemma4KVSrcAt(l) == l {
			if _, _, base := cache.LayerKV(l); base > 0 {
				floor = max(floor, base+r.W-1)
			}
		}
	}
	defer func() {
		a := m.resHost().KVSlots()
		if err == nil {
			a.SetFloor(a.Bound(), floor)
			return
		}
		// Some rows are written and the rest are what they were, so nothing the record says holds, and nothing the slot already lacked came back.
		m.resHost().ForgetIDs()
		a.RaiseFloor(a.Bound(), floor)
	}()
	if kv := m.resKV(); kv.KVUploadPassAvailable() {
		if err := kv.BeginKVUpload(); err != nil {
			return err
		}
		defer func() {
			if e := kv.EndKVUpload(err != nil); err == nil {
				err = e
			}
		}()
	}
	for l := 0; l < m.w.Arch.NumLayers; l++ {
		if m.w.Arch.Gemma4KVSrcAt(l) != l {
			continue // a Gemma 4 E-model KV-shared layer owns no KV: the resident reads its source's
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
// and the UploadKV bridge entirely. One span goes to ImagePrefillAvailable; several need
// ImageBlocksPrefillAvailable. Any error is a DECLINE, not fatal — the caller falls through to the
// CPU-prefill+UploadKV bridge (gap 0, docs/multimodal.md).
func (m *Model) residentImagePrefill(ctx context.Context, r any, ids []int, imageEmbeds []float32, spans []ImageSpan) (logits []float32, gpuPos int, err error) {
	if len(spans) > 1 && !canStep(r, stepBatch{Pass: passRows, In: inImageBlocks, Emb: stepProbe}) {
		return nil, 0, fmt.Errorf("decoder: this backend's resident image prefill takes one image block and the prompt has %d; using the CPU prefill and upload", len(spans))
	}
	rows, err := vlprep.SplicedRows(m.embedN, ids, spans, imageEmbeds, m.w.Arch.HiddenDim)
	if err != nil {
		return nil, 0, err
	}
	var sr stepResult
	if len(spans) > 1 {
		sr, err = stepOn(r, stepBatch{Pass: passRows, Ctx: ctx, In: inImageBlocks, Emb: rows, Spans: imageSpanBlocks(spans)})
	} else {
		sr, err = stepOn(r, stepBatch{Pass: passRows, Ctx: ctx, In: inImage, Emb: rows, Spans: [][2]int{{spans[0].Pos, spans[0].Pos + spans[0].Len}}})
	}
	logits = sr.Logits
	if err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}

// residentMRoPEPrefillDeep builds the spliced embedding rows and hands them with mropePos and optional DeepStack sets to
// the resident backend's m-RoPE prefill. Any error is a decline falling through to CPU prefill + UploadKV.
func (m *Model) residentMRoPEPrefillDeep(ctx context.Context, r any, ids []int, imageFeats []float32, spans []ImageSpan, mropePos [][3]int, deep [][]float32) (logits []float32, gpuPos int, err error) {
	if len(mropePos) != len(ids) {
		if err := checkImageSpans(spans, len(ids), imageFeats, m.w.Arch.HiddenDim); err != nil {
			return nil, 0, err
		}
		return nil, 0, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	if deep != nil && len(spans) > 1 {
		if err := checkImageSpans(spans, len(ids), imageFeats, m.w.Arch.HiddenDim); err != nil {
			return nil, 0, err
		}
		return nil, 0, fmt.Errorf("decoder: the resident DeepStack prefill takes one image and the prompt has %d; using the CPU prefill and upload", len(spans))
	}
	rows, err := vlprep.SplicedRows(m.embedN, ids, spans, imageFeats, m.w.Arch.HiddenDim)
	if err != nil {
		return nil, 0, err
	}
	var sr stepResult
	if deep != nil {
		b := stepBatch{Pass: passRows, Ctx: ctx, In: inMRoPEDeepstack, Emb: rows, MRoPE: mropePos, Deep: deep,
			Spans: [][2]int{{spans[0].Pos, spans[0].Pos + spans[0].Len}}}
		if !canStep(r, b) {
			return nil, 0, fmt.Errorf("decoder: the resident has no DeepStack prefill")
		}
		sr, err = stepOn(r, b)
	} else {
		sr, err = stepOn(r, stepBatch{Pass: passRows, Ctx: ctx, In: inMRoPE, Emb: rows, MRoPE: mropePos})
	}
	logits = sr.Logits
	if err != nil {
		return nil, 0, err
	}
	return logits, len(ids), nil
}

// residentCausalImagePrefill runs spliced multimodal rows for a causal image family through the backend's batched pass.
func (m *Model) residentCausalImagePrefill(ctx context.Context, r any, ids []int, imageEmbeds []float32, spans []ImageSpan) (logits []float32, gpuPos int, err error) {
	rows, err := vlprep.SplicedRows(m.embedN, ids, spans, imageEmbeds, m.w.Arch.HiddenDim)
	if err != nil {
		return nil, 0, err
	}
	m.resHost().ForgetIDsIn(kvScopeOf(ctx))
	m.resHost().WritesFromZero()
	sr, err := stepOn(r, stepBatch{Pass: passRows, Ctx: ctx, Emb: rows})
	if err != nil {
		return nil, 0, err
	}
	return sr.Logits, len(ids), nil
}
