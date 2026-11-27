package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
)

// residentUploadPrefill pushes a CPU-computed prefill's KV (every non-shared layer) into the resident GPU cache.
func (m *Model) residentUploadPrefill(cache *KVCache) error {
	h := m.resHost()
	return vlprep.UploadPrefillKV(m.w.Arch, unwrapCache(cache), m.resKV(), h.KVSlots(), h.ForgetIDs, m.resident.UploadKV)
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
