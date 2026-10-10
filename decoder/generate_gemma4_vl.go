package decoder

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
)

// prefillLogitsGemma4VL sequentially prefills a Gemma 4 multimodal prompt (ids with a run of imgLen image-placeholder
// ids at [imgPos, imgPos+imgLen)) and returns the last-position logits.
//
// Unlike prefillLogitsVL and prefillLogitsQwenVL, this walks ids one token at a time via runLayersGemma4 and
// runLayersGemma4FromEmbed (forward_gemma4.go): Gemma 4's own-forward layer loop has no batched variant (gemma4 is
// dispatched via arch.ownForward(), not the generic runLayersFromEmbedN path). That is exact for a checkpoint that ships
// use_bidirectional_attention="" (the E-models, docs/multimodal.md P7 entry): the reference attends to the image block
// strictly causally too, so a sequential per-position walk is that forward. A checkpoint with
// use_bidirectional_attention="vision" needs the batched, blockwise-masked forward
// (prefillLogitsGemma4VLBidirectional); GenerateGemma4VL dispatches, and this function is never called for a
// "vision"-mode checkpoint.
func (m *Model) prefillLogitsGemma4VL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VLSpans(ctx, ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}}, cache)
}

// prefillLogitsGemma4VLSpans is prefillLogitsGemma4VL for several images: every span's rows, concatenated in span
// order in imageEmbeds, replace that span's placeholders.
func (m *Model) prefillLogitsGemma4VLSpans(ctx context.Context, ids []int, imageEmbeds []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if arch.gemma4 == nil {
		return nil, fmt.Errorf("decoder: prefillLogitsGemma4VL called on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	padID := arch.gemma4.PadTokenID
	var h []float32
	var err error
	for i, id := range ids {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if r, in := imageSpanRow(spans, i); in {
			off := r * hidden
			// runLayersGemma4FromEmbed names this case: a multimodal caller substitutes a projected embedding for the token-id
			// lookup, and pleTokenID must be the checkpoint's pad_token_id, not the placeholder's own id, because the reference
			// substitutes the PAD token's embedding at every image position before computing PLE's token-identity term. The feature
			// row is used as is, with no embed_scale: that scaling is runLayersGemma4's job for a real token embedding, and a
			// tower's projected output is already in the target embedding space.
			row := append([]float32(nil), imageEmbeds[off:off+hidden]...)
			h, err = m.runLayersGemma4FromEmbed(row, padID, cache)
		} else {
			h, err = m.runLayersGemma4(id, cache)
		}
		if err != nil {
			return nil, err
		}
	}
	return m.logitsFromHidden(h, cache), nil
}

// prefillLogitsGemma4VLBidirectional is prefillLogitsGemma4VL's batched twin for checkpoints with
// use_bidirectional_attention: "vision": a query position inside the image or audio block must see later block
// positions too, which a sequential per-token walk can never provide (their K/V do not exist yet when an earlier
// position is processed). See runLayersGemma4FromEmbedN (forward_gemma4_batched.go) for the batched forward and
// gemma4AttendRange for the masking primitive and its correctness proof.
func (m *Model) prefillLogitsGemma4VLBidirectional(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VLBidirectionalSpans(ctx, ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}}, cache)
}

// prefillLogitsGemma4VLBidirectionalSpans is prefillLogitsGemma4VLBidirectional for several images: each span is its
// own bidirectional block (gemma4AttendRange).
func (m *Model) prefillLogitsGemma4VLBidirectionalSpans(ctx context.Context, ids []int, imageEmbeds []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if arch.gemma4 == nil {
		return nil, fmt.Errorf("decoder: prefillLogitsGemma4VLBidirectional called on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden) // raw projected features, no embed scale — same convention as embedN's own doc comment (forwardn.go)
	hLast, err := m.runLayersGemma4FromEmbedN(ctx, h, ids, spans, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(hLast, cache), nil
}

// GenerateGemma4VL streams a continuation for a Gemma 4 multimodal prompt, in the shape of GenerateVL and
// GenerateQwenVL.
//
// Prefill dispatches between two forwards depending on the checkpoint: sequential (prefillLogitsGemma4VL, causal) or
// batched (prefillLogitsGemma4VLBidirectional, use_bidirectional_attention: "vision"). Decode is always the unchanged
// per-token path: a decode token is never inside the image block again, so no masking distinction applies after prefill.
//
// Resident GPU decode (docs/multimodal.md): when the model has a resident, the CPU prefill's KV is uploaded through the
// generic residentUploadPrefill bridge (the one GenerateVL uses) and decode runs resident. The bridge skips KV-shared
// layers, which own no KV, and embedResident carries the PLE inputs, so the E-model class is admitted wherever a backend
// can run it (gemma4ImageDecodeResident holds the rule); a backend that declines the E-model shape at load keeps decoding
// on the CPU after an image. A causal, non-E Gemma 4 also keeps decoding on the CPU after an image: widening to it is
// untested on CUDA.
//
// imgHash, imgPos and imgLen are also what the resident commit records as this turn's residentImageBlock, so a later
// turn's prefix scan (residentReuseLen) can tell this image's span from ordinary text: Gemma 4's soft-token placeholder
// ids are content-independent, so two different images of the same size produce identical ids, and without a block
// record the scan cannot tell them apart.
func (m *Model) GenerateGemma4VL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.GenerateGemma4VLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen, Hash: imgHash}}, features, maxTokens, sp)
}

// GenerateGemma4VLSpans is GenerateGemma4VL for several images: spans in prompt order, features every span's rows
// concatenated in span order. A bidirectional checkpoint gives each span its own block; an E-model is causal, so its
// resident prefill takes several images unchanged.
func (m *Model) GenerateGemma4VLSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		if len(spans) == 0 {
			g.err = fmt.Errorf("decoder: GenerateGemma4VLSpans needs at least one image span")
			return
		}
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		bidirectional := m.w.Cfg.UseBidirectionalAttention != ""
		eModel := gemma4IsEModel(m.w.arch)
		// An E-model's image turn prefills on the resident when the backend takes it, from the rows the CPU path would run: a
		// text position's resident embedding ([h || PLE]), an image position's projected feature, unscaled, with PLE from PAD's
		// token identity. A decline falls through to the CPU prefill and upload below, unchanged.
		if eModel && !bidirectional {
			if done := m.gemma4VLResidentPrefill(ctx, out, g, ids, feats, spans, maxTokens, sp); done {
				return
			}
		}
		cache := m.NewCache(len(ids) + maxTokens)
		var logits []float32
		if bidirectional {
			logits, err = m.prefillLogitsGemma4VLBidirectionalSpans(ctx, ids, feats, spans, cache)
		} else {
			logits, err = m.prefillLogitsGemma4VLSpans(ctx, ids, feats, spans, cache)
		}
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		if gemma4ImageDecodeResident(bidirectional, eModel) && m.tryClaimResident() { // support_table.go reads the same rule
			// The resident cache is about to hold this turn's content. If decode completes naturally, residentCommitIDs below
			// records it and this defer's forget is skipped (committed=true); any other exit (error, cancel, or the upload never
			// engaging) forgets unconditionally, as in GenerateVL's upload bridge.
			defer func() {
				if !committed {
					m.residentForgetIDs()
				}
				atomic.StoreInt32(&m.resBusy, 0)
			}()
			if uerr := m.residentUploadPrefill(cache); uerr == nil {
				useGPU = true
				g.DecodeResident = true
				gpuPos = len(ids)
				if capper, ok := m.resident.(ResidentCapped); ok {
					if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
						maxTokens = ctxCap - gpuPos
						g.BudgetClamped = true // the resident cap (not the request) bounds this turn
					}
				}
			}
		}

		g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
		sampler := NewSampler(sp)
		sampler.Observe(ids...)
		generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
			if useGPU {
				l, err := m.resident.Forward(m.embedResident(next), gpuPos)
				gpuPos++
				return l, err
			}
			// Decode reuses the plain per-token path unmodified: m.forward already
			// dispatches to runLayersGemma4 via arch.ownForward(), so no
			// gemma4-specific decode step is needed once the image is prefilled.
			return m.forward(next, cache)
		})
		if useGPU && g.err == nil {
			// Commit the image block the way GenerateVL and GenerateQwenVL do, so residentReuseLen's prefix scan has something to
			// key a later turn's reuse check on, instead of a plain id comparison that Gemma 4's content-independent soft-token ids
			// can satisfy by coincidence.
			m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
			committed = true
		}
	}()
	return out, g
}

// gemma4VLResidentPrefill is GenerateGemma4VL's resident image prefill for an E-model: the whole prompt through the
// backend's Prefiller, then resident decode. It reports whether it ran the turn (true: the generation is done, g holds
// its result); false means the backend has no prefill, the resident is busy, or the prefill declined, and the caller
// runs the CPU path. A cancelled prefill is not a decline: it ends the turn.
func (m *Model) gemma4VLResidentPrefill(ctx context.Context, out chan<- int, g *Generation, ids []int, feats []float32, spans []ImageSpan, maxTokens int, sp SamplingParams) bool {
	pf, ok := m.resident.(Prefiller)
	if !ok || m.knobs.get(knobBatchedPrefill) == "0" || !m.tryClaimResident() {
		return false
	}
	rows, err := m.gemma4EModelImageRowsSpans(ids, feats, spans)
	if err != nil { // a malformed image run is the CPU path's to report; it is not this one's to panic on
		atomic.StoreInt32(&m.resBusy, 0)
		return false
	}
	m.residentForgetIDs() // the prefill overwrites the resident cache from position 0
	logits, err := pf.PrefillLast(ctx, rows, 0)
	if err != nil {
		atomic.StoreInt32(&m.resBusy, 0)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			g.err = err
			return true
		}
		return false
	}
	g.ImgPrefillResident, g.DecodeResident = true, true
	gpuPos := len(ids)
	if capper, ok := m.resident.(ResidentCapped); ok {
		if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
			maxTokens = ctxCap - gpuPos
			g.BudgetClamped = true
		}
	}
	g.Budget = maxTokens
	sampler := NewSampler(sp)
	sampler.Observe(ids...)
	generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
		l, err := m.resident.Forward(m.embedResident(next), gpuPos)
		gpuPos++
		return l, err
	})
	if g.err == nil {
		m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
	} else {
		m.residentForgetIDs()
	}
	atomic.StoreInt32(&m.resBusy, 0)
	return true
}

// gemma4ResidentMediaRow is the resident row for one image or audio position of a Gemma 4 E-model: the projected
// feature, unscaled (as the CPU's prefillLogitsGemma4VL splices it), then for an E-model the PLE inputs from PAD's token
// identity (as runLayersGemma4FromEmbed computes them).
func (m *Model) gemma4ResidentMediaRow(feature []float32) []float32 {
	hidden := m.w.arch.HiddenDim
	row := make([]float32, m.ResidentEmbedLen())
	copy(row[:hidden], feature)
	if len(row) > hidden {
		m.gemma4PLEInputs(row[:hidden], m.w.arch.gemma4.PadTokenID, row[hidden:])
	}
	return row
}

// gemma4EModelImageRowsSpans builds the resident rows of an E-model image prompt (any number of images): a text
// position's row is embedResident's ([h || PLE]); an image position's is gemma4ResidentMediaRow of its projected
// feature. It checks the image run against the prompt and the feature count, and returns an error (a decline) rather
// than slice out of range. gemma4VLRowsDefectForTest plants defects here for the tests that prove the gate can fail.
func (m *Model) gemma4EModelImageRowsSpans(ids []int, feats []float32, spans []ImageSpan) ([][]float32, error) {
	arch := m.w.arch
	if arch.gemma4 == nil {
		return nil, fmt.Errorf("decoder: gemma4EModelImageRows on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), feats, hidden); err != nil {
		return nil, err
	}
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		r, in := imageSpanRow(spans, i)
		if !in || gemma4VLRowsDefectForTest == 4 {
			rows[i] = m.embedResident(id)
			continue
		}
		f := feats[r*hidden : (r+1)*hidden]
		switch gemma4VLRowsDefectForTest {
		case 1: // planted: the placeholder's own id instead of the pad id
			row := make([]float32, m.ResidentEmbedLen())
			copy(row[:hidden], f)
			m.gemma4PLEInputs(row[:hidden], id, row[hidden:])
			rows[i] = row
		case 2: // planted: the feature scaled like a token embedding
			sc := float32(1)
			if e := arch.EmbedScale; e != 0 && e != 1 {
				sc = float32(e)
			}
			g := make([]float32, hidden)
			for k := range g {
				g[k] = f[k] * sc
			}
			rows[i] = m.gemma4ResidentMediaRow(g)
		case 3: // planted: the PLE tail left zero
			row := m.gemma4ResidentMediaRow(f)
			clear(row[hidden:])
			rows[i] = row
		default:
			rows[i] = m.gemma4ResidentMediaRow(f)
		}
	}
	return rows, nil
}

// gemma4VLRowsDefectForTest is a planted-defect selector for gemma4EModelImageRowsSpans (0 = none): 1 pad id replaced
// by the placeholder's id, 2 the feature scaled by the embed scale, 3 the PLE tail zero, 4 image positions built as
// text rows. Set only through SetGemma4VLRowsDefectForTest (goinfer_testhooks).
var gemma4VLRowsDefectForTest int
