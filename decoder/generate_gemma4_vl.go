package decoder

import (
	"context"
	"fmt"
	"sync/atomic"
)

// prefillLogitsGemma4VL sequentially prefills a Gemma 4 multimodal prompt — ids
// with a run of imgLen image-placeholder ids at [imgPos, imgPos+imgLen) — and
// returns the last-position logits.
//
// UNLIKE prefillLogitsVL/prefillLogitsQwenVL, this walks ids ONE TOKEN AT A TIME
// via runLayersGemma4/runLayersGemma4FromEmbed (decoder/forward_gemma4.go) —
// Gemma 4's own-forward layer loop has no batched (multi-token-at-once) variant
// (gemma4 is dispatched via arch.ownForward(), not the generic
// runLayersFromEmbedN path every other multimodal family's CPU prefill uses).
// This is NOT an approximation for the checkpoints this reaches: E2B/E4B ship
// use_bidirectional_attention="" (docs/multimodal.md's P7 entry, confirmed
// against a real checkpoint), so the real HF forward attends to the image block
// strictly causally too — a sequential per-position walk IS that forward, not a
// stand-in for a bidirectional one. A checkpoint with
// use_bidirectional_attention="vision" (26B-A4B/31B) needs a genuinely batched,
// blockwise-masked forward — see prefillLogitsGemma4VLBidirectional /
// runLayersGemma4FromEmbedN (decoder/forward_gemma4_batched.go) below.
// GenerateGemma4VL dispatches to the right one; this function itself is never
// called for a "vision"-mode checkpoint.
func (m *Model) prefillLogitsGemma4VL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if arch.gemma4 == nil {
		return nil, fmt.Errorf("decoder: prefillLogitsGemma4VL called on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageEmbeds) != imgLen*hidden {
		return nil, fmt.Errorf("decoder: imageEmbeds len %d, want %d (%d tokens x %d hidden)", len(imageEmbeds), imgLen*hidden, imgLen, hidden)
	}
	padID := arch.gemma4.PadTokenID
	var h []float32
	var err error
	for i, id := range ids {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if i >= imgPos && i < imgPos+imgLen {
			off := (i - imgPos) * hidden
			// runLayersGemma4FromEmbed's own doc comment names exactly this case: a
			// multimodal caller substitutes a projected embedding for the normal
			// token-id lookup, and pleTokenID must be the checkpoint's pad_token_id
			// (not the placeholder token's own id) — the real HF multimodal forward
			// substitutes the PAD token's embedding at every image position before
			// computing PLE's token-identity term. The feature row is used as-is: no
			// embed_scale re-application (that scaling is runLayersGemma4's own
			// wrapper's job for a real token embedding; a tower's projected output is
			// already in the target embedding space, same as Gemma3/Qwen's own
			// embed-by-vector callers).
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

// prefillLogitsGemma4VLBidirectional is prefillLogitsGemma4VL's batched twin for
// 26B-A4B/31B-class checkpoints (use_bidirectional_attention: "vision"): a query
// position INSIDE the image/audio block must see LATER block positions too, which
// a sequential per-token walk can never provide (their K/V don't exist yet when
// an earlier position is processed). See runLayersGemma4FromEmbedN
// (decoder/forward_gemma4_batched.go) for the batched forward and
// gemma4AttendRange for the masking primitive + its correctness proof.
func (m *Model) prefillLogitsGemma4VLBidirectional(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if arch.gemma4 == nil {
		return nil, fmt.Errorf("decoder: prefillLogitsGemma4VLBidirectional called on a non-gemma4 model")
	}
	hidden := arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageEmbeds) != imgLen*hidden {
		return nil, fmt.Errorf("decoder: imageEmbeds len %d, want %d (%d tokens x %d hidden)", len(imageEmbeds), imgLen*hidden, imgLen, hidden)
	}
	h := m.embedN(ids)
	copy(h[imgPos*hidden:(imgPos+imgLen)*hidden], imageEmbeds) // raw projected features, no embed scale — same convention as embedN's own doc comment (forwardn.go)
	hLast, err := m.runLayersGemma4FromEmbedN(ctx, h, ids, imgPos, imgLen, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(hLast, cache), nil
}

// GenerateGemma4VL streams a continuation for a Gemma 4 multimodal prompt. Like
// GenerateVL (Gemma 3) / GenerateQwenVL in shape.
//
// Prefill dispatches between two forwards depending on the checkpoint:
// sequential (prefillLogitsGemma4VL, E2B/E4B-class, causal) or batched
// (prefillLogitsGemma4VLBidirectional, 26B-A4B/31B-class,
// use_bidirectional_attention: "vision"). Decode is always the unchanged
// per-token path either way — a decode token is never "inside" the image
// block again, so no masking distinction applies post-prefill.
//
// Resident GPU decode (gap 0, docs/multimodal.md): when the model has a resident, the CPU prefill's KV is
// uploaded through the generic residentUploadPrefill bridge (the one Gemma 3/Qwen's GenerateVL use) and decode
// runs resident. That was gated on UseBidirectionalAttention (the 26B-A4B/31B class) until no backend could run an
// E2B/E4B; since S1 (docs/tasks/task-multimodal-support-2026-10.md) Metal does — the bridge skips KV-shared layers,
// which own no KV, and embedResident carries the PLE inputs — so the E-model class is admitted too. CUDA and WebGPU
// still decline the E-model shape at load, so only Metal reaches this with one. A causal, non-E Gemma 4 keeps
// decoding on the CPU after an image, as before: widening to it is untested on CUDA.
//
// `imgHash`, `imgPos` and `imgLen` are also what the resident commit below records as this
// turn's residentImageBlock, so a later turn's prefix scan (residentReuseLen) can tell this
// image's span apart from ordinary text — Gemma4's soft-token placeholder ids are
// content-independent (same id/count per patch grid), so two different images of the same size
// produce IDENTICAL ids, and without a block record the scan cannot tell them apart (M-07,
// docs/audit-2026-09-10.md).
func (m *Model) GenerateGemma4VL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		bidirectional := m.w.Cfg.UseBidirectionalAttention != ""
		cache := m.NewCache(len(ids) + maxTokens)
		var logits []float32
		if bidirectional {
			logits, err = m.prefillLogitsGemma4VLBidirectional(ctx, ids, feats, imgPos, imgLen, cache)
		} else {
			logits, err = m.prefillLogitsGemma4VL(ctx, ids, feats, imgPos, imgLen, cache)
		}
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		eModel := m.w.arch.gemma4.HiddenSizePerLayerInput > 0 || m.w.arch.gemma4.SharedKVLayers > 0
		if (bidirectional || eModel) && m.tryClaimResident() {
			// The resident cache is about to hold THIS turn's content. If decode
			// completes naturally, residentCommitIDs below records it and this
			// defer's forget is skipped (committed=true); any other exit (error,
			// cancel, or the upload never engaging at all) forgets
			// unconditionally — same discipline as GenerateVL's own upload
			// bridge (decoder/generate_vl.go).
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
						g.BudgetClamped = true // the resident cap (not the request) bounds this turn (M-02, docs/audit-2026-09-10.md)
					}
				}
			}
		}

		g.Budget = maxTokens // M-02: publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
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
			// M-07 (docs/audit-2026-09-10.md): commit the image block the same way GenerateVL
			// and GenerateQwenVL do, so residentReuseLen's prefix scan has something to key a
			// later turn's reuse check on instead of falling through to a plain id comparison
			// that Gemma4's content-independent soft-token ids can satisfy by coincidence.
			m.residentCommitIDs(ids, generated, &residentImageBlock{start: imgPos, end: imgPos + imgLen, hash: imgHash}, nil)
			committed = true
		}
	}()
	return out, g
}
