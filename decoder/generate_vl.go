package decoder

import (
	"context"
	"fmt"
	"sync/atomic"
)

// vlDecodeLoop is the shared sampler/stream/stop decode loop behind GenerateVL and GenerateQwenVL, so the control-flow
// branches they need (CPU decode, resident decode, and each of those again inside the image-reuse fast path,
// docs/multimodal.md) do not each carry a copy. logits is the seed (from prefill or from the reused prefix's last
// resident position); forward advances exactly one token (its closure decides CPU, resident or resident m-RoPE). It
// returns every token actually streamed, for the caller's commit-or-forget decision; g.err is set on any early exit.
func (m *Model) vlDecodeLoop(ctx context.Context, out chan<- int, g *Generation, sampler *Sampler, maxTokens int, sp SamplingParams, logits []float32, forward func(next int) ([]float32, error)) (generated []int) {
	for range maxTokens {
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			return generated
		default:
		}
		if sp.LogitProcessor != nil {
			sp.LogitProcessor(generated, logits)
		}
		info, err := sampler.SampleWithInfo(logits)
		if err != nil {
			g.err = err
			return generated
		}
		next := info.ID
		if m.isStop(next, sp) {
			return generated
		}
		if sp.Logprobs {
			g.Logprobs = append(g.Logprobs, info)
		}
		// Select on ctx.Done so a consumer that stops ranging cannot wedge this goroutine forever on a bare send while it
		// holds the KV cache.
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			return generated
		case out <- next:
		}
		generated = append(generated, next)
		if logits, err = forward(next); err != nil {
			g.err = err
			return generated
		}
	}
	return generated
}

// GenerateVL streams a continuation for a multimodal (vision-language) prompt. ids are the text token ids with a run of
// imgLen image-placeholder ids starting at imgPos; imgHash is a content hash of the raw image bytes behind that run,
// used for resident prefix reuse (docs/multimodal.md). A caller with no reuse story of its own may pass 0, which
// residentReuseLen treats as "no claim" and never matches. features is invoked at most once, and only when the image
// cannot be fully reused from the resident KV: a lazy closure, so an unchanged, resent screenshot never re-runs the
// vision tower.
//
// Prefill runs through the bidirectional image-block mask (prefillLogitsVL): image tokens attend mutually, text
// attends causally, and tokens decoded after the image attend causally too. It then decodes up to maxTokens like
// Generate (same Sampler, LogitProcessor and stop rule).
//
// Resident GPU decode claims the shared resident KV when available and not busy with a concurrent generation, and falls
// back to the CPU decode loop otherwise. Before paying for the tower or the CPU prefill, it checks whether this turn's
// image and everything before it is already in the resident KV from a prior turn; if so, decode resumes from there
// with no re-tower, re-prefill or re-upload. Any other outcome (no resident, a different image, busy) takes the
// full-prefill path. The caller ranges over the returned channel, then checks Generation.Err for a terminal error.
func (m *Model) GenerateVL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.GenerateVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen, Hash: imgHash}}, features, maxTokens, sp)
}

// GenerateVLSpans is GenerateVL for a prompt with several images: spans are the images' placeholder runs in prompt
// order, each its own bidirectional block, and features returns every span's rows concatenated in span order. Reuse
// claims every span; the resident image prefill takes them all when the backend implements
// ResidentImageBlocksPrefill (one span needs only ResidentImagePrefill), and otherwise the turn takes the CPU-prefill +
// UploadKV bridge with the reason in Generation.ImgPrefillDecline.
func (m *Model) GenerateVLSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.generateVLSpans(ctx, ids, spans, features, maxTokens, sp, false)
}

// GenerateVLCausalSpans is GenerateVLSpans for a family whose image tokens attend causally, with plain 1-D positions
// (Ministral 3's Pixtral): features fill only the [IMG] positions, one span per merged row, the [IMG_BREAK]/[IMG_END]
// between them ordinary text. No bidirectional block and no window check; the resident prefill is the backend's plain
// batched pass (Prefiller) over the spliced rows.
func (m *Model) GenerateVLCausalSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.generateVLSpans(ctx, ids, spans, features, maxTokens, sp, true)
}

func (m *Model) generateVLSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, error), maxTokens int, sp SamplingParams, causal bool) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)

		if len(spans) == 0 {
			g.err = fmt.Errorf("decoder: GenerateVLSpans needs at least one image span")
			return
		}
		for _, s := range spans {
			if causal {
				break // no bidirectional block, so no window to outgrow
			}
			if err := m.checkImageBlockFitsWindow(s.Len); err != nil {
				g.err = err
				return
			}
		}
		lastEnd := spans[len(spans)-1].Pos + spans[len(spans)-1].Len

		// Reuse fast path: peek the resident KV for a full-image reuse before touching the tower or the CPU prefill. Held only
		// long enough to check and use; released either way.
		if m.tryClaimResident() {
			claim := residentImageClaimsOf(spans)
			// A prompt in (ResidentContextCap, MaxPositions) has passed prepare's MaxPositions check, so residentPrefillSeed
			// below would die mid-prefill with no CPU fallback (its error just sets g.err and returns). Decline the fast path up
			// front when it would overrun the cap, falling through to the "not fully reused" path (release resBusy, tower + CPU
			// prefill + re-claim).
			if reuseFrom := m.residentReuseLen(ids, claim, nil); reuseFrom >= lastEnd {
				if ctxCap := m.ResidentContextCap(); ctxCap <= 0 || len(ids) <= ctxCap {
					// FULL reuse: the image, and everything before it, is already resident.
					// Stay inside THIS claim (no release/reclaim) — reseed via the ordinary
					// ResidentForward.Forward path (no CPU prefill, no tower call at all).
					g.PrefillReused = reuseFrom                                      // observable proof the fast path actually fired
					m.residentForgetIDs()                                            // forget first: from here the cache is mid-write
					logits, err := m.residentPrefillSeed(ctx, ids, reuseFrom, false) // no adapter path here yet
					if err != nil {
						atomic.StoreInt32(&m.resBusy, 0)
						g.err = err
						return
					}
					gpuPos := len(ids)
					if capper, ok := m.resident.(ResidentCapped); ok {
						if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
							maxTokens = ctxCap - gpuPos
							g.BudgetClamped = true // the resident cap (not the request) bounds this turn
						}
					}
					g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
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
					return
				}
			}
			// Not fully reused, or declined by the cap check above: release immediately; do not hold the claim across the tower
			// call below.
			atomic.StoreInt32(&m.resBusy, 0)
		}

		// Ordinary path: the tower runs outside any claim, then a full CPU prefill (always from=0; this branch never partially
		// reuses), then a fresh re-claim for upload and decode only.
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}

		// Resident image-prefill fast path, tried before the CPU prefillLogitsVL call: a cold image turn's prefill, not only
		// its decode, runs on the resident GPU when the backend implements ResidentImagePrefill. Any decline (no capability,
		// prompt too large for one chunk, resident busy) falls through unchanged to the CPU-prefill + UploadKV bridge; the
		// claim is released first so the tower and CPU-prefill work never run while holding it.
		var resPrefill func() ([]float32, int, error)
		if causal {
			if pf, ok := m.resident.(Prefiller); ok && m.knobs.get(knobBatchedPrefill) != "0" {
				resPrefill = func() ([]float32, int, error) { return m.residentCausalImagePrefill(ctx, pf, ids, feats, spans) }
			}
		} else if rip, ok := m.resident.(ResidentImagePrefill); ok {
			resPrefill = func() ([]float32, int, error) { return m.residentImagePrefill(ctx, rip, ids, feats, spans) }
		}
		if resPrefill != nil && m.tryClaimResident() {
			if logits, gpuPos, ferr := resPrefill(); ferr == nil {
				g.ImgPrefillResident = true
				if capper, ok := m.resident.(ResidentCapped); ok {
					if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
						maxTokens = ctxCap - gpuPos
						g.BudgetClamped = true // the resident cap (not the request) bounds this turn
					}
				}
				g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
				sampler := NewSampler(sp)
				sampler.Observe(ids...)
				committed := false
				defer func() {
					if !committed {
						m.residentForgetIDs()
					}
					atomic.StoreInt32(&m.resBusy, 0)
				}()
				generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
					l, err := m.resident.Forward(m.embedResident(next), gpuPos)
					gpuPos++
					return l, err
				})
				if g.err == nil {
					m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
					committed = true
				}
				return
			} else {
				g.ImgPrefillDecline = ferr.Error()
			}
			atomic.StoreInt32(&m.resBusy, 0) // declined: release before falling through
		}

		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsVLSpans(ctx, ids, spans, feats, cache, causal)
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		if m.tryClaimResident() {
			// The resident cache is about to hold this turn's image content. If decode completes naturally, residentCommitIDs
			// below records it and this defer's forget is skipped (committed=true); any other exit (error, cancel, or the
			// claim or upload never engaging) forgets unconditionally, so a later caller never trusts a stale or half-written
			// record.
			defer func() {
				if !committed {
					m.residentForgetIDs()
				}
				atomic.StoreInt32(&m.resBusy, 0)
			}()
			if uerr := m.residentUploadPrefill(cache); uerr == nil {
				useGPU = true
				gpuPos = len(ids)
				if capper, ok := m.resident.(ResidentCapped); ok {
					if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
						maxTokens = ctxCap - gpuPos // may clamp to 0 (prompt filled the cap)
						g.BudgetClamped = true      // the resident cap (not the request) bounds this turn
					}
				}
			}
		}

		g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
		sampler := NewSampler(sp)
		sampler.Observe(ids...) // repetition penalties see the whole prompt
		generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
			if useGPU {
				l, err := m.resident.Forward(m.embedResident(next), gpuPos)
				gpuPos++
				return l, err
			}
			return m.forward(next, cache)
		})
		if useGPU && g.err == nil {
			m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
			committed = true
		}
	}()
	return out, g
}

// GenerateQwenVL streams a continuation for a Qwen2.5-VL multimodal prompt. Like GenerateVL, but the merged vision
// features (the ViT+merger output, [imgLen*HiddenDim]) replace the <image> run at [imgPos,imgPos+imgLen), and rotary
// positions are m-RoPE, computed from the image grid(s) (gridTHW, t/h/w in patch units; merge = spatial_merge_size;
// imageToken = the placeholder id). Image tokens attend causally (Qwen's bidirectionality is in the ViT). Decode past
// the prompt resumes scalar positions at the block max + 1 (the cache's m-RoPE delta). imgHash and the lazy features:
// see GenerateVL.
//
// Resident GPU decode additionally needs ResidentMRoPE, not only the base ResidentForward GenerateVL uses: decode past
// an image block needs the rope-angle position (pos+mropeDelta) and the KV/attention position (pos) to differ, which
// only ForwardMRoPE can express. A resident backend without it falls back to the CPU decode loop as if no resident were
// configured, including for image reuse: that fast path is gated on ResidentMRoPE too, for the same reason.
func (m *Model) GenerateQwenVL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.GenerateQwenVLDeepstack(ctx, ids, imgPos, imgLen, imgHash, func() ([]float32, [][]float32, error) {
		f, err := features()
		return f, nil, err
	}, gridTHW, merge, imageToken, maxTokens, sp)
}

// GenerateQwenVLDeepstack is GenerateQwenVL for a tower that also returns DeepStack sets (Qwen3-VL): features returns
// the merged rows and, per early decoder layer, a set of imgLen rows the prefill adds to the image positions' hidden
// state after that layer. nil sets are GenerateQwenVL. The sets touch only the prefill: a resident whose m-RoPE prefill
// cannot inject them is not offered it, and the CPU prefill carries them before the usual upload.
func (m *Model) GenerateQwenVLDeepstack(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, [][]float32, error), gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.GenerateQwenVLDeepstackSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen, Hash: imgHash}}, features, gridTHW, merge, imageToken, maxTokens, sp)
}

// GenerateQwenVLDeepstackSpans is GenerateQwenVLDeepstack for several images: spans are the images' placeholder runs in
// prompt order and gridTHW holds one (t, h, w) per image in the same order, each image its own t = 1 grid, never a pair
// merged into one (docs/multimodal.md, "Do not pair images"). features returns every span's merged rows concatenated in
// span order and, per DeepStack layer, a set laid out the same way.
func (m *Model) GenerateQwenVLDeepstackSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, [][]float32, error), gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		if len(spans) == 0 || len(gridTHW) != len(spans) {
			g.err = fmt.Errorf("decoder: %d image spans and %d grids; each image needs its own grid", len(spans), len(gridTHW))
			return
		}
		lastEnd := spans[len(spans)-1].Pos + spans[len(spans)-1].Len
		mropePos, err := mropePositions(ids, imageToken, gridTHW, merge)
		if err != nil {
			g.err = err
			return
		}
		// mropeDelta only depends on the placeholder run and the image grid's own dimensions —
		// never the tower's output — so it's available up front, before any reuse decision.
		mropeDelta := mropeDelta(mropePos, len(ids))

		// A recurrent family (the Gated DeltaNet hybrids) takes no resident reuse branch and no CPU-prefill-then-UploadKV bridge
		// here, and must be refused them, not merely not engaged: every resident executor implements ResidentMRoPE, so the type
		// assertions below succeed on a CUDA-resident qwen3.5, and the bridge copies only layers that have KV. A DeltaNet layer
		// has none, so resident decode would start from a zeroed recurrent state, with no error and wrong tokens. Its one
		// resident route is the resident m-RoPE prefill, taken only when the resident says it builds the recurrent state itself
		// (ResidentHybridMRoPEPrefill); otherwise an image turn on a recurrent family is CPU prefill + CPU decode. See the
		// docs/multimodal.md P8 record, item 5.
		recurrent := m.hasRecurrentState()

		if r, ok := m.resident.(ResidentMRoPE); ok && !recurrent && m.tryClaimResident() {
			claim := residentImageClaimsOf(spans)
			// Same cap decline as GenerateVL's reuse site: a prompt in (ResidentContextCap, MaxPositions) already passed prepare's
			// MaxPositions check, so residentPrefillSeedMRoPE below would die mid-prefill with no CPU fallback. Decline up front
			// and fall through to the "not fully reused" release and re-claim path.
			if reuseFrom := m.residentReuseLen(ids, claim, nil); reuseFrom >= lastEnd {
				if ctxCap := m.ResidentContextCap(); ctxCap <= 0 || len(ids) <= ctxCap {
					g.PrefillReused = reuseFrom // observable proof the fast path actually fired
					m.residentForgetIDs()
					logits, err := m.residentPrefillSeedMRoPE(ctx, r, ids, reuseFrom, mropeDelta)
					if err != nil {
						atomic.StoreInt32(&m.resBusy, 0)
						g.err = err
						return
					}
					gpuPos := len(ids)
					if capper, ok := m.resident.(ResidentCapped); ok {
						if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
							maxTokens = ctxCap - gpuPos
							g.BudgetClamped = true // the resident cap (not the request) bounds this turn
						}
					}
					g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
					sampler := NewSampler(sp)
					sampler.Observe(ids...)
					generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
						ropePos := gpuPos + mropeDelta
						l, err := r.ForwardMRoPE(m.embedResident(next), gpuPos, ropePos)
						gpuPos++
						return l, err
					})
					if g.err == nil {
						m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
					} else {
						m.residentForgetIDs()
					}
					atomic.StoreInt32(&m.resBusy, 0)
					return
				}
			}
			atomic.StoreInt32(&m.resBusy, 0)
		}

		feats, deep, err := features()
		if err == nil {
			err = m.checkDeepstack(deep, imageSpansTotal(spans))
		}
		if err != nil {
			g.err = err
			return
		}

		// Resident m-RoPE prefill fast path (Qwen2.5-VL's twin of GenerateVL's resident image-prefill branch), tried before the
		// CPU prefillLogitsQwenVL call: a cold image turn's prefill, not only its decode, runs on the resident GPU when the
		// backend implements ResidentMRoPEPrefill. It needs that and ResidentMRoPE (this path's decode continuation via
		// ForwardMRoPE): a resident that can prefill but not decode m-RoPE would strand the turn right after prefill. Any
		// decline (no capability, prompt too large for one chunk, resident busy) falls through unchanged to the CPU-prefill +
		// UploadKV bridge; the claim is released first so the CPU prefill never runs while holding it.
		//
		// A recurrent family joins this branch only if the resident says its prefill builds the recurrent state itself
		// (ResidentHybridMRoPEPrefill). It has no reuse fast path above and no UploadKV bridge below, and the resident's
		// recorded ids are forgotten before the attempt: a prefill that fails partway has already overwritten the recurrent
		// state, and a stale record of the previous conversation would let a later strict-extension reuse (residentReuseLen)
		// continue from that garbage.
		hybridOK := false
		if recurrent {
			hp, isHP := m.resident.(ResidentHybridMRoPEPrefill)
			hybridOK = isHP && hp.HybridMRoPEPrefill()
		}
		_, deepOK := m.resident.(ResidentMRoPEDeepstackPrefill) // a turn with DeepStack sets needs a resident that injects them
		if rmp, ok := m.resident.(ResidentMRoPEPrefill); ok && (!recurrent || hybridOK) && (deep == nil || deepOK) {
			if r, ok2 := m.resident.(ResidentMRoPE); ok2 && m.tryClaimResident() {
				if recurrent {
					m.residentForgetIDs()
				}
				if logits, gpuPos, ferr := m.residentMRoPEPrefillDeep(ctx, rmp, ids, feats, spans, mropePos, deep); ferr == nil {
					g.ImgPrefillResident = true
					if capper, ok := m.resident.(ResidentCapped); ok {
						if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
							maxTokens = ctxCap - gpuPos
							g.BudgetClamped = true // the resident cap (not the request) bounds this turn
						}
					}
					g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
					sampler := NewSampler(sp)
					sampler.Observe(ids...)
					committed := false
					defer func() {
						if !committed {
							m.residentForgetIDs()
						}
						atomic.StoreInt32(&m.resBusy, 0)
					}()
					generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
						ropePos := gpuPos + mropeDelta
						l, err := r.ForwardMRoPE(m.embedResident(next), gpuPos, ropePos)
						gpuPos++
						return l, err
					})
					if g.err == nil {
						m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
						committed = true
					}
					return
				} else {
					g.ImgPrefillDecline = ferr.Error()
				}
				atomic.StoreInt32(&m.resBusy, 0) // declined: release before falling through
			}
		}

		cache := m.NewCache(len(ids) + maxTokens)
		if deep != nil {
			cache.deepstack = &deepstackRows{spans: spans, rows: deep}
		}
		logits, err := m.prefillLogitsQwenVLSpans(ctx, ids, spans, feats, mropePos, cache)
		cache.deepstack = nil // the prefill's alone; nothing after it may see the sets
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		var mrope ResidentMRoPE
		// Gate the resBusy claim on ResidentMRoPE support BEFORE claiming — a resident that
		// cannot do Qwen decode (no m-RoPE support) shouldn't contend with a concurrent
		// plain-text Generate for a claim it can't use. Safe on a nil m.resident: a type
		// assertion on a nil interface value just reports ok=false.
		if r, ok := m.resident.(ResidentMRoPE); ok && !recurrent && m.tryClaimResident() {
			defer func() {
				if !committed {
					m.residentForgetIDs() // see GenerateVL's identical comment
				}
				atomic.StoreInt32(&m.resBusy, 0)
			}()
			if uerr := m.residentUploadPrefill(cache); uerr == nil {
				useGPU = true
				mrope = r
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
				ropePos := gpuPos + mropeDelta
				l, err := mrope.ForwardMRoPE(m.embedResident(next), gpuPos, ropePos)
				gpuPos++
				return l, err
			}
			return m.forward(next, cache) // decode m-RoPE via cache.mropeDelta
		})
		if useGPU && g.err == nil {
			m.residentCommitIDs(ids, generated, residentImageBlocksOf(spans), nil)
			committed = true
		}
	}()
	return out, g
}

// checkImageBlockFitsWindow refuses an image block longer than the sliding window (docs/multimodal.md, F3). The
// bidirectional image-block mask this path uses (KVCache.attendHi, and the resident image prefill that mirrors it)
// bounds a sliding layer's keys by the query's own window and never extends it back to the block's start, so a longer
// block would under-attend its own image. Gemma 4's own path applies the correction and does not come here. A
// refusal by name instead of a quietly wrong answer.
func (m *Model) checkImageBlockFitsWindow(imgLen int) error {
	if w := m.w.arch.SlidingWindow; w > 0 && imgLen > w {
		return fmt.Errorf("decoder: an image block of %d tokens is longer than this model's %d-token sliding window, which the bidirectional image mask does not support (docs/multimodal.md F3)", imgLen, w)
	}
	return nil
}

// checkDeepstack validates GenerateQwenVLDeepstack's sets: at most one per decoder layer, each imgLen rows of hidden
// (every image's rows, for several images).
func (m *Model) checkDeepstack(deep [][]float32, imgLen int) error {
	if len(deep) > m.w.arch.NumLayers {
		return fmt.Errorf("decoder: %d DeepStack sets for %d layers", len(deep), m.w.arch.NumLayers)
	}
	for l, set := range deep {
		if len(set) != imgLen*m.w.arch.HiddenDim {
			return fmt.Errorf("decoder: DeepStack set %d has %d values, want %d x %d", l, len(set), imgLen, m.w.arch.HiddenDim)
		}
	}
	return nil
}
