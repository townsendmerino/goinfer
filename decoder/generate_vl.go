package decoder

import (
	"context"
	"sync/atomic"
)

// vlDecodeLoop is the shared sampler/stream/stop decode loop behind GenerateVL/GenerateQwenVL —
// factored out so the four control-flow branches those two functions need (CPU decode, resident
// decode, and — P9a, docs/multimodal.md — each of those again inside the image-reuse fast path)
// don't each carry their own copy. logits is the seed (from prefill or from the reused prefix's
// last resident position); forward advances exactly one token (its own closure decides CPU vs
// resident vs resident-m-RoPE). Returns every token actually streamed, for the caller's own
// commit/forget decision — g.err is set on any early exit, exactly as each inlined loop did
// before this was factored out.
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
		// Select on ctx.Done so a consumer that stops ranging can't wedge this
		// goroutine forever on a bare send (holding the KV cache) — M8.
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

// GenerateVL streams a continuation for a multimodal (vision-language) prompt.
// `ids` are the text token ids with a run of `imgLen` image-placeholder ids
// starting at `imgPos`; `imgHash` is a content hash of the raw image bytes behind
// that run (P9a, docs/multimodal.md — used for resident prefix-reuse; a caller
// with no reuse story of its own may pass 0, which just never matches). `features`
// is invoked AT MOST ONCE, and only when the image cannot be fully reused from the
// resident KV — a lazy closure specifically so an unchanged, resent screenshot
// never re-runs the vision tower at all.
//
// Prefill runs through the bidirectional image-block mask (prefillLogitsVL) — image
// tokens attend mutually, text attends causally, and tokens decoded after the image
// attend causally too — then decodes up to maxTokens exactly like Generate (same
// Sampler, LogitProcessor, and stop rule).
//
// Resident GPU decode (gap 0, docs/multimodal.md): claims the shared resident KV
// when available and not busy with a concurrent generation; falls back to the CPU
// decode loop otherwise. Image-aware prefix reuse (P9a): before paying for the tower
// or the CPU prefill at all, checks whether this turn's image (and everything
// before it) is already sitting in the resident KV from a prior turn — if so,
// decode resumes directly from there with no re-tower, no re-prefill, no re-upload.
// Any other outcome (no resident, a different image, busy) falls through to the
// unconditional full-prefill path, unchanged from before P9a. The caller owns the
// returned channel: range over it to consume tokens, then check Generation.Err for
// a terminal error.
func (m *Model) GenerateVL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)

		// P9a fast path: peek the resident KV for a full-image reuse before touching the
		// tower or the CPU prefill. Held only long enough to check+use; released either way.
		if m.tryClaimResident() {
			claim := []residentImageClaim{{Start: imgPos, Len: imgLen, Hash: imgHash}}
			if reuseFrom := m.residentReuseLen(ids, claim); reuseFrom >= imgPos+imgLen {
				// FULL reuse: the image, and everything before it, is already resident.
				// Stay inside THIS claim (no release/reclaim) — reseed via the ordinary
				// ResidentForward.Forward path (no CPU prefill, no tower call at all).
				g.PrefillReused = reuseFrom // observable proof the fast path actually fired
				m.residentForgetIDs()       // forget first — from here the cache is mid-write
				logits, err := m.residentPrefillSeed(ctx, ids, reuseFrom)
				if err != nil {
					atomic.StoreInt32(&m.resBusy, 0)
					g.err = err
					return
				}
				gpuPos := len(ids)
				if capper, ok := m.resident.(ResidentCapped); ok {
					if ctxCap := capper.ContextCap(); ctxCap > 0 && gpuPos+maxTokens > ctxCap {
						maxTokens = ctxCap - gpuPos
					}
				}
				sampler := NewSampler(sp)
				sampler.Observe(ids...)
				generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
					l, err := m.resident.Forward(m.embedResident(next), gpuPos)
					gpuPos++
					return l, err
				})
				if g.err == nil {
					m.residentCommitIDs(ids, generated, &residentImageBlock{start: imgPos, end: imgPos + imgLen, hash: imgHash})
				} else {
					m.residentForgetIDs()
				}
				atomic.StoreInt32(&m.resBusy, 0)
				return
			}
			// Not fully reused: release immediately — do not hold the claim across the
			// tower call below.
			atomic.StoreInt32(&m.resBusy, 0)
		}

		// Ordinary path (unchanged shape from before P9a): tower runs outside any claim,
		// full CPU prefill (always from=0 — this branch never partially reuses), re-claim
		// fresh for upload+decode only.
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsVL(ctx, ids, feats, imgPos, imgLen, cache)
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		if m.tryClaimResident() {
			// The resident cache is about to hold THIS turn's (image) content. If decode
			// completes naturally, residentCommitIDs below records it and this defer's
			// forget is skipped (committed=true); any other exit (error, cancel, or the
			// claim/upload never engaging at all) forgets unconditionally, so a later
			// caller never trusts a stale or half-written record (V-11,
			// docs/review-2026-09-04.md — the same discipline gap 0 already established
			// here, now with a real commit path alongside it).
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
					}
				}
			}
		}

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
			m.residentCommitIDs(ids, generated, &residentImageBlock{start: imgPos, end: imgPos + imgLen, hash: imgHash})
			committed = true
		}
	}()
	return out, g
}

// GenerateQwenVL streams a continuation for a Qwen2.5-VL multimodal prompt. Like
// GenerateVL but for the Qwen family: the merged vision features (the ViT+merger
// output, [imgLen*HiddenDim]) replace the <image> run at [imgPos,imgPos+imgLen),
// and rotary positions are m-RoPE — computed from the image grid(s) (gridTHW, t/h/w
// in patch units; merge = spatial_merge_size; imageToken = the placeholder id).
// Image tokens attend CAUSALLY (no bidirectional block — Qwen's bidirectionality is
// in the ViT). Decode past the prompt resumes scalar positions at the block max + 1
// (handled by the cache's m-RoPE delta). `imgHash`/lazy `features`: see GenerateVL.
//
// Resident GPU decode (gap 0) additionally needs decoder.ResidentMRoPE here (not just
// the base ResidentForward GenerateVL uses): decode past an image block needs the
// rope-angle position (pos+mropeDelta) and the KV-cache/attention position (pos) to
// differ, which only ResidentMRoPE's ForwardMRoPE can express — see that interface's
// doc comment. A resident backend without it (e.g. Metal, which also lacks UploadKV)
// falls back to the CPU decode loop exactly as if no resident were configured at all
// — including for image-reuse (P9a): the fast path below is gated on ResidentMRoPE
// support too, for the identical reason.
func (m *Model) GenerateQwenVL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		mropePos, err := mropePositions(ids, imageToken, gridTHW, merge)
		if err != nil {
			g.err = err
			return
		}
		// mropeDelta only depends on the placeholder run and the image grid's own dimensions —
		// never the tower's output — so it's available up front, before any reuse decision.
		mropeDelta := mropeDelta(mropePos, len(ids))

		if r, ok := m.resident.(ResidentMRoPE); ok && m.tryClaimResident() {
			claim := []residentImageClaim{{Start: imgPos, Len: imgLen, Hash: imgHash}}
			if reuseFrom := m.residentReuseLen(ids, claim); reuseFrom >= imgPos+imgLen {
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
					}
				}
				sampler := NewSampler(sp)
				sampler.Observe(ids...)
				generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
					ropePos := gpuPos + mropeDelta
					l, err := r.ForwardMRoPE(m.embedResident(next), gpuPos, ropePos)
					gpuPos++
					return l, err
				})
				if g.err == nil {
					m.residentCommitIDs(ids, generated, &residentImageBlock{start: imgPos, end: imgPos + imgLen, hash: imgHash})
				} else {
					m.residentForgetIDs()
				}
				atomic.StoreInt32(&m.resBusy, 0)
				return
			}
			atomic.StoreInt32(&m.resBusy, 0)
		}

		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsQwenVL(ctx, ids, feats, imgPos, imgLen, mropePos, cache)
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
		if r, ok := m.resident.(ResidentMRoPE); ok && m.tryClaimResident() {
			defer func() {
				if !committed {
					m.residentForgetIDs() // see GenerateVL's identical comment — same V-11 discipline
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
					}
				}
			}
		}

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
			m.residentCommitIDs(ids, generated, &residentImageBlock{start: imgPos, end: imgPos + imgLen, hash: imgHash})
			committed = true
		}
	}()
	return out, g
}
