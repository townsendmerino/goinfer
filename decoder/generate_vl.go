package decoder

import (
	"context"
	"sync/atomic"
)

// GenerateVL streams a continuation for a multimodal (vision-language) prompt.
// `ids` are the text token ids with a run of `imgLen` image-placeholder ids
// starting at `imgPos`; `features` ([imgLen*HiddenDim]) are the projected vision
// embeddings (the projector's output) that replace those placeholders' token
// embeddings. It prefills once through the bidirectional image-block mask
// (prefillLogitsVL) — image tokens attend mutually, text attends causally, and
// tokens decoded after the image attend causally too — then decodes up to
// maxTokens exactly like Generate (same Sampler, LogitProcessor, and stop rule).
//
// Stateless: a fresh cache, no warm-KV prefix reuse (a multimodal turn opts out of
// the session cache). Prefill always runs on CPU (the bidirectional image-block mask
// has no resident equivalent). Decode past the prefill runs on the resident GPU
// backend when one is available and not busy with a concurrent generation (gap 0,
// docs/multimodal.md — the hybrid CPU-prefill / resident-decode bridge,
// residentUploadPrefill) — otherwise it falls back to the same CPU decode loop this
// function has always used. The caller owns the returned channel: range over it to
// consume tokens, then check Generation.Err for a terminal error.
func (m *Model) GenerateVL(ctx context.Context, ids []int, features []float32, imgPos, imgLen, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	cache := m.NewCache(len(ids) + maxTokens)
	go func() {
		defer close(out)
		// Prefill: text embeddings with the projected vision features spliced in at
		// the placeholder run, under the bidirectional image-block mask. The last
		// position's logits seed the decode.
		logits, err := m.prefillLogitsVL(ctx, ids, features, imgPos, imgLen, cache)
		if err != nil {
			g.err = err
			return
		}

		// Resident GPU decode (gap 0): claim the shared resident KV, push this turn's
		// CPU-computed prefill into it (residentUploadPrefill), and decode on GPU from
		// there. Any claim failure or upload failure falls straight back to the CPU
		// decode loop below — zero behavior change for that path.
		useGPU := false
		gpuPos := 0
		if m.tryClaimResident() {
			// The resident cache is about to hold THIS turn's (image) content, which does
			// not match whatever resIDs described before this call — forget unconditionally
			// on the way out, whether or not the upload/decode below actually engages, so a
			// later plain Generate never trusts a stale prefix-reuse record against content
			// that isn't there (the exact bug shape V-11, docs/review-2026-09-04.md, fixed
			// once already — this is the first time since that fix either function touches
			// resident state again, so the same discipline applies from the start).
			defer func() {
				m.residentForgetIDs()
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
		var generated []int
		for range maxTokens {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			default:
			}
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			info, err := sampler.SampleWithInfo(logits)
			if err != nil {
				g.err = err
				return
			}
			next := info.ID
			if m.isStop(next, sp) {
				break
			}
			if sp.Logprobs {
				g.Logprobs = append(g.Logprobs, info)
			}
			// Select on ctx.Done so a consumer that stops ranging can't wedge this
			// goroutine forever on a bare send (holding the KV cache) — M8.
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			case out <- next:
			}
			generated = append(generated, next)
			// Decode steps after the image block are ordinary causal forwards (the
			// image-block mask only governs the image positions, which are now in KV) —
			// on GPU when resident engaged above, on CPU otherwise.
			if useGPU {
				if logits, err = m.resident.Forward(m.embedResident(next), gpuPos); err != nil {
					g.err = err
					return
				}
				gpuPos++
			} else if logits, err = m.forward(next, cache); err != nil {
				g.err = err
				return
			}
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
// (handled by the cache's m-RoPE delta). Stateless, like GenerateVL.
//
// Resident GPU decode (gap 0) additionally needs decoder.ResidentMRoPE here (not just
// the base ResidentForward GenerateVL uses): decode past an image block needs the
// rope-angle position (pos+mropeDelta) and the KV-cache/attention position (pos) to
// differ, which only ResidentMRoPE's ForwardMRoPE can express — see that interface's
// doc comment. A resident backend without it (e.g. Metal, which also lacks UploadKV)
// falls back to the CPU decode loop exactly as if no resident were configured at all.
func (m *Model) GenerateQwenVL(ctx context.Context, ids []int, features []float32, imgPos, imgLen int, gridTHW [][3]int, merge, imageToken, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		mropePos, err := mropePositions(ids, imageToken, gridTHW, merge)
		if err != nil {
			g.err = err
			return
		}
		cache := m.NewCache(len(ids) + maxTokens)
		logits, err := m.prefillLogitsQwenVL(ctx, ids, features, imgPos, imgLen, mropePos, cache)
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		mropeDelta := cache.mropeDelta
		var mrope ResidentMRoPE
		// Gate the resBusy claim on ResidentMRoPE support BEFORE claiming — a resident that
		// cannot do Qwen decode (no m-RoPE support) shouldn't contend with a concurrent
		// plain-text Generate for a claim it can't use. Safe on a nil m.resident: a type
		// assertion on a nil interface value just reports ok=false.
		if r, ok := m.resident.(ResidentMRoPE); ok && m.tryClaimResident() {
			defer func() {
				m.residentForgetIDs() // see GenerateVL's identical comment — same V-11 discipline
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
		var generated []int
		for range maxTokens {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			default:
			}
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			info, err := sampler.SampleWithInfo(logits)
			if err != nil {
				g.err = err
				return
			}
			next := info.ID
			if m.isStop(next, sp) {
				break
			}
			if sp.Logprobs {
				g.Logprobs = append(g.Logprobs, info)
			}
			// Select on ctx.Done so a consumer that stops ranging can't wedge this
			// goroutine forever on a bare send (holding the KV cache) — M8.
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			case out <- next:
			}
			generated = append(generated, next)
			if useGPU {
				ropePos := gpuPos + mropeDelta
				if logits, err = mrope.ForwardMRoPE(m.embedResident(next), gpuPos, ropePos); err != nil {
					g.err = err
					return
				}
				gpuPos++
			} else if logits, err = m.forward(next, cache); err != nil { // decode m-RoPE via cache.mropeDelta
				g.err = err
				return
			}
		}
	}()
	return out, g
}
