package serveapp

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// engine is what the serving shell asks for a text generation (docs/tasks/task-subsystems-2026-10.md, SB7): start
// begins one over a prepared request and streams its token ids. report, read once tokens has closed, is the
// generation's account (its budget clamp, logprobs, prefix reuse, error); nil from an engine that keeps none. done is
// called once, after the stream has been read to its end.
type engine interface {
	start(ctx context.Context, gr genRequest) (tokens <-chan int, report *decoder.Generation, done func())
}

// engine is the model's: the scripted seam when one is set, else the decoder.
func (lm *loadedModel) engine() engine {
	if lm.scripted != nil {
		return scriptedEngine{lm}
	}
	return decoderEngine{lm}
}

// scriptedEngine streams lm.scripted (at most the request's max tokens) and keeps no report. A test seam.
type scriptedEngine struct{ lm *loadedModel }

func (e scriptedEngine) start(_ context.Context, gr genRequest) (<-chan int, *decoder.Generation, func()) {
	lm := e.lm
	if lm.onScripted != nil {
		lm.onScripted(gr.promptIDs)
	}
	if lm.onScriptedScope != nil {
		lm.onScriptedScope(gr.kvScope())
	}
	return scriptedStream(lm.scripted, gr.maxTokens), nil, func() {}
}

// decoderEngine generates on lm.model: the GPU-resident paths, or a CPU session with its speculative variants.
type decoderEngine struct{ lm *loadedModel }

func (e decoderEngine) start(ctx context.Context, gr genRequest) (stream <-chan int, gen *decoder.Generation, done func()) {
	lm := e.lm
	// GPU-resident models take the stateless path. Generate engages the resident DecodeRunner only with no session
	// commit and no prefix reuse (decoder/model.go's useGPU): the resident's KV lives on the GPU and a session's
	// prefix-reuse cache is CPU-side, so the two cannot both be the source of truth, and going through a session
	// would silently drop the request to the staged CPU path. The OpenAI API is stateless (the client resends the
	// whole conversation), so sessions here are only a TTFT optimisation and trading prefix reuse for resident decode
	// is safe.
	//
	// N-gram speculative decode does mix with residency: the drafter is pure Go and the verify is the resident
	// batched ForwardN. genNgramInto claims the shared resident KV (resBusy) like Generate. Constrained and tool
	// requests (grammar masker) keep plain resident Generate; a validation error (sampler not on the spec path) falls
	// back to plain Generate before the KV is touched, so the fallback is exact.
	//
	// Adapter (compute-time LoRA) models must not take this path: the LoRA is applied only through the session
	// binding (sessionLRU.bindAdapter, Session.UseAdapter), and the stateless Generate runs on a fresh cache with no
	// LoRA, so it would silently return base-model output. Adapter requests take the session path below. There a
	// session's first turn can still run on the resident GPU path (generateInto's useGPU, decoder/model.go); a later
	// turn continuing off the reused warm prefix drops to CPU, since compute-time LoRA is not wired into the resident
	// prefix-reuse path.
	if lm.model.ResidentActive() && lm.adapter == "" {
		// The stable section of the prompt, for the prefix snapshot taken during a cold prefill (kv_boundary.go). Only the plain resident Generate reads it.
		gctx := decoder.WithKVScope(ctx, gr.kvScope())
		if lm.blockSpec != nil && gr.masker == nil {
			// Pretrained block drafter (--drafter): a whole block per round, verified in one batched pass.
			// GenerateStream validates the sampler and returns an error before touching any state, so a request with
			// temperature or penalties falls back exactly.
			if s, gn, err := lm.blockSpec.GenerateStream(ctx, gr.promptIDs, gr.maxTokens, gr.sp); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = lm.model.Generate(gctx, gr.promptIDs, gr.maxTokens, gr.sp)
			}
		} else if lm.spec && gr.masker == nil {
			if s, gn, err := lm.model.GenerateNgramSpeculativeAdaptive(ctx, gr.promptIDs, gr.maxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, gr.sp); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = lm.model.Generate(gctx, gr.promptIDs, gr.maxTokens, gr.sp)
			}
		} else {
			stream, gen = lm.model.Generate(gctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
		return stream, gen, func() {}
	}

	// Reuse the KV of whichever cached session already holds this prompt as a prefix (continuing chat, agent loop):
	// only the new suffix is prefilled. The session is checked out for this generation: sessMu covers the LRU
	// operation only, and checkin (done) runs once the shell has drained the stream (the generation goroutine
	// reconciles the session before closing it), so the next acquire, or a background demote or save, sees a
	// consistent session.
	lm.sessMu.Lock()
	sess := lm.sessions.acquire(gr.promptIDs)
	lm.sessMu.Unlock()
	done = func() {
		lm.sessMu.Lock()
		lm.sessions.checkin(sess)
		lm.sessMu.Unlock()
	}
	switch {
	case gr.conf != nil:
		// goinfer_confidence needs plain constrained decode: the speculative paths below drive the masker without
		// Process (grammar-fused) or batch verify positions past what is accepted (n-gram), and the capture records
		// in Process, one call per emitted position.
		stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
	case lm.spec && gr.masker != nil && gr.sp.Temperature == 0:
		// Constrained request (response_format or tool grammar), greedy: grammar-fused speculative decode. A
		// RouterDrafter fuses the grammar's forced byte-run (structural tokens) with an n-gram copy of free values
		// that echo the context; the masked verify keeps output identical to constrained Generate, and a miss costs
		// almost nothing. Any validation error, which precedes touching the session cache, falls back to plain
		// constrained decode.
		spSpec := gr.sp
		spSpec.LogitProcessor = nil // the verify applies the grammar mask itself
		drafter := &decoder.RouterDrafter{Sources: []decoder.Drafter{
			&decoder.GrammarDrafter{Mask: gr.masker, Encode: func(s string) []int { ids, _ := lm.tk.Encode(s, false); return ids }},
			&decoder.NgramDrafter{},
		}}
		var err error
		stream, gen, err = sess.GenerateGrammarSpeculative(ctx, gr.promptIDs, gr.maxTokens, gr.masker, drafter, 8, spSpec)
		if err != nil {
			stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
	case lm.spec:
		// Lossless n-gram speculative decode with adaptive depth. Falls back to plain Generate when the request's
		// sampler is not supported on the spec path (repetition, presence or frequency penalties, logit bias, or a
		// constrained or tool LogitProcessor with temperature>0); the validation error precedes touching the session
		// cache, so the fallback is exact.
		var err error
		stream, gen, err = sess.GenerateNgramSpeculativeAdaptive(ctx, gr.promptIDs, gr.maxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, gr.sp)
		if err != nil {
			stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
	default:
		stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
	}
	return stream, gen, done
}
