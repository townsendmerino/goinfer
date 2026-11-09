package engine

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// SessionPool hands out the CPU sessions a Local engine generates on: Acquire checks out the session that best extends
// prompt (a fresh one when none does), and Checkin returns it once its generation's stream has drained. The pool does
// its own locking.
type SessionPool interface {
	Acquire(prompt []int) *decoder.Session
	Checkin(s *decoder.Session)
}

// Local is the Engine a loaded model generates on in this process: the GPU-resident paths, or a session from Pool with
// its speculative variants.
type Local struct {
	Model        *decoder.Model     // the loaded model; nil answers as decoder's methods do for a nil *Model
	Pool         SessionPool        // the CPU sessions
	Drafter      *decoder.BlockSpec // an attached pretrained block drafter, or nil
	Speculative  bool               // n-gram speculative decoding is on
	AdapterBound bool               // a compute-time LoRA adapter is bound, so generation goes through a session
	Encode       func(string) []int // the tokenizer, for the grammar drafter's forced runs

	media mediaModel // a test's fake for Media requests; nil means Model
}

// Start begins one generation on d.
func (d Local) Start(ctx context.Context, req Request) *Stream {
	if req.Lazy != nil && req.Masker == nil && req.Media == nil && !req.NoSpeculation && (d.Drafter != nil || d.Speculative) {
		return lazyTurn(decoder.WithKVScope(ctx, req.Scope), req, req.Lazy, d.lazyFree, d.lazyConstrained) // the scope plain resident Generate takes
	}
	return GenerationStream(d.generate(ctx, req))
}

// resident reports whether d generates on the GPU-resident stateless path rather than through a session (generate's first branch).
func (d Local) resident() bool { return d.Model.ResidentActive() && !d.AdapterBound }

// lazyFree is a lazy-union turn's phase 1: d's speculative path, the block drafter or n-gram speculation, with no fallback (an error
// sends the whole turn to lazyConstrained, as a non-speculating server runs it).
func (d Local) lazyFree(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
	if d.resident() {
		if dr := d.Drafter; dr != nil {
			s, g, err := dr.GenerateStream(cost.Off(ctx), prompt, max, sp)
			return s, g, nil, err
		}
		s, g, err := d.Model.GenerateNgramSpeculativeAdaptive(cost.Off(ctx), prompt, max, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, sp)
		return s, g, nil, err
	}
	sess := d.Pool.Acquire(prompt)
	s, g, err := sess.GenerateNgramSpeculativeAdaptive(cost.Off(ctx), prompt, max, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, sp)
	if err != nil {
		d.Pool.Checkin(sess)
		return nil, nil, nil, err
	}
	return s, g, func() { d.Pool.Checkin(sess) }, nil
}

// lazyConstrained is a lazy-union turn's phase 2, sp carrying the lazy masker: plain gated Generate, on the resident when d is resident,
// else on a session. Not the CPU session's grammar-fused speculation, which needs a grammar live from token 1 and cannot disarm after a
// call, as the lazy masker does.
func (d Local) lazyConstrained(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
	if d.resident() {
		s, g := d.Model.Generate(ctx, prompt, max, sp)
		return s, g, nil, nil
	}
	sess := d.Pool.Acquire(prompt)
	s, g := sess.Generate(ctx, prompt, max, sp)
	return s, g, func() { d.Pool.Checkin(sess) }, nil
}

// A cost counter on ctx (internal/cost, a request that asked for its cost) reaches only the plain Generate paths, which count;
// the media, drafted and speculative paths get it turned off, so the request's report is empty there, never partial.
func (d Local) generate(ctx context.Context, req Request) (stream <-chan int, gen *decoder.Generation, done func()) {
	if req.Media != nil {
		var m mediaModel = d.Model
		if d.media != nil {
			m = d.media
		}
		stream, gen = generateMedia(cost.Off(ctx), m, req)
		return stream, gen, nil
	}
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
	// binding (Session.UseAdapter), and the stateless Generate runs on a fresh cache with no
	// LoRA, so it would silently return base-model output. Adapter requests take the session path below. There a
	// session's first turn can still run on the resident GPU path (generateInto's useGPU, decoder/model.go); a later
	// turn continuing off the reused warm prefix drops to CPU, since compute-time LoRA is not wired into the resident
	// prefix-reuse path.
	if d.Model.ResidentActive() && !d.AdapterBound {
		// The stable section of the prompt, for the prefix snapshot taken during a cold prefill (kv_boundary.go). Only the plain resident Generate reads it.
		gctx := decoder.WithKVScope(ctx, req.Scope)
		if dr := d.Drafter; dr != nil && req.Masker == nil {
			// Pretrained block drafter (--drafter): a whole block per round, verified in one batched pass.
			// GenerateStream validates the sampler and returns an error before touching any state, so a request with
			// temperature or penalties falls back exactly.
			if s, gn, err := dr.GenerateStream(cost.Off(ctx), req.PromptIDs, req.MaxTokens, req.Sampling); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = d.Model.Generate(gctx, req.PromptIDs, req.MaxTokens, req.Sampling)
			}
		} else if d.Speculative && req.Masker == nil {
			if s, gn, err := d.Model.GenerateNgramSpeculativeAdaptive(cost.Off(ctx), req.PromptIDs, req.MaxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, req.Sampling); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = d.Model.Generate(gctx, req.PromptIDs, req.MaxTokens, req.Sampling)
			}
		} else {
			stream, gen = d.Model.Generate(gctx, req.PromptIDs, req.MaxTokens, req.Sampling)
		}
		return stream, gen, func() {}
	}

	// Reuse the KV of whichever cached session already holds this prompt as a prefix (continuing chat, agent loop):
	// only the new suffix is prefilled. The session is checked out for this generation, and its check-in (done) runs once the shell has drained the stream (the generation goroutine
	// reconciles the session before closing it), so the next acquire, or a background demote or save, sees a
	// consistent session.
	sess := d.Pool.Acquire(req.PromptIDs)
	done = func() { d.Pool.Checkin(sess) }
	switch {
	case req.NoSpeculation:
		// goinfer_confidence needs plain constrained decode: the speculative paths below drive the masker without
		// Process (grammar-fused) or batch verify positions past what is accepted (n-gram), and the capture records
		// in Process, one call per emitted position.
		stream, gen = sess.Generate(ctx, req.PromptIDs, req.MaxTokens, req.Sampling)
	case d.Speculative && req.Masker != nil && req.Sampling.Temperature == 0:
		// Constrained request (response_format or tool grammar), greedy: grammar-fused speculative decode. A
		// RouterDrafter fuses the grammar's forced byte-run (structural tokens) with an n-gram copy of free values
		// that echo the context; the masked verify keeps output identical to constrained Generate, and a miss costs
		// almost nothing. Any validation error, which precedes touching the session cache, falls back to plain
		// constrained decode.
		spSpec := req.Sampling
		spSpec.LogitProcessor = nil // the verify applies the grammar mask itself
		drafter := &decoder.RouterDrafter{Sources: []decoder.Drafter{
			&decoder.GrammarDrafter{Mask: req.Masker, Encode: d.Encode},
			&decoder.NgramDrafter{},
		}}
		var err error
		stream, gen, err = sess.GenerateGrammarSpeculative(cost.Off(ctx), req.PromptIDs, req.MaxTokens, req.Masker, drafter, 8, spSpec)
		if err != nil {
			stream, gen = sess.Generate(ctx, req.PromptIDs, req.MaxTokens, req.Sampling)
		}
	case d.Speculative:
		// Lossless n-gram speculative decode with adaptive depth. Falls back to plain Generate when the request's
		// sampler is not supported on the spec path (repetition, presence or frequency penalties, logit bias, or a
		// constrained or tool LogitProcessor with temperature>0); the validation error precedes touching the session
		// cache, so the fallback is exact.
		var err error
		stream, gen, err = sess.GenerateNgramSpeculativeAdaptive(cost.Off(ctx), req.PromptIDs, req.MaxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, req.Sampling)
		if err != nil {
			stream, gen = sess.Generate(ctx, req.PromptIDs, req.MaxTokens, req.Sampling)
		}
	default:
		stream, gen = sess.Generate(ctx, req.PromptIDs, req.MaxTokens, req.Sampling)
	}
	return stream, gen, done
}
