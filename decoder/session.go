package decoder

import (
	"context"
	"fmt"
	"time"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// Session couples a KVCache with the token sequence currently materialized in
// it, so successive generations that share a leading token prefix reuse the
// prefix's KV instead of re-prefilling it. The win is largest for chat servers
// (each turn extends the prior prompt) and agent loops (a long, fixed system
// prompt + tool specs precede every call): only the divergent suffix is prefilled,
// turning an O(history) prefill into O(new tokens).
//
// Reuse is exact, not approximate. The KV at a position is a causal function of
// the tokens up to it, so an identical token prefix has identical KV — Session
// finds the longest common prefix with its stored tokens, rewinds the cache to it
// with TruncateTo, and prefills only the rest. There is no quality cost and the
// output is bit-identical to a full prefill.
//
// Not goroutine-safe: like the KVCache it owns, a Session backs one in-flight
// sequence at a time. Callers running one model across requests (e.g. cmd/serve)
// already serialize generations; a SessionLRU layers prefix-keyed reuse on top.
type Session struct {
	m      *Model
	cache  *KVCache
	tokens []int // the token sequence whose KV is live in cache (prompt + generated)

	// cleanCache: the cache was fully reset and nothing has run through it since, so a second TruncateTo(0) would find nothing
	// to do. True only from NewSession and Reset; rewindForReuse consumes it. The zero value is the safe side (a session
	// restored from a snapshot, and every session after its first generation, takes the full reset), and it is not a general
	// "the state is clean" claim: rewindForReuse also checks no tokens and pos 0, because tests and snapshot loaders build
	// caches by hand.
	cleanCache bool

	// kvAdapter is the compute-time adapter tokens' KV was built under — the cache-state grid's
	// adapter × prefix-reuse cell (cachestate.go): rewindForReuse goes cold when the adapter bound
	// now is a different one, instead of projecting a new turn through one fine-tune on top of a
	// prefix another built. LoadSession sets it from the snapshot (format v3 records the adapter).
	kvAdapter *loraRuntime

	// classes is which prefill class wrote each row the cache holds, so SharedHit offers another session only rows its
	// own cold prefill would compute. rewindForReuse clips it to the rows reused, so rows a generation writes carry no
	// class unless it records one (recordClasses); nil (a restored snapshot, a reset) offers nothing.
	classes []kvmgr.ClassRun
	// shared and copyDur are the rows the last CopyPrefixFrom copied in and its wall time, reported by the next
	// Generate (Generation.PrefillShared, KVCopy).
	shared  int
	copyDur time.Duration
}

// NewSession allocates an empty session. capHint pre-sizes the KV cache for an
// expected max sequence length (0 = grow on demand); the cache persists and is
// reused across this session's Generate calls.
func (m *Model) NewSession(capHint int) *Session {
	return &Session{m: m, cache: m.NewCache(capHint), cleanCache: true}
}

// Tokens returns the token sequence currently materialized in the cache (the
// last prompt plus everything generated from it). Read-only — callers must not
// mutate the returned slice. The SessionLRU keys prefix matching off it.
func (s *Session) Tokens() []int { return s.tokens }

// UseAdapter activates a compute-time LoRA adapter (Model.LoadAdapter) for this session's subsequent Generate calls, so one
// resident base serves many fine-tunes. Switching adapters changes the projections, so KV built under a different (or no)
// adapter is stale: the next Generate prefills cold rather than reusing it.
func (s *Session) UseAdapter(name string) error {
	rt := s.m.adapter(name) // locked read — LoadAdapter may mutate the map concurrently (audit C-29)
	if rt == nil {
		return fmt.Errorf("decoder: no compute-time adapter %q loaded", name)
	}
	s.cache.lora = rt
	return nil
}

// ClearAdapter reverts this session to the base (merged/no-adapter) projections.
func (s *Session) ClearAdapter() { s.cache.lora = nil }

// Reset drops all cached KV, returning the session to empty without freeing the
// backing arrays (so the next generation re-grows into them).
func (s *Session) Reset() {
	s.cache.TruncateTo(0)
	s.tokens = s.tokens[:0]
	s.classes = nil
	s.cleanCache = true
}

// rewindForReuse rewinds the cache to the longest reusable prefix shared with prompt and returns how many tokens to reuse.
// On an inexact rewind (a wrapped sliding-window ring cannot restore the positions it dropped) it resets and returns 0, so
// prefix reuse never reads stale history. Callers must skip it, and reconcile, for an empty prompt so a rejected call
// leaves a warm session untouched.
func (s *Session) rewindForReuse(prompt []int) (matched int) {
	defer func() { s.classes = clipClassRuns(s.classes, matched) }()
	// A just-created or Reset session has nothing for TruncateTo(0) to clear, and the reset would otherwise zero a recurrent
	// state that is already zero. Every guard fails toward resetting: the flag is consumed here whatever happens.
	clean := s.cleanCache && len(s.tokens) == 0 && s.cache.Pos() == 0
	s.cleanCache = false
	return kvmgr.RewindSequence(s.SequenceState(), kvmgr.Request{Root: s.sequenceRoot(s.cache.lora), Tokens: prompt}, clean)
}

// reconcile sets s.tokens to exactly what the cache holds after a generation. seq mirrors the cache (prompt + each
// committed token); on a prefill/forward error the cache may hold fewer positions than seq claims, so clamp, or the next
// call reuses KV that was never written and prefills at the wrong position.
func (s *Session) reconcile(seq []int) {
	var keep bool
	s.tokens, keep = sessionReconcile(s.cache.Pos(), seq, s.cache.hasRecurrentState, s.cache.TruncateTo)
	if keep {
		s.kvAdapter = s.cache.lora
	}
}

// sessionReconcile is the session's record after a turn (SB5): seq is the prompt plus every token the turn committed,
// pos the KV's position. A KV that ended short of seq (a turn cut off mid-write) is trusted only up to pos, unless it
// holds recurrent state, which cannot be rolled back and so starts over; a truncation that cannot be exact also starts
// over. It returns the tokens the KV now holds (nil: none) and whether they are live (the adapter they were built under
// is then recorded).
func sessionReconcile(pos int, seq []int, recurrent func() bool, truncate func(int) bool) ([]int, bool) {
	return kvmgr.ReconcileSequence(pos, seq, recurrent, truncate)
}

// Generate is Model.Generate with cross-call KV reuse. It rewinds the cache to
// the longest token prefix shared with the session's current state, prefills
// only the divergent suffix of prompt, decodes, and leaves the cache holding
// prompt + the generated tokens for the next call to extend. Streaming, sampling,
// stop handling, and the *Generation result match Model.Generate exactly.
func (s *Session) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}

	// Longest shared token prefix → reuse its KV (cold-prefill on an inexact rewind). An empty
	// prompt is a generateInto error: don't touch the cache (rewind/reconcile), so a warm session
	// survives a rejected call.
	matched := 0
	g.PrefillSlot, g.PrefillMatch = len(s.tokens), commonPrefixLen(s.tokens, prompt)
	if len(prompt) > 0 {
		matched = s.rewindForReuse(prompt)
	}
	// generateInto sets g.PrefillReused only on the resident path. A plain session takes its CPU/staged path, which never
	// touches it, so set it here or the API's usage.prefill_reused_tokens reports 0 for every session turn.
	g.PrefillReused = matched
	if s.shared > 0 && matched >= s.shared {
		g.PrefillShared, g.KVCopy = s.shared, s.copyDur
	}
	s.shared, s.copyDur = 0, 0
	batched := s.cache.lora == nil && len(prompt) > matched && s.m.canBatchN(len(prompt)-matched)

	// After prefill the cache holds the whole prompt; commit appends each generated token as its
	// forward lands it in the cache, so seq mirrors the cache exactly — including a clean rollback
	// if a forward errors mid-stream.
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	s.m.runGen(ctx, out, g, nil, func(ctx context.Context) {
		// A session always owns its cache, so generateInto's lazy cache allocation is never taken (newCache is nil).
		s.m.generateInto(ctx, out, g, s.cache, nil, prompt, matched, maxTokens, sp, commit)
		// Bookkeeping runs before the deferred close, so a consumer that observes the channel close
		// (and then starts the next request) always sees a reconciled session: tokens == what the
		// cache holds, no partial position.
		if len(prompt) > 0 {
			s.reconcile(seq)
			s.recordClasses(prompt, matched, batched)
		}
	})
	return out, g
}

// GenerateNgramSpeculative is Session.Generate with n-gram speculative decoding:
// it reuses the warm KV prefix exactly as Generate does (rewind to the longest
// shared token prefix, prefill only the divergent suffix), then runs the lossless
// speculative loop over the session's cache. Output matches Session.Generate in
// distribution (token-identical when greedy). The win is largest on agent/chat
// loops — a large warm prefix plus output that echoes the prompt is the n-gram
// drafter's home turf.
func (s *Session) GenerateNgramSpeculative(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	return s.genSpec(ctx, prompt, maxTokens, drafter, K, sp, nil)
}

// GenerateNgramSpeculativeAdaptive is GenerateNgramSpeculative with the 04
// adaptive-depth controller (nil ⇒ a fresh default).
func (s *Session) GenerateNgramSpeculativeAdaptive(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, ad *AdaptiveDepth, sp SamplingParams) (<-chan int, *Generation, error) {
	if ad == nil {
		ad = &AdaptiveDepth{}
	}
	if ad.Theta <= 0 { // unset by the caller: take this model's measured verify cost
		ad.Theta = s.m.verifyTheta()
	}
	ad.ensure()
	if ad.Theta >= 1 {
		// Theta >= 1 means Depth() would always return 0 (see Model.GenerateNgramSpeculativeAdaptive), so every round would pay
		// to draft and verify nothing: run plain Session.Generate, which reuses the warm prefix the same way.
		if err := validateNgramSpec(s.m, drafter, sp); err != nil {
			return nil, nil, err
		}
		ch, gen := s.Generate(ctx, prompt, maxTokens, sp)
		return ch, gen, nil
	}
	return s.genSpec(ctx, prompt, maxTokens, drafter, ad.MaxDraft, sp, ad)
}

// GenerateGrammarSpeculative is Session.Generate with grammar-masked speculative
// decoding (01/03) for constrained requests: it reuses the warm KV prefix exactly as
// Generate does, then runs the masked speculative loop over the session's cache. Output
// is identical to constrained Generate (token-identical greedy). The drafter is usually
// a RouterDrafter fusing the grammar's forced byte-run with an n-gram copy of free
// values; the mask keeps every position lossless regardless.
func (s *Session) GenerateGrammarSpeculative(ctx context.Context, prompt []int, maxTokens int, mask *constrain.Masker, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if err := validateGrammarSpec(s.m, mask, drafter, sp); err != nil {
		return nil, nil, err
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}

	matched := s.rewindForReuse(prompt) // cold-prefill on an inexact rewind
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	s.m.runGen(ctx, out, g, nil, func(ctx context.Context) {
		s.m.genGrammarInto(ctx, out, g, stats, mask, drafter, prompt, matched, maxTokens, K, sp, nil, s.cache, commit)
		s.reconcile(seq)
	})
	return out, g, nil
}

func (s *Session) genSpec(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams, ad *AdaptiveDepth) (<-chan int, *Generation, error) {
	if err := validateNgramSpec(s.m, drafter, sp); err != nil {
		return nil, nil, err
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}

	// An empty prompt is a genNgramInto error: don't rewind/reconcile the cache, so a rejected
	// call leaves a warm session's KV intact — matching Session.Generate.
	matched := 0
	if len(prompt) > 0 {
		matched = s.rewindForReuse(prompt) // cold-prefill on an inexact rewind
	}
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	s.m.runGen(ctx, out, g, nil, func(ctx context.Context) {
		s.m.genNgramInto(ctx, out, g, stats, drafter, prompt, matched, maxTokens, K, sp, nil, ad, s.cache, commit)
		// An empty prompt is a genNgramInto error: skip rewind and reconcile so a rejected call leaves a warm session's KV intact,
		// as Session.Generate does.
		if len(prompt) > 0 {
			s.reconcile(seq)
		}
	})
	return out, g, nil
}

// commonPrefixLen returns the length of the longest common leading run of a and b.
func commonPrefixLen(a, b []int) int {
	return kvmgr.CommonPrefixLen(a, b)
}
