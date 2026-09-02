package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/constrain"
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
}

// NewSession allocates an empty session. capHint pre-sizes the KV cache for an
// expected max sequence length (0 = grow on demand); the cache persists and is
// reused across this session's Generate calls.
func (m *Model) NewSession(capHint int) *Session {
	return &Session{m: m, cache: m.NewCache(capHint)}
}

// Tokens returns the token sequence currently materialized in the cache (the
// last prompt plus everything generated from it). Read-only — callers must not
// mutate the returned slice. The SessionLRU keys prefix matching off it.
func (s *Session) Tokens() []int { return s.tokens }

// UseAdapter activates a compute-time LoRA adapter (Model.LoadAdapter, #7) for
// this session's subsequent Generate calls, so a single resident base serves many
// fine-tunes without paying its RAM per adapter. Switching adapters changes the
// projections, so any KV built under a different (or no) adapter is stale — the
// caller should Reset first if the cached prefix was produced by another adapter.
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
}

// rewindForReuse rewinds the cache to the longest reusable prefix shared with prompt and returns
// how many tokens to reuse. On an INEXACT rewind — a wrapped sliding-window ring can't restore the
// positions it dropped (C1) — it Resets and returns 0 (cold prefill), so prefix reuse never reads
// stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejected call
// leaves a warm session untouched.
func (s *Session) rewindForReuse(prompt []int) int {
	matched := max(min(commonPrefixLen(s.tokens, prompt), len(prompt)-1), 0)
	if !s.cache.TruncateTo(matched) {
		s.cache.TruncateTo(0) // drop everything; a full cold prefill re-populates the ring slots
		return 0
	}
	return matched
}

// reconcile sets s.tokens to EXACTLY what the cache holds after a generation. seq mirrors the cache
// (prompt + each committed token); on a prefill/forward error the cache may hold fewer positions
// than seq claims, so clamp — otherwise the next call "reuses" KV that was never written and
// prefills at the wrong position (M10).
func (s *Session) reconcile(seq []int) {
	// A forward that errored mid-sweep leaves the cache position behind the emitted token stream.
	p := s.cache.Pos()
	rolledBack := p < len(seq)
	if rolledBack {
		seq = seq[:p]
	}
	// Recurrent (Mamba-2 / Gated DeltaNet) rolling state is not positional: TruncateTo cannot rewind
	// it (it reports inexact), and a mid-sweep error can leave it over-advanced past the committed KV.
	// Clamping seq to cache.Pos() makes the truncate a no-op (exact), so the corrupt state would be
	// warm-reused on the next call and decode a new sequence from leaked state (C-01 class). On any
	// rollback, reset a recurrent session to cold so the next call re-prefills (audit R-14).
	if rolledBack && s.cache.hasRecurrentState() {
		s.cache.TruncateTo(0) // re-zeroes the rolling state (C-01)
		s.tokens = nil
		return
	}
	// CONSUME THE BOOL. TruncateTo reports whether the rewind was EXACT, and this line discarded it
	// — the same answer rewindForReuse above has always acted on.
	//
	// It is reachable, and G18 is what made it so (3a16a4b, 2026-08-25: the batched sweep aborts per
	// LAYER on a cancelled context). Layers below the abort point have already commitBatch'd, so
	// their ring count is startPos+K while c.pos is still startPos — advanceTo runs only at the end
	// of a completed sweep. Truncating to c.pos then rewinds those rings by K on a WRAPPED window,
	// which cannot restore the rows the commit evicted: ring.truncate returns false. Ignoring that
	// left the aborted turn's K/V — RoPE'd at positions startPos.. — physically resident, and
	// s.tokens pointing at the prefix, so the next request that extends the conversation matched,
	// rewound to a no-op "exact", and read those rows as history for EARLIER positions. Silently
	// wrong attention on every local layer below the abort point, and plausible text
	// (audit-2026-09-02 C-05). A cancelled request reports a clean end, so nothing else notices.
	//
	// Rings cannot restore evicted rows, so cold reset is the only exact answer — same remedy,
	// same reason, as rewindForReuse.
	if !s.cache.TruncateTo(len(seq)) {
		s.cache.TruncateTo(0)
		s.tokens = nil
		return
	}
	s.tokens = seq
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
	if len(prompt) > 0 {
		matched = s.rewindForReuse(prompt)
	}

	// After prefill the cache holds the whole prompt; commit appends each generated token as its
	// forward lands it in the cache, so seq mirrors the cache exactly — including a clean rollback
	// if a forward errors mid-stream.
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	go func() {
		defer close(out)
		s.m.generateInto(ctx, out, g, s.cache, prompt, matched, maxTokens, sp, commit)
		// Bookkeeping runs before the deferred close, so a consumer that observes the channel close
		// (and then starts the next request) always sees a reconciled session: tokens == what the
		// cache holds, no partial position.
		if len(prompt) > 0 {
			s.reconcile(seq)
		}
	}()
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

	matched := s.rewindForReuse(prompt) // cold-prefill on an inexact rewind (C1)
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	go func() {
		defer close(out)
		s.m.genGrammarInto(ctx, out, g, stats, mask, drafter, prompt, matched, maxTokens, K, sp, nil, s.cache, commit)
		s.reconcile(seq)
	}()
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
	// call leaves a warm session's KV intact — matching Session.Generate (N-01).
	matched := 0
	if len(prompt) > 0 {
		matched = s.rewindForReuse(prompt) // cold-prefill on an inexact rewind (C1)
	}
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	go func() {
		defer close(out)
		s.m.genNgramInto(ctx, out, g, stats, drafter, prompt, matched, maxTokens, K, sp, nil, ad, s.cache, commit)
		// Reconcile: seq == prompt + every token committed to the cache, so the session's token
		// list mirrors the cache exactly for the next call's prefix match — clamped to what the
		// cache actually holds if a forward errored (the final pending token was emitted but not
		// committed — one behind, same as a fresh prefill would leave it). Skip on an empty prompt:
		// genNgramInto rejected it without touching the cache, so reconcile(seq=[]) would TruncateTo(0)
		// and wipe a warm session's KV — the same guard Session.Generate has (audit R-13 / N-01).
		if len(prompt) > 0 {
			s.reconcile(seq)
		}
	}()
	return out, g, nil
}

// commonPrefixLen returns the length of the longest common leading run of a and b.
func commonPrefixLen(a, b []int) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}
