package decoder

import "context"

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

// Reset drops all cached KV, returning the session to empty without freeing the
// backing arrays (so the next generation re-grows into them).
func (s *Session) Reset() {
	s.cache.TruncateTo(0)
	s.tokens = s.tokens[:0]
}

// Generate is Model.Generate with cross-call KV reuse. It rewinds the cache to
// the longest token prefix shared with the session's current state, prefills
// only the divergent suffix of prompt, decodes, and leaves the cache holding
// prompt + the generated tokens for the next call to extend. Streaming, sampling,
// stop handling, and the *Generation result match Model.Generate exactly.
func (s *Session) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}

	// Longest shared token prefix → reuse its KV. Keep at least one token to
	// prefill (the seed logits come from re-running the last prompt token), so
	// cap reuse at len(prompt)-1 even on an exact-prompt repeat. matched falls to
	// 0 on an empty prompt; generateInto then reports the empty-prompt error.
	matched := max(min(commonPrefixLen(s.tokens, prompt), len(prompt)-1), 0)
	s.cache.TruncateTo(matched)

	// After prefill the cache holds the whole prompt; commit appends each
	// generated token as its forward lands it in the cache, so seq mirrors the
	// cache exactly — including a clean rollback if a forward errors mid-stream.
	seq := append([]int(nil), prompt...)
	commit := func(id int) { seq = append(seq, id) }

	go func() {
		defer close(out)
		s.m.generateInto(ctx, out, g, s.cache, prompt, matched, maxTokens, sp, commit)
		// Bookkeeping runs before the deferred close, so a consumer that observes
		// the channel close (and then starts the next request) always sees a
		// reconciled session: tokens == what the cache holds, no partial position.
		s.tokens = seq
		s.cache.TruncateTo(len(seq))
	}()
	return out, g
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
