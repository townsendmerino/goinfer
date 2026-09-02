// Package constrain implements constrained / structured decoding: a logit mask
// that forces a language model's output to satisfy a grammar (e.g. a small model
// that physically cannot emit malformed JSON). It plugs into the decoder via the
// SamplingParams.LogitProcessor hook — Masker.Process matches that signature.
//
// The mechanism is the standard one: at each decode step, for every vocab token,
// check whether appending that token's bytes keeps the output a valid prefix of
// the grammar; set the logits of the ones that don't to −∞ so the sampler can
// never pick them. The end-of-sequence token is masked until the output is a
// complete document (Grammar.CanEnd), so generation can't stop mid-structure.
//
// The package is stdlib-only: the vocabulary → bytes mapping is injected as a
// func (e.g. tokenizer.Tokenizer.TokenText), so constrain does not depend on the
// tokenizer or decoder packages — it just produces a LogitProcessor-shaped func.
package constrain

import "math"

// Grammar is an incremental byte-level acceptor for constrained decoding. A
// token is allowed at a step iff appending its bytes keeps the output a valid
// prefix (TryBytes); the chosen token is then Committed. CanEnd reports whether
// the output so far is a complete, valid document.
type Grammar interface {
	// TryBytes reports whether appending bs keeps the output a valid prefix,
	// WITHOUT changing state.
	TryBytes(bs []byte) bool
	// Commit advances the state over bs (which must have passed TryBytes).
	Commit(bs []byte)
	// CanEnd reports whether the output so far is a complete, valid document.
	CanEnd() bool
	// Reset returns the grammar to its initial state.
	Reset()
	// Clone returns an independent copy of the grammar at its current state, for
	// non-mutating multi-step lookahead (speculative forced-run extraction, 01).
	Clone() Grammar
}

// Masker masks a step's logits to the tokens a Grammar permits. Build one with
// NewMasker and pass its Process method as decoder.SamplingParams.LogitProcessor.
// A Masker is single-use per generation (it tracks committed state); call Reset
// to reuse it for another sequence.
type Masker struct {
	g      Grammar
	tokens [][]byte // per-id surface bytes
	// isEOS is indexed, not mapped: it is probed once per vocab id per decode step —
	// 151,936 probes/step on Qwen2.5 — and a map lookup there was pure overhead on the
	// hottest loop in constrained decoding (audit P-20's cheapest lever, measured).
	// Sized to cover the largest EOS id; ids past the end are simply not EOS, which is the
	// same answer the map gave for an absent key.
	isEOS     []bool
	eosIDs    []int // the EOS ids, in order (for StopWhenComplete)
	stopAtEnd bool  // once CanEnd, mask everything but EOS to force a stop
	committed int   // how many generated tokens have been folded into g
}

// eosAt reports whether id is an end/stop token. Bounds-checked because logits can be the
// MODEL's padded vocab length, which runs past both the tokenizer table and this slice — the
// same over-long-logits case M26 guarded in tokenBytes.
func (m *Masker) eosAt(id int) bool { return id >= 0 && id < len(m.isEOS) && m.isEOS[id] }

// NewMasker builds a Masker for a vocabulary. tokens[id] is the surface bytes
// token id contributes (see TokenBytes); eosIDs are end/stop tokens, allowed
// only when the document is complete. The grammar is Reset to its initial state.
func NewMasker(g Grammar, tokens [][]byte, eosIDs []int) *Masker {
	g.Reset()
	n := len(tokens)
	for _, id := range eosIDs {
		if id >= n {
			n = id + 1
		}
	}
	eos := make([]bool, n)
	for _, id := range eosIDs {
		if id >= 0 {
			eos[id] = true
		}
	}
	return &Masker{g: g, tokens: tokens, isEOS: eos, eosIDs: eosIDs}
}

// ForcedRun returns up to max tokens the grammar FORCES from its current committed
// state: successive positions where exactly one surface token is grammar-legal and
// the document cannot yet end (so EOS is not an alternative). Under the grammar mask
// such a position has a point-mass target distribution, so a drafter that proposes
// the forced token is accepted with probability 1 at zero model cost (01
// grammar-fused). Non-mutating — it probes on a Clone of the grammar; the caller
// commits the run through the normal Process path once the verifier confirms it.
//
// CAVEAT: goinfer's grammars permit optional whitespace at every structural
// boundary, so a whitespace token is also legal there — strict single-token forcing
// fires mainly INSIDE fixed literals (object keys, enum/const values), not at the
// scaffolding between them. Measure the forced fraction before relying on it.
func (m *Masker) ForcedRun(max int) []int {
	if max <= 0 {
		return nil
	}
	g := m.g.Clone()
	var run []int
	for len(run) < max {
		if g.CanEnd() {
			break // a complete document ⇒ EOS is a legal alternative ⇒ not single-forced
		}
		forced, count := -1, 0
		for id, b := range m.tokens {
			if len(b) == 0 || m.eosAt(id) {
				continue // control / EOS tokens are not surface continuations
			}
			if g.TryBytes(b) {
				forced, count = id, count+1
				if count > 1 {
					break
				}
			}
		}
		if count != 1 {
			break
		}
		run = append(run, forced)
		g.Commit(m.tokenBytes(forced))
	}
	return run
}

// GrammarClone returns a clone of the masker's current grammar state — a working
// copy for speculative per-position masking (MaskAt) and forced-run advance that the
// verifier rolls forward over a draft block without touching the live grammar.
func (m *Masker) GrammarClone() Grammar { return m.g.Clone() }

// Commit advances the masker's LIVE grammar over a committed token (its surface
// bytes), keeping it in sync as the speculative loop emits tokens — the
// out-of-order analogue of Process's fold-in step.
func (m *Masker) Commit(id int) { m.g.Commit(m.tokenBytes(id)) }

// TokenBytes returns the surface bytes of token id (for advancing a grammar clone
// over an accepted draft token in the speculative verifier).
func (m *Masker) TokenBytes(id int) []byte { return m.tokenBytes(id) }

// MaskAt masks logits to the tokens grammar state g permits — the same rule as
// Process (EOS only when g.CanEnd; StopWhenComplete forces EOS at a completion
// point) — but using the supplied g rather than the masker's own committed grammar.
// Does not mutate m or g. The verifier uses it to mask each speculative position
// against a clone rolled forward over the accepted draft prefix.
func (m *Masker) MaskAt(g Grammar, logits []float32) {
	neg := float32(math.Inf(-1))
	canEnd := g.CanEnd()
	for id := range logits {
		if m.maskID(g, id, canEnd) {
			logits[id] = neg
		}
	}
}

// ForcedBytesRun returns the maximal run of BYTES the grammar forces from its
// current committed state — successive positions where exactly one byte is legal and
// the document cannot yet end. This is the right granularity for a BPE vocab (unlike
// ForcedRun's strict single-token test, which a real tokenizer almost never satisfies
// because many tokens are legal byte-prefixes of a forced continuation): the caller
// retokenizes the returned bytes (canonically) into draft tokens, and the verifier
// confirms — lossless, with acceptance < 1 only where the model tokenizes the forced
// bytes differently. Non-mutating (probes on a Clone). max caps the run in bytes.
func (m *Masker) ForcedBytesRun(max int) []byte {
	if max <= 0 {
		return nil
	}
	g := m.g.Clone()
	var run []byte
	var probe [1]byte
	for len(run) < max {
		if g.CanEnd() {
			break // a complete document ⇒ EOS is a legal alternative ⇒ not forced
		}
		only, count := byte(0), 0
		for b := 0; b < 256 && count < 2; b++ {
			probe[0] = byte(b)
			if g.TryBytes(probe[:]) {
				only, count = byte(b), count+1
			}
		}
		if count != 1 {
			break
		}
		run = append(run, only)
		g.Commit([]byte{only})
	}
	return run
}

// StopWhenComplete makes the Masker mask every non-EOS token once the document
// is complete (CanEnd), forcing the next token to be EOS — so generation stops
// at the first complete document instead of trailing whitespace to maxTokens.
// No-op (and unsafe to enable) without EOS ids. Returns the Masker for chaining.
func (m *Masker) StopWhenComplete() *Masker {
	if len(m.eosIDs) > 0 {
		m.stopAtEnd = true
	}
	return m
}

// Process is a decoder.SamplingParams.LogitProcessor: it folds any
// newly-generated tokens into the grammar, then sets the logits of every token
// that would break the grammar to −∞ (and every EOS token to −∞ unless the
// document is already complete), so the sampler can only pick a valid next token.
func (m *Masker) Process(generated []int, logits []float32) {
	for ; m.committed < len(generated); m.committed++ {
		m.g.Commit(m.tokenBytes(generated[m.committed]))
	}
	neg := float32(math.Inf(-1))
	canEnd := m.g.CanEnd()
	for id := range logits {
		if m.maskID(m.g, id, canEnd) {
			logits[id] = neg
		}
	}
}

// maskID reports whether token id must be masked in grammar state g. It is the ONE
// masking rule; Process and MaskAt both call it, because they had two verbatim copies
// of it and M-27 was a defect in the copy — the shape this audit keeps turning up.
//
// canEnd is passed in rather than recomputed: both callers need it for the EOS branch
// and CanEnd is not free on every grammar.
func (m *Masker) maskID(g Grammar, id int, canEnd bool) bool {
	if m.eosAt(id) {
		return !canEnd // EOS only once the document is complete
	}
	b := m.tokenBytes(id)
	// An id with no surface bytes (a control token, or a padded-vocab id past the
	// tokenizer table — tokenBytes returns nil for both) can never advance the
	// grammar: TryBytes(nil) is vacuously true, so leaving it legal lets the sampler
	// pick an id that never progresses, livelocking to maxTokens and then failing to
	// Decode. Mask it (EOS was already handled above) — M26. tokenBytes also
	// bounds-checks id, so a model-vocab-length logits slice (padded past the
	// tokenizer) can't index m.tokens out of range.
	if len(b) == 0 || !g.TryBytes(b) {
		return true
	}
	// StopWhenComplete: stop at the first complete document rather than trailing to
	// maxTokens. M-27: this used to mask EVERY non-EOS token at a completion point,
	// which conflates MAY-end with MUST-end. CanEnd is true after `1` for a top-level
	// number — but `12` is a longer legal document, not trailing filler, so blanket
	// masking made `{"type":"integer"}` return exactly one digit and
	// `{"enum":[1,10,100]}` able to produce only `1`.
	//
	// What StopWhenComplete is actually for is suppressing the WHITESPACE the grammars
	// permit at every structural boundary, so that is what it suppresses. A token that
	// genuinely extends the VALUE has already passed TryBytes above and is kept. This
	// needs no per-grammar may-end/must-end split: whitespace-only is the exact
	// property, and it is the same one for json, schema and tool grammars.
	if canEnd && m.stopAtEnd && allJSONSpace(b) {
		return true
	}
	return false
}

// allJSONSpace reports whether b is non-empty and entirely RFC 8259 whitespace — the
// only continuation a completed document can take that does not change its value.
func allJSONSpace(b []byte) bool {
	for _, c := range b {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return false
		}
	}
	return len(b) > 0
}

// CanEnd reports whether the committed output is a complete, valid document.
func (m *Masker) CanEnd() bool { return m.g.CanEnd() }

// Reset clears the committed state so the Masker can drive a new sequence.
func (m *Masker) Reset() {
	m.g.Reset()
	m.committed = 0
}

func (m *Masker) tokenBytes(id int) []byte {
	if id < 0 || id >= len(m.tokens) {
		return nil
	}
	return m.tokens[id]
}

// TokenBytes materializes the per-id surface bytes for a vocabulary of vocabSize
// tokens by calling text(id) for each — e.g. TokenBytes(vocab, tk.TokenText).
// Precomputed once so the per-step mask is pure grammar work.
func TokenBytes(vocabSize int, text func(id int) []byte) [][]byte {
	out := make([][]byte, vocabSize)
	for id := range out {
		out[id] = text(id)
	}
	return out
}
