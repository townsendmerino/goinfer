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
}

// Masker masks a step's logits to the tokens a Grammar permits. Build one with
// NewMasker and pass its Process method as decoder.SamplingParams.LogitProcessor.
// A Masker is single-use per generation (it tracks committed state); call Reset
// to reuse it for another sequence.
type Masker struct {
	g         Grammar
	tokens    [][]byte     // per-id surface bytes
	isEOS     map[int]bool // ids that may only be emitted when the grammar CanEnd
	eosIDs    []int        // the EOS ids, in order (for StopWhenComplete)
	stopAtEnd bool         // once CanEnd, mask everything but EOS to force a stop
	committed int          // how many generated tokens have been folded into g
}

// NewMasker builds a Masker for a vocabulary. tokens[id] is the surface bytes
// token id contributes (see TokenBytes); eosIDs are end/stop tokens, allowed
// only when the document is complete. The grammar is Reset to its initial state.
func NewMasker(g Grammar, tokens [][]byte, eosIDs []int) *Masker {
	g.Reset()
	eos := make(map[int]bool, len(eosIDs))
	for _, id := range eosIDs {
		eos[id] = true
	}
	return &Masker{g: g, tokens: tokens, isEOS: eos, eosIDs: eosIDs}
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
		if m.isEOS[id] {
			if !canEnd {
				logits[id] = neg
			}
			continue
		}
		// StopWhenComplete: at a completion point, only EOS survives.
		if canEnd && m.stopAtEnd {
			logits[id] = neg
			continue
		}
		if !m.g.TryBytes(m.tokenBytes(id)) {
			logits[id] = neg
		}
	}
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
