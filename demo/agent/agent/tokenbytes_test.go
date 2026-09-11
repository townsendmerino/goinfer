package agent

import (
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSchemaMasker_reusesCachedTokenBytes is P-17's demo half (audit-2026-09-10): schemaMasker
// used to rebuild the constraint masker's token→bytes table (constrain.TokenBytes(s.vocab,
// s.tk.TokenText)) on every DECIDE turn — the same O(vocab) rebuild serve's own cachedTokenBytes
// exists to avoid. newSession now builds it once into s.tokenBytes instead, and schemaMasker must
// read that cached field rather than touching the tokenizer again.
//
// Proven by nil-ing s.tk AFTER building s.tokenBytes by hand (mirroring newSession's own one-line
// wiring) and confirming schemaMasker still runs: if it still called s.tk.TokenText, this would
// panic on the nil pointer instead of silently passing. No committed fixture pairs a real
// tokenizer with a loadable model (internal/serveapp/prefillpath_test.go's tinyServed notes the
// same gap), so newSession's own construction isn't exercised end-to-end here — its wiring is the
// one-line `s.tokenBytes = constrain.TokenBytes(...)` this test's setup mirrors exactly.
func TestSchemaMasker_reusesCachedTokenBytes(t *testing.T) {
	const vocab = 8
	s := &Session{
		vocab:   vocab,
		special: tokenizer.SpecialTokens{EOS: -1, EndOfTurn: -1},
	}
	s.tokenBytes = constrain.TokenBytes(vocab, func(id int) []byte { return []byte{byte('a' + id)} })

	s.tk = nil // if schemaMasker still reads s.tk.TokenText, this panics
	mask := s.schemaMasker([]byte(decisionSchema))
	if mask == nil {
		t.Fatal("schemaMasker() returned nil")
	}
}
