package agent

import (
	"math"
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

// TestSchemaMasker_holdsBackTemplateStopIDs is N-71 (docs/audit-2026-09-10.md): schemaMasker only
// held back EOS/EndOfTurn, not the chat template's own turn-stop ids (s.stopIDs) — Llama-3's
// <|eot_id|> and harmony's <|end|> are neither, so a constrained DECIDE turn for those families
// could emit the real stop token mid-document instead of it being masked. A held-back id must be
// masked to -Inf before the grammar can end (an empty document is not valid JSON yet, so CanEnd
// is false here) — internal/serveapp/openai.go's own masker already unions eosIDs with stopIDs;
// this pins the demo doing the same.
//
// The stop id's own surface MUST be a byte the grammar would otherwise accept here (decisionSchema
// requires an object, so its first byte is "{") — otherwise grammar-validity masking alone would
// mask it regardless of EOS registration, and the test could not tell the fix apart from a no-op.
func TestSchemaMasker_holdsBackTemplateStopIDs(t *testing.T) {
	const vocab = 8
	const stopID = 5 // e.g. Llama-3's <|eot_id|> — not EOS, not EndOfTurn
	s := &Session{
		vocab:   vocab,
		special: tokenizer.SpecialTokens{EOS: -1, EndOfTurn: -1},
		stopIDs: []int{stopID},
	}
	s.tokenBytes = constrain.TokenBytes(vocab, func(id int) []byte {
		if id == stopID {
			return []byte("{") // grammar-valid here on its own; only EOS registration masks it
		}
		return []byte{byte('a' + id)}
	})

	mask := s.schemaMasker([]byte(decisionSchema))
	logits := make([]float32, vocab)
	mask(nil, logits)
	if !math.IsInf(float64(logits[stopID]), -1) {
		t.Errorf("logits[%d] (a stopIDs entry) = %v, want -Inf — an empty document cannot end, "+
			"so an unheld stop id would let the generation emit the real template stop token "+
			"mid-JSON instead of masking it", stopID, logits[stopID])
	}
}
