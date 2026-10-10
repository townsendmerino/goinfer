package agent

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSchemaMasker_reusesCachedTokenBytes pins that schemaMasker reads the token-to-bytes table newSession builds once
// into s.tokenBytes, instead of rebuilding constrain.TokenBytes(s.vocab, s.tk.TokenText), an O(vocab) cost, on every
// DECIDE turn (the rebuild serve's cachedTokenBytes exists to avoid).
//
// Proof: s.tk is nil-ed after s.tokenBytes is built by hand (mirroring newSession's one-line wiring), so a call to
// s.tk.TokenText panics. No committed fixture pairs a real tokenizer with a loadable model (see tinyServed in
// internal/serveapp/prefillpath_test.go), so newSession's own construction is not exercised end to end. Origin (P-17):
// docs/code-notes/demo-agent-agent.md#TestSchemaMasker_reusesCachedTokenBytes.
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

// TestSchemaMasker_holdsBackTemplateStopIDs pins that schemaMasker holds back the chat template's own turn-stop ids
// (s.stopIDs), not only EOS/EndOfTurn: Llama-3's <|eot_id|> and harmony's <|end|> are neither, so an unheld one could be
// emitted mid-document on a constrained DECIDE turn. A held-back id must be masked to -Inf before the grammar can end
// (an empty document is not valid JSON yet, so CanEnd is false here), as internal/serveapp/openai.go's masker does by
// unioning eosIDs with stopIDs. Origin (N-71):
// docs/code-notes/demo-agent-agent.md#TestSchemaMasker_holdsBackTemplateStopIDs.
//
// The stop id's own surface MUST be a byte the grammar would otherwise accept here (decisionSchema requires an object,
// so its first byte is "{"); otherwise grammar-validity masking alone would mask it regardless of stop-id registration,
// and the test could not tell the fix from a no-op.
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
