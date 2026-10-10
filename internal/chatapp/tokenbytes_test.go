package chatapp

import (
	"bytes"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestJSONMasker_reusesCachedTokenBytes pins that jsonMasker reads the token-to-bytes table newSession builds once into
// s.tokenBytes, instead of rebuilding constrain.TokenBytes(s.vocab, s.tk.TokenText), an O(vocab) cost, on every
// constrained turn (the rebuild serve's cachedTokenBytes exists to avoid).
//
// Proof: s.tk is nil-ed after s.tokenBytes is built by hand (mirroring newSession's one-line wiring), so a call to
// s.tk.TokenText panics. No committed fixture pairs a real tokenizer with a loadable model (see tinyServed in
// internal/serveapp/prefillpath_test.go), so newSession's own construction is not exercised end to end. Origin (P-17):
// docs/code-notes/internal-chatapp.md#TestJSONMasker_reusesCachedTokenBytes.
func TestJSONMasker_reusesCachedTokenBytes(t *testing.T) {
	const vocab = 8
	s := &session{
		vocab:   vocab,
		special: tokenizer.SpecialTokens{EOS: -1, EndOfTurn: -1},
	}
	s.tokenBytes = constrain.TokenBytes(vocab, func(id int) []byte { return []byte{byte('a' + id)} })

	s.tk = nil // if jsonMasker still reads s.tk.TokenText, this panics
	mask := s.jsonMasker()
	if mask == nil {
		t.Fatal("jsonMasker() returned nil")
	}
}

// TestJSONMasker_holdsBackTemplateStopIDs pins that jsonMasker holds back the chat template's own turn-stop ids
// (s.stopIDs), not only EOS/EndOfTurn: Llama-3's <|eot_id|> and harmony's <|end|> are neither, so an unheld one could be
// emitted mid-document on a constrained generation. A held-back id must be masked to -Inf before the grammar can end (an
// empty document is not valid JSON yet, so CanEnd is false here), as internal/serveapp/openai.go's masker does by
// unioning eosIDs with stopIDs. Origin (N-71):
// docs/code-notes/internal-chatapp.md#TestJSONMasker_holdsBackTemplateStopIDs.
func TestJSONMasker_holdsBackTemplateStopIDs(t *testing.T) {
	const vocab = 8
	const stopID = 5 // e.g. Llama-3's <|eot_id|> — not EOS, not EndOfTurn
	s := &session{
		vocab:   vocab,
		special: tokenizer.SpecialTokens{EOS: -1, EndOfTurn: -1},
		stopIDs: []int{stopID},
	}
	s.tokenBytes = constrain.TokenBytes(vocab, func(id int) []byte { return []byte{byte('a' + id)} })

	mask := s.jsonMasker()
	logits := make([]float32, vocab)
	mask(nil, logits)
	if !math.IsInf(float64(logits[stopID]), -1) {
		t.Errorf("logits[%d] (a stopIDs entry) = %v, want -Inf — an empty document cannot end, "+
			"so an unheld stop id would let the generation emit the real template stop token "+
			"mid-JSON instead of masking it", stopID, logits[stopID])
	}
}

// TestJSONMasker_warnsOnUnreachableSchemaCompileFailure pins that jsonMasker's fallback for a schema that fails to
// compile is visible on stderr, not silent. main's flag parsing already fail-fasts (os.Exit) on an invalid --schema
// before s.schema is set, so the branch never fires via the CLI, but a silent downgrade is the wrong shape for the day
// something else sets s.schema without that check. The test drives the branch directly, bypassing the fail-fast, with a
// session holding intentionally invalid schema bytes. Origin (N-78):
// docs/code-notes/internal-chatapp.md#TestJSONMasker_warnsOnUnreachableSchemaCompileFailure.
func TestJSONMasker_warnsOnUnreachableSchemaCompileFailure(t *testing.T) {
	const vocab = 8
	s := &session{
		vocab:   vocab,
		special: tokenizer.SpecialTokens{EOS: -1, EndOfTurn: -1},
		schema:  []byte(`{"type": "not-a-real-schema-type"}`),
	}
	s.tokenBytes = constrain.TokenBytes(vocab, func(id int) []byte { return []byte{byte('a' + id)} })
	if _, err := constrain.JSONSchema(s.schema); err == nil {
		t.Fatal("test setup bug: schema must actually fail to compile")
	}

	oldStderr := os.Stderr
	r, w, perr := os.Pipe()
	if perr != nil {
		t.Fatalf("os.Pipe: %v", perr)
	}
	os.Stderr = w
	mask := s.jsonMasker()
	w.Close()
	os.Stderr = oldStderr
	var buf bytes.Buffer
	buf.ReadFrom(r)

	if mask == nil {
		t.Fatal("jsonMasker() returned nil")
	}
	if !strings.Contains(buf.String(), "schema failed to compile") {
		t.Errorf("expected a stderr warning naming the compile failure, got: %q", buf.String())
	}
}
