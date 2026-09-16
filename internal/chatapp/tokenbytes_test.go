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

// TestJSONMasker_reusesCachedTokenBytes is P-17's demo half (audit-2026-09-10): jsonMasker used to
// rebuild the constraint masker's token→bytes table (constrain.TokenBytes(s.vocab,
// s.tk.TokenText)) on every constrained turn — the same O(vocab) rebuild serve's own
// cachedTokenBytes exists to avoid. newSession now builds it once into s.tokenBytes instead, and
// jsonMasker must read that cached field rather than touching the tokenizer again.
//
// Proven by nil-ing s.tk AFTER building s.tokenBytes by hand (mirroring newSession's own one-line
// wiring) and confirming jsonMasker still runs: if it still called s.tk.TokenText, this would
// panic on the nil pointer instead of silently passing. No committed fixture pairs a real
// tokenizer with a loadable model (internal/serveapp/prefillpath_test.go's tinyServed notes the
// same gap: "the committed tiny GGUF carries no embedded tokenizer"), so newSession's own
// construction isn't exercised end-to-end here — its wiring is the one-line
// `s.tokenBytes = constrain.TokenBytes(...)` this test's setup mirrors exactly.
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

// TestJSONMasker_holdsBackTemplateStopIDs is N-71 (docs/audit-2026-09-10.md): jsonMasker only
// held back EOS/EndOfTurn, not the chat template's own turn-stop ids (s.stopIDs) — Llama-3's
// <|eot_id|> and harmony's <|end|> are neither, so a constrained generation for those families
// could emit the real stop token mid-document instead of it being masked. A held-back id must
// be masked to -Inf before the grammar can end (an empty document is not valid JSON yet, so
// CanEnd is false here) — internal/serveapp/openai.go's own masker already unions eosIDs with
// stopIDs; this pins the demo doing the same.
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

// TestJSONMasker_warnsOnUnreachableSchemaCompileFailure is N-78 (docs/audit-2026-09-10.md):
// jsonMasker's fallback for a schema that fails to compile used to be SILENT — main's flag
// parsing already fail-fasts (os.Exit) on an invalid --schema before s.schema is ever set, so in
// practice this branch never fires via the CLI, but a silent downgrade is still the wrong shape
// for the day something else sets s.schema without going through that check. This drives the
// branch directly (bypassing the CLI's fail-fast, which is the whole point) by constructing a
// session with intentionally-invalid schema bytes, and asserts the fallback is now visible on
// stderr rather than swallowed.
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
