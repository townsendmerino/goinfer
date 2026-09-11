package chatapp

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestStreamGen_matchesWholeSequenceDecode is P-17's deferred demo half (audit-2026-09-10):
// streamGen decodes each token INCREMENTALLY (DecodePiece, appended to a strings.Builder) instead
// of re-decoding the whole generated slice every token — proven here against a REAL tokenizer by
// comparing the concatenation of every flushed chunk to tk.DecodeContinuation(ids), the
// established whole-sequence reference for a CONTINUATION (not a fresh sequence — see DecodePiece
// and DecodeContinuation's own doc comments on why Decode's leading-space strip does not apply
// here). TestDecodeContinuation_isIncrementallyAssociative (tokenizer/) proves the underlying
// per-piece concatenation property directly; this proves THIS function's own cursor/flush
// bookkeeping composes with it correctly, end to end, with no model needed (streamGen takes the
// token channel directly).
func TestStreamGen_matchesWholeSequenceDecode(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)
	s := &session{tk: tk}

	ids := encodeSomeRealTextForTest(t, tk)
	if len(ids) < 3 {
		t.Fatalf("test setup: only %d ids encoded, want at least 3 for a meaningful streaming test", len(ids))
	}

	want, err := tk.DecodeContinuation(ids)
	if err != nil {
		t.Fatalf("DecodeContinuation: %v", err)
	}

	ch := make(chan int)
	go func() {
		defer close(ch)
		for _, id := range ids {
			ch <- id
		}
	}()

	var chunks []string
	text, nTok := s.streamGen(ch, func(chunk string) { chunks = append(chunks, chunk) })

	if nTok != len(ids) {
		t.Errorf("nTok = %d, want %d", nTok, len(ids))
	}
	if text != want {
		t.Errorf("streamGen returned text = %q, want %q (tk.DecodeContinuation(ids))", text, want)
	}
	got := strings.Join(chunks, "")
	if got != want {
		t.Errorf("concatenated flushed chunks = %q, want %q — streamGen's incremental flush "+
			"diverged from a whole-sequence decode", got, want)
	}
	if len(chunks) == 0 {
		t.Error("onChunk was never called — nothing was ever flushed")
	}
}

// TestStreamGen_emptyChannelReturnsEmpty is the zero-token edge case: streamGen must not panic or
// call onChunk on an immediately-closed channel.
func TestStreamGen_emptyChannelReturnsEmpty(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)
	s := &session{tk: tk}

	ch := make(chan int)
	close(ch)

	called := false
	text, nTok := s.streamGen(ch, func(string) { called = true })
	if text != "" || nTok != 0 {
		t.Errorf("streamGen(empty) = (%q, %d), want (\"\", 0)", text, nTok)
	}
	if called {
		t.Error("onChunk was called on an empty stream")
	}
}

// loadTinyTokenizerForTest loads the committed tokenizer-only fixture (no model weights needed —
// streamGen only touches s.tk) shared with the tokenizer package's own tests.
func loadTinyTokenizerForTest(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	p := filepath.Join("..", "..", "tokenizer", "testdata", "chatml-tiny.gguf")
	tk, err := tokenizer.LoadGGUF(p)
	if err != nil {
		t.Skipf("no committed tiny tokenizer fixture at %s: %v", p, err)
	}
	return tk
}

// encodeSomeRealTextForTest round-trips a short ASCII string through the real tokenizer so the
// test drives streamGen with ids this exact vocabulary actually produces, rather than
// hand-guessed ones.
func encodeSomeRealTextForTest(t *testing.T, tk *tokenizer.Tokenizer) []int {
	t.Helper()
	ids, err := tk.Encode("The quick brown fox jumps over the lazy dog.", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return ids
}
