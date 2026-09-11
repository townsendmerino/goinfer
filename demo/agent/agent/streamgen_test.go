package agent

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestStreamGen_matchesWholeSequenceDecode is P-17's deferred demo half (audit-2026-09-10):
// streamGen decodes each token INCREMENTALLY (DecodePiece, appended to a strings.Builder) instead
// of re-decoding the whole generated slice every token — proven here against a REAL tokenizer by
// comparing the concatenation of every emitted chunk to tk.DecodeContinuation(ids), the
// established whole-sequence reference for a CONTINUATION (not a fresh sequence — see DecodePiece
// and DecodeContinuation's own doc comments on why Decode's leading-space strip does not apply
// here). TestDecodeContinuation_isIncrementallyAssociative (tokenizer/) proves the underlying
// per-piece concatenation property directly; this proves THIS function's own cursor/flush
// bookkeeping composes with it correctly, end to end, with no model needed — streamGen takes the
// token channel and a bare *decoder.Generation directly.
func TestStreamGen_matchesWholeSequenceDecode(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)
	s := &Session{tk: tk}

	ids, err := tk.Encode("The quick brown fox jumps over the lazy dog.", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
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
	text, err := s.streamGen(context.Background(), ch, &decoder.Generation{}, func(chunk string) {
		chunks = append(chunks, chunk)
	})
	if err != nil {
		t.Fatalf("streamGen: %v", err)
	}

	if text != want {
		t.Errorf("streamGen returned text = %q, want %q (tk.DecodeContinuation(ids))", text, want)
	}
	got := strings.Join(chunks, "")
	if got != want {
		t.Errorf("concatenated emitted chunks = %q, want %q — streamGen's incremental flush "+
			"diverged from a whole-sequence decode", got, want)
	}
	if len(chunks) == 0 {
		t.Error("onToken was never called — nothing was ever emitted")
	}
}

// TestStreamGen_nilOnTokenIsSafe: onToken may be nil (generateImage's non-streaming callers) —
// streamGen must still return the full decoded text without panicking.
func TestStreamGen_nilOnTokenIsSafe(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)
	s := &Session{tk: tk}

	ids, err := tk.Encode("hello world", false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
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

	text, err := s.streamGen(context.Background(), ch, &decoder.Generation{}, nil)
	if err != nil {
		t.Fatalf("streamGen: %v", err)
	}
	if text != want {
		t.Errorf("streamGen with nil onToken returned %q, want %q", text, want)
	}
}

// loadTinyTokenizerForTest loads the committed tokenizer-only fixture (no model weights needed —
// streamGen only touches s.tk), the same fixture internal/chatapp's own streamGen test uses.
func loadTinyTokenizerForTest(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	p := filepath.Join("..", "..", "..", "tokenizer", "testdata", "chatml-tiny.gguf")
	tk, err := tokenizer.LoadGGUF(p)
	if err != nil {
		t.Skipf("no committed tiny tokenizer fixture at %s: %v", p, err)
	}
	return tk
}
