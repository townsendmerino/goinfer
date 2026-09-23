package gemmaapp

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestStreamGen_matchesWholeSequenceDecode is R-08's gemmaapp half (task-recompute-audit.md):
// streamGen decodes the prompt once, then accumulates each GENERATED token INCREMENTALLY
// (DecodePiece, appended to a strings.Builder) instead of re-decoding the whole running sequence
// (prompt + every generated token so far) on every token — proven here against a REAL tokenizer by
// comparing streamGen's output to tk.Decode(append(promptIDs, generatedIDs...)), the whole-sequence
// reference the OLD code always called. This is the "own design pass" this file's original R-08
// deferral named as the reason it wasn't ported unchanged from chatapp/agent.go's own streamGen:
// unlike those two, this loop also decodes the PROMPT (not just a continuation), so it needs the
// leading-space strip to land once, at the true sequence start — which a per-token DecodePiece
// never applies on its own.
func TestStreamGen_matchesWholeSequenceDecode(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)

	promptIDs, genIDs := encodeSomePromptAndContinuationForTest(t, tk)
	if len(genIDs) < 3 {
		t.Fatalf("test setup: only %d generated ids, want at least 3 for a meaningful streaming test", len(genIDs))
	}

	full := append(append([]int(nil), promptIDs...), genIDs...)
	want, err := tk.Decode(full)
	if err != nil {
		t.Fatalf("Decode(full): %v", err)
	}

	ch := make(chan int)
	go func() {
		defer close(ch)
		for _, id := range genIDs {
			ch <- id
		}
	}()

	var chunks []string
	text, nTok, err := streamGen(tk, promptIDs, ch, func(chunk string) { chunks = append(chunks, chunk) })
	if err != nil {
		t.Fatalf("streamGen: %v", err)
	}

	if nTok != len(genIDs) {
		t.Errorf("nTok = %d, want %d", nTok, len(genIDs))
	}
	if text != want {
		t.Errorf("streamGen returned text = %q, want %q (tk.Decode(prompt+generated))", text, want)
	}
	got := strings.Join(chunks, "")
	if got != want {
		t.Errorf("concatenated flushed chunks = %q, want %q — streamGen's incremental flush "+
			"diverged from a whole-sequence decode", got, want)
	}
	if len(chunks) == 0 {
		t.Error("onChunk was never called — not even the prompt was flushed")
	}
}

// TestStreamGen_emptyChannelRendersOnlyThePrompt is the zero-generated-tokens edge case:
// streamGen must still render the prompt exactly once and must not panic or call onChunk again on
// an immediately-closed channel.
func TestStreamGen_emptyChannelRendersOnlyThePrompt(t *testing.T) {
	tk := loadTinyTokenizerForTest(t)
	promptIDs, _ := encodeSomePromptAndContinuationForTest(t, tk)

	want, err := tk.Decode(promptIDs)
	if err != nil {
		t.Fatalf("Decode(promptIDs): %v", err)
	}

	ch := make(chan int)
	close(ch)

	var chunks []string
	text, nTok, err := streamGen(tk, promptIDs, ch, func(chunk string) { chunks = append(chunks, chunk) })
	if err != nil {
		t.Fatalf("streamGen: %v", err)
	}
	if text != want {
		t.Errorf("streamGen(empty) text = %q, want %q (prompt only)", text, want)
	}
	if nTok != 0 {
		t.Errorf("nTok = %d, want 0", nTok)
	}
	if got := strings.Join(chunks, ""); got != want {
		t.Errorf("concatenated flushed chunks = %q, want %q", got, want)
	}
}

// loadTinyTokenizerForTest loads the committed tokenizer-only fixture shared with chatapp's own
// equivalent test (no model weights needed — streamGen only touches the tokenizer).
func loadTinyTokenizerForTest(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	p := filepath.Join("..", "..", "tokenizer", "testdata", "chatml-tiny.gguf")
	tk, err := tokenizer.LoadGGUF(p)
	if err != nil {
		t.Skipf("no committed tiny tokenizer fixture at %s: %v", p, err)
	}
	return tk
}

// encodeSomePromptAndContinuationForTest round-trips two short ASCII strings through the real
// tokenizer so the test drives streamGen with ids this exact vocabulary actually produces, split
// into a "prompt" half and a "generated continuation" half the way run() itself splits them.
func encodeSomePromptAndContinuationForTest(t *testing.T, tk *tokenizer.Tokenizer) (promptIDs, genIDs []int) {
	t.Helper()
	promptIDs, err := tk.Encode("The quick brown fox jumps over the lazy dog.", true /* addBOS */)
	if err != nil {
		t.Fatalf("Encode(prompt): %v", err)
	}
	if len(promptIDs) > 0 && promptIDs[0] == tk.Special().BOS {
		promptIDs = promptIDs[1:]
	}
	genIDs, err = tk.Encode(" And then it kept going, for quite a while longer than expected.", false)
	if err != nil {
		t.Fatalf("Encode(continuation): %v", err)
	}
	return promptIDs, genIDs
}
