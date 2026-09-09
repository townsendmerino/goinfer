//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"testing"
)

// TestPrefillImageLast_declinesOverChunk pins v1's core safety invariant: a bidirectional image
// block must never be split across a prefillChunked chunk boundary (unverified, likely unsafe —
// see PrefillImageLast's own doc comment and decoder.ResidentImagePrefill's), so a prompt wider
// than one weight-stationary pass must DECLINE cleanly rather than being silently truncated,
// crashing, or (worst case) producing a wrong answer with no error anywhere. Needs no device: the
// M>chunk check runs before prefillCore ever touches the executor.
//
// Pins prefillImageChunkRows(), NOT prefillChunkRows() — PrefillImageLast has its own, larger,
// image-specific row budget (prefillImageDefaultChunk); using the wrong constant here would build
// an "oversized" prompt that is actually well UNDER the real boundary, silently stop exercising
// the decline path, and fall through to a real device call this no-device test isn't set up for.
// GOINFER_PREFILL_IMAGE_CHUNK pinned small so the test stays cheap (no need to allocate 2048+ rows).
func TestPrefillImageLast_declinesOverChunk(t *testing.T) {
	r := declineFixture(1, "int4")
	t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", "8")
	chunk := prefillImageChunkRows()
	if chunk != 8 {
		t.Fatalf("test setup: prefillImageChunkRows() = %d, want 8 (env override didn't take)", chunk)
	}

	embeddings := make([][]float32, chunk+1)
	for i := range embeddings {
		embeddings[i] = make([]float32, r.hidden)
	}
	_, err := r.PrefillImageLast(context.Background(), embeddings, 0, 0, len(embeddings))
	if err == nil {
		t.Fatalf("prompt of %d rows (chunk width %d) must decline, not silently proceed", len(embeddings), chunk)
	}
	if !errors.Is(err, errPrefillDeclined) {
		t.Errorf("decline must wrap errPrefillDeclined so GenerateVL can fall back to the CPU-prefill+UploadKV bridge cleanly: %v", err)
	}
}

// TestPrefillImageChunkRows_defaultAndOverride pins the default (2048, chosen because
// prefillDefaultChunk's own measurement table already measured that width safe on this box — see
// prefillImageDefaultChunk's doc comment) and the GOINFER_PREFILL_IMAGE_CHUNK override, mirroring
// prefillChunkRows's own env-var behavior (an unparseable/non-positive value is ignored, not fatal
// — a typo in a tuning knob must not take a model off the fast path).
func TestPrefillImageChunkRows_defaultAndOverride(t *testing.T) {
	if got := prefillImageChunkRows(); got != prefillImageDefaultChunk {
		t.Errorf("prefillImageChunkRows() with no override = %d, want the default %d", got, prefillImageDefaultChunk)
	}
	t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", "1024")
	if got := prefillImageChunkRows(); got != 1024 {
		t.Errorf("prefillImageChunkRows() with GOINFER_PREFILL_IMAGE_CHUNK=1024 = %d, want 1024", got)
	}
	for _, bad := range []string{"0", "-5", "not-a-number", ""} {
		t.Run("ignores "+bad, func(t *testing.T) {
			t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", bad)
			if got := prefillImageChunkRows(); got != prefillImageDefaultChunk {
				t.Errorf("prefillImageChunkRows() with GOINFER_PREFILL_IMAGE_CHUNK=%q = %d, want the default %d (bad override must be ignored, not fatal)",
					bad, got, prefillImageDefaultChunk)
			}
		})
	}
}

// TestPrefillImageLast_invalidRangeDeclines pins the other input-validation half: an image block
// that does not fit the prompt (negative start, empty/inverted range, or a range extending past
// the prompt) must decline rather than index out of bounds or silently clamp.
func TestPrefillImageLast_invalidRangeDeclines(t *testing.T) {
	r := declineFixture(1, "int4")
	embeddings := make([][]float32, 8)
	for i := range embeddings {
		embeddings[i] = make([]float32, r.hidden)
	}
	for _, tc := range []struct {
		name             string
		imgStart, imgEnd int
	}{
		{"negative start", -1, 4},
		{"empty range", 3, 3},
		{"inverted range", 5, 2},
		{"past the prompt", 4, 9},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := r.PrefillImageLast(context.Background(), embeddings, 0, tc.imgStart, tc.imgEnd)
			if err == nil {
				t.Fatalf("image block [%d,%d) over %d rows must decline", tc.imgStart, tc.imgEnd, len(embeddings))
			}
			if !errors.Is(err, errPrefillDeclined) {
				t.Errorf("decline must wrap errPrefillDeclined: %v", err)
			}
		})
	}
}
