//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"os"
	"testing"
)

// TestPrefillImageLast_declinesOverChunk pins that a bidirectional image block is never split across a
// prefillChunked chunk boundary (unverified, likely unsafe: see PrefillImageLast's doc comment and
// decoder.ResidentImagePrefill's), so a prompt wider than one weight-stationary pass must DECLINE cleanly
// rather than truncate, crash or return a wrong answer with no error. Needs no device: the M>chunk check
// runs before prefillCore touches the executor.
//
// It pins prefillImageChunkRows(), NOT prefillChunkRows(): PrefillImageLast has its own, larger row budget
// (prefillImageDefaultChunk), and the wrong constant builds a prompt well UNDER the real boundary, which
// stops exercising the decline path and falls through to a device call this test is not set up for.
// GOINFER_PREFILL_IMAGE_CHUNK is pinned small so the test stays cheap.
func TestPrefillImageLast_declinesOverChunk(t *testing.T) {
	r := declineFixture(1, "int4")
	t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", "8")
	chunk := prefillImageChunkRows(r.knobValue("GOINFER_PREFILL_IMAGE_CHUNK"))
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

// TestPrefillImageChunkRows_defaultAndOverride pins the default (2048; see prefillImageDefaultChunk) and the
// GOINFER_PREFILL_IMAGE_CHUNK override, mirroring prefillChunkRows: an unparseable or non-positive value is
// ignored, not fatal, because a typo in a tuning knob must not take a model off the fast path.
func TestPrefillImageChunkRows_defaultAndOverride(t *testing.T) {
	if got := prefillImageChunkRows(os.Getenv("GOINFER_PREFILL_IMAGE_CHUNK")); got != prefillImageDefaultChunk {
		t.Errorf("prefillImageChunkRows() with no override = %d, want the default %d", got, prefillImageDefaultChunk)
	}
	t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", "1024")
	if got := prefillImageChunkRows(os.Getenv("GOINFER_PREFILL_IMAGE_CHUNK")); got != 1024 {
		t.Errorf("prefillImageChunkRows() with GOINFER_PREFILL_IMAGE_CHUNK=1024 = %d, want 1024", got)
	}
	for _, bad := range []string{"0", "-5", "not-a-number", ""} {
		t.Run("ignores "+bad, func(t *testing.T) {
			t.Setenv("GOINFER_PREFILL_IMAGE_CHUNK", bad)
			if got := prefillImageChunkRows(os.Getenv("GOINFER_PREFILL_IMAGE_CHUNK")); got != prefillImageDefaultChunk {
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
