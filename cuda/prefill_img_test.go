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
func TestPrefillImageLast_declinesOverChunk(t *testing.T) {
	r := declineFixture(1, "int4")
	chunk := prefillChunkRows()

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
