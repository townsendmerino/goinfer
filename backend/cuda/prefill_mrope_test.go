//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"testing"
)

// TestPrefillMRoPELast_declinesWithoutKernel pins the family-gate: a fixture with no m-RoPE
// kernel loaded (mropePrefillReady false, the default zero value — mirrors every non-Qwen-VL
// family) must decline cleanly, not panic on a nil pipeline. Needs no device: the readiness check
// runs before prefillCore ever touches the executor.
func TestPrefillMRoPELast_declinesWithoutKernel(t *testing.T) {
	r := declineFixture(1, "int4")
	embeddings := [][]float32{{0}, {0}, {0}}
	mropePos := [][3]int{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}
	_, err := r.PrefillMRoPELast(context.Background(), embeddings, 0, mropePos)
	if err == nil {
		t.Fatal("mropePrefillReady=false must decline, not silently proceed to a nil pipeline launch")
	}
	if !errors.Is(err, errPrefillDeclined) {
		t.Errorf("decline must wrap errPrefillDeclined so GenerateQwenVL can fall back to the CPU-prefill+UploadKV bridge cleanly: %v", err)
	}
}

// TestPrefillMRoPELast_declinesOnLengthMismatch pins the other input-validation half: mropePos is
// prompt-absolute and must cover [0, startPos+M) exactly — a caller passing a mismatched length
// (chunk-relative by mistake, or simply the wrong slice) must decline rather than index out of
// bounds inside prefillCore/mropePosWindow.
func TestPrefillMRoPELast_declinesOnLengthMismatch(t *testing.T) {
	r := declineFixture(1, "int4")
	embeddings := [][]float32{{0}, {0}, {0}}
	for _, tc := range []struct {
		name     string
		mropePos [][3]int
	}{
		{"too short", [][3]int{{0, 0, 0}, {1, 1, 1}}},
		{"too long", [][3]int{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}}},
		{"empty", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := r.PrefillMRoPELast(context.Background(), embeddings, 0, tc.mropePos)
			if err == nil {
				t.Fatalf("mropePos len %d against %d embeddings must decline", len(tc.mropePos), len(embeddings))
			}
			if !errors.Is(err, errPrefillDeclined) {
				t.Errorf("decline must wrap errPrefillDeclined: %v", err)
			}
		})
	}
}

// TestPrefillMRoPELast_declinesOnEmptyPrompt mirrors the empty-prompt guard every other prefill
// entrypoint in this file carries.
func TestPrefillMRoPELast_declinesOnEmptyPrompt(t *testing.T) {
	r := declineFixture(1, "int4")
	_, err := r.PrefillMRoPELast(context.Background(), nil, 0, nil)
	if err == nil {
		t.Fatal("an empty prompt must be rejected, not proceed to a zero-row launch")
	}
}
