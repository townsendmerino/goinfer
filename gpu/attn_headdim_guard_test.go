//go:build gpu

package gpu

import (
	"strings"
	"testing"
)

// TestAttnHeadDimSupported_M12 pins that newDecodeRunner declines a resident plan whose head_dim
// exceeds what the single-query attention kernels can dot, at the model level and for any
// per-layer geometry override (Gemma 4). Without the guard an oversized head_dim silently dots
// only the first workgroup's worth of dims and the o-projection consumes half-zero context. The
// bound is now the wide kernel's stride reach, no longer a practical limit; the guard stays
// because silent truncation with no error is the failure it prevents. Background:
// docs/code-notes/gpu.md#TestAttnHeadDimSupported_M12.
func TestAttnHeadDimSupported_M12(t *testing.T) {
	// The invariant is attnMaxHeadDim == attnWGWide*attnMaxPerLane, the wide kernel's stride reach:
	// if attnMaxPerLane shrinks without the limit, the predicate would admit a head_dim the kernel's
	// per-lane array cannot hold.
	if attnMaxHeadDim != attnWGWide*attnMaxPerLane {
		t.Fatalf("attnMaxHeadDim=%d but the wide kernel reaches %d×%d — keep them equal",
			attnMaxHeadDim, attnWGWide, attnMaxPerLane)
	}
	for _, hd := range []int{64, attnWG, attnWGWide, 512, attnMaxHeadDim} {
		if err := attnHeadDimSupported(hd, nil); err != nil {
			t.Errorf("head_dim %d must be supported, got %v", hd, err)
		}
		if err := attnHeadDimSupported(64, []runLayer{{ghd: hd}, {ghd: 0}}); err != nil {
			t.Errorf("per-layer head_dim %d must be supported, got %v", hd, err)
		}
	}
	// One past the reach is still declined, so the predicate neither over- nor under-rejects.
	over := attnMaxHeadDim + 1
	if err := attnHeadDimSupported(over, nil); err == nil {
		t.Errorf("model head_dim %d must be declined (M-12)", over)
	} else if !strings.Contains(err.Error(), "head_dim=2049") {
		t.Errorf("wrong decline message: %v", err)
	}
	if err := attnHeadDimSupported(64, []runLayer{{ghd: 0}, {ghd: over, gnKV: 2, ghalf: 128}}); err == nil {
		t.Errorf("per-layer head_dim %d must be declined (M-12)", over)
	} else if !strings.Contains(err.Error(), "layer 1") {
		t.Errorf("per-layer decline should name the layer: %v", err)
	}
}
