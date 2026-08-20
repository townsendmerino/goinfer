//go:build gpu

package gpu

import (
	"strings"
	"testing"
)

// TestAttnHeadDimSupported_M12 gates M-12: newDecodeRunner must decline a resident plan whose
// head_dim exceeds what the single-query attention kernels can dot, at both the model level and
// any per-layer geometry override (Gemma 4). Before the guard an admitted arch with an oversized
// head_dim silently dotted only the first workgroup's worth of dims and the o-projection consumed
// half-zero context.
//
// The bound used to be the 128-wide workgroup. It is now the WIDE kernel's stride reach, which is
// 2048 — head_dim is no longer a practical reason to decline. The guard stays because the failure
// mode it protects against (silent truncation, no error) has not changed, only moved.
func TestAttnHeadDimSupported_M12(t *testing.T) {
	// The invariant is no longer "== the workgroup width". The wide kernel strides, so its reach
	// is width × dims-per-lane, and attnMaxHeadDim must track THAT product — if a future change
	// shrinks attnMaxPerLane without shrinking the limit, the predicate would admit a head_dim
	// the kernel's per-lane array cannot hold.
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
