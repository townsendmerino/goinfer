//go:build gpu

package gpu

import (
	"strings"
	"testing"
)

// TestAttnHeadDimSupported_M12 gates M-12: newDecodeRunner must decline a resident plan whose
// head_dim exceeds the single-query attention kernel's 128-wide workgroup, at both the model
// level and any per-layer geometry override (Gemma 4). Before the guard an admitted arch with
// head_dim>128 silently dotted only dims 0..127 and the o-projection consumed half-zero context.
// Pure predicate → no device needed; the boundary (128 ok, 129 declined) proves it neither
// over- nor under-rejects.
func TestAttnHeadDimSupported_M12(t *testing.T) {
	if attnMaxHeadDim != 128 {
		t.Fatalf("attnMaxHeadDim=%d but the attention kernels are @workgroup_size(128) — keep them equal", attnMaxHeadDim)
	}
	if err := attnHeadDimSupported(128, nil); err != nil {
		t.Errorf("head_dim 128 (== workgroup) must be supported, got %v", err)
	}
	if err := attnHeadDimSupported(64, []runLayer{{ghd: 128}, {ghd: 0}}); err != nil {
		t.Errorf("per-layer head_dim 128 must be supported, got %v", err)
	}
	if err := attnHeadDimSupported(256, nil); err == nil {
		t.Error("model head_dim 256 must be declined (M-12)")
	} else if !strings.Contains(err.Error(), "head_dim=256") {
		t.Errorf("wrong decline message: %v", err)
	}
	if err := attnHeadDimSupported(64, []runLayer{{ghd: 0}, {ghd: 256, gnKV: 2, ghalf: 128}}); err == nil {
		t.Error("per-layer head_dim 256 must be declined (M-12)")
	} else if !strings.Contains(err.Error(), "layer 1") {
		t.Errorf("per-layer decline should name the layer: %v", err)
	}
}
