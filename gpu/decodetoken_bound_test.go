//go:build gpu

package gpu

import (
	"strings"
	"testing"
)

// TestDecodeTokenFusedBatched_Mbound_C13 gates C-13: the thin-M gemmRow kernel accumulates into a
// private array<i32, gemmRowMaxM>, so a block of M > gemmRowMaxM rows would silently alias the last
// accumulator under WGSL robustness clamping — wrong logits, no error. The entry point must reject
// it. The bound check runs before any device call, so no adapter is needed.
func TestDecodeTokenFusedBatched_Mbound_C13(t *testing.T) {
	var c *Context // the M>gemmRowMaxM guard returns before touching the device
	xs := make([][]float32, gemmRowMaxM+1)
	_, err := c.DecodeTokenFusedBatched(xs, ModelW{}, 8, 4, 2, 2, 16, make([]int, gemmRowMaxM+1), 0, 1e-6, 0.1, false)
	if err == nil {
		t.Fatalf("M=%d (> gemmRowMaxM=%d) accepted — C-13: must reject to avoid accumulator aliasing", gemmRowMaxM+1, gemmRowMaxM)
	}
	if !strings.Contains(err.Error(), "gemmRowMaxM") {
		t.Errorf("wrong error for oversized M: %v", err)
	}
}
