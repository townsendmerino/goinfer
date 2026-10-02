//go:build darwin

package metal

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestF16ScaleRoundTrip_exhaustive is why C-P01 (docs/audit-metal-2026-09-30.md) is bit-identical on every checkpoint,
// not just on the fixtures the paging tests load. The paged MoE pagers used to build a heap cache of each expert's f16
// scales by widening the WeightMat's binary16 storage to f32 (decoder.Int4F32, which calls linalg.F16ToF32Slice) and
// narrowing it back (f32ToF16, as parallelF32ToF16 does). They now stage the WeightMat's storage directly. The two
// agree on every scale iff that round trip returns the same bits for every binary16 value, which this checks for all
// 65,536 of them. NaN payloads are exempt: a NaN group scale is a broken checkpoint, and the count is logged.
func TestF16ScaleRoundTrip_exhaustive(t *testing.T) {
	src := make([]uint16, 1<<16)
	for i := range src {
		src[i] = uint16(i)
	}
	wide := make([]float32, len(src))
	linalg.F16ToF32Slice(wide, src)
	checked, nan := 0, 0
	for i, h := range src {
		if h&0x7c00 == 0x7c00 && h&0x03ff != 0 {
			nan++
			continue
		}
		checked++
		if got := f32ToF16(wide[i]); got != h {
			t.Fatalf("f16 %#04x widens to %v and narrows to %#04x: the old scale cache would differ from the WeightMat's own scales", h, wide[i], got)
		}
	}
	if checked != 1<<16-2046 {
		t.Fatalf("checked %d values, want %d (every binary16 but the 2046 NaNs)", checked, 1<<16-2046)
	}
	t.Logf("%d binary16 values round-trip exactly (%d NaN payloads exempt)", checked, nan)
}
