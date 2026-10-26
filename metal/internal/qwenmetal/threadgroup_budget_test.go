//go:build darwin

package qwenmetal

import (
	"testing"

	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

// TestAttnThreadgroupBytes covers the audit C-02 guard's arithmetic without a
// checkpoint. The three qwenmetal suite tests all skip when
// testdata/qwen25vl-vision-tiny is absent, so the guard in ForwardViT would
// otherwise ship with zero executed coverage — which is the shape of defect the
// audit's own G-07 is about.
func TestAttnThreadgroupBytes(t *testing.T) {
	const apple = 32 << 10 // ~32 KiB, what Apple GPUs report

	// The static pair is always present, even at a zero-length segment.
	if got, want := attnThreadgroupBytes(0), 2*gpu.ViTBlock*4; got != want {
		t.Errorf("attnThreadgroupBytes(0) = %d, want %d (the two static arrays)", got, want)
	}
	// Dynamic term is 4 bytes per segment element on top of that.
	if got, want := attnThreadgroupBytes(1000), 1000*4+2*gpu.ViTBlock*4; got != want {
		t.Errorf("attnThreadgroupBytes(1000) = %d, want %d", got, want)
	}

	// The boundary against a 32 KiB budget. With ViTBlock=256 the static pair is
	// 2048 B, leaving 30720 B => the largest admissible segment is 7680, NOT the
	// 8192 a dynamic-only reading gives. Both sides of the edge are pinned so a
	// change to ViTBlock or to the kernel's static arrays fails here loudly.
	limit := (apple - 2*gpu.ViTBlock*4) / 4
	if attnThreadgroupBytes(limit) > apple {
		t.Errorf("segment %d should fit in %d B, needs %d", limit, apple, attnThreadgroupBytes(limit))
	}
	if attnThreadgroupBytes(limit+1) <= apple {
		t.Errorf("segment %d should NOT fit in %d B, needs %d", limit+1, apple, attnThreadgroupBytes(limit+1))
	}
	if gpu.ViTBlock == 256 && limit != 7680 {
		t.Errorf("expected the 32 KiB limit to be a 7680 segment, got %d", limit)
	}
}
