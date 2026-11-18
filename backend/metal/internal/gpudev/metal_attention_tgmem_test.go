//go:build darwin

package gpudev

import "testing"

// TestAttentionThreadgroupBytes pins the budget arithmetic of the shared `attention` and `attention_seg` kernels without a
// device or checkpoint: the static pair is always present, the dynamic term is 4 bytes per element, and against a 32 KiB
// budget the largest admissible score row is 7680 (the static 2 KiB, not the dynamic term alone, sets the edge).
func TestAttentionThreadgroupBytes(t *testing.T) {
	const apple = 32 << 10

	if got, want := AttentionThreadgroupBytes(0), 2*ViTBlock*4; got != want {
		t.Errorf("AttentionThreadgroupBytes(0) = %d, want %d (the two static arrays)", got, want)
	}
	if got, want := AttentionThreadgroupBytes(1000), 1000*4+2*ViTBlock*4; got != want {
		t.Errorf("AttentionThreadgroupBytes(1000) = %d, want %d", got, want)
	}
	limit := (apple - 2*ViTBlock*4) / 4
	if AttentionThreadgroupBytes(limit) > apple {
		t.Errorf("n=%d should fit in %d B, needs %d", limit, apple, AttentionThreadgroupBytes(limit))
	}
	if AttentionThreadgroupBytes(limit+1) <= apple {
		t.Errorf("n=%d should NOT fit in %d B, needs %d", limit+1, apple, AttentionThreadgroupBytes(limit+1))
	}
	if ViTBlock == 256 && limit != 7680 {
		t.Errorf("expected the 32 KiB limit to be n=7680, got %d", limit)
	}
}

// TestAttentionThreadgroupBytes_deviceBudget checks the arithmetic against the device's own limit: the 7680-element edge
// fits, and an Apple-sized (32 KiB) device refuses one element past it.
func TestAttentionThreadgroupBytes_deviceBudget(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })
	have := d.MaxThreadgroupMemoryLength()
	if need := AttentionThreadgroupBytes(7680); need > have {
		t.Errorf("a 7680-element score row needs %d B but the device allows %d", need, have)
	}
	if have == 32<<10 && AttentionThreadgroupBytes(7681) <= have {
		t.Errorf("one past the edge should exceed a 32 KiB budget, needs %d", AttentionThreadgroupBytes(7681))
	}
}
