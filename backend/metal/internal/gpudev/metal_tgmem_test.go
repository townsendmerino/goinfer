//go:build darwin

package gpudev

import "testing"

// TestMaxThreadgroupMemoryLength sanity-checks the device tile-memory accessor used by goinfer's
// M-11 over-budget decline: it must report a plausible byte limit (Apple GPUs are ~32 KiB).
func TestMaxThreadgroupMemoryLength(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	got := d.MaxThreadgroupMemoryLength()
	// 16 KiB..1 MiB brackets every real Apple GPU without pinning an exact value.
	if got < 16*1024 || got > 1<<20 {
		t.Fatalf("MaxThreadgroupMemoryLength = %d bytes, implausible (expected ~32 KiB)", got)
	}
	t.Logf("device tile-memory limit: %d bytes", got)
}
