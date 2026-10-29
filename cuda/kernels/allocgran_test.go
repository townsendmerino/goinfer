//go:build cuda

package kernels

import (
	"testing"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
)

// TestAllocGranularity records what this driver actually charges for a device allocation, because the
// expert-cache sizing arithmetic must predict device consumption and summing requested bytes does not.
// The charge is 2 MiB granular, not next-power-of-two: the discriminating requests are 5 / 6 / 9 MiB
// (2 MiB granularity charges 6 / 6 / 10, next-power-of-two would charge 8 / 8 / 16), and the test fails
// when the quantum changes. Modelling it as a power of two would over-charge by up to 2x and under-grant
// slots. Measured table: docs/code-notes/cuda.md#TestAllocGranularity.
func TestAllocGranularity(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	const K = 32
	MiB := int64(1) << 20
	for _, mib := range []int64{5, 6, 9} {
		sz := mib * MiB
		before, _, _ := dev.Context().MemInfo()
		for range K {
			_ = gpu.NewBufferLenOf[byte](dev, int(sz))
		}
		after, _, _ := dev.Context().MemInfo()
		actual := int64(before) - int64(after)
		perAlloc := actual / K
		want := ((sz + 2*MiB - 1) / (2 * MiB)) * 2 * MiB
		pow2 := int64(1)
		for pow2 < sz {
			pow2 <<= 1
		}
		t.Logf("%d MiB -> %.2f MiB/alloc (2MiB-granular predicts %d, nextPow2 %d)",
			mib, float64(perAlloc)/float64(MiB), want/MiB, pow2/MiB)
		if perAlloc != want {
			t.Errorf("%d MiB request charged %d B, 2 MiB granularity predicts %d B — the driver's "+
				"allocation quantum changed, and slotBytesPerLayer's model is built on it", mib, perAlloc, want)
		}
	}
}

// TestSmallAllocPool records that sub-granularity allocations are NOT free, which a single allocation
// appears to show and cannot: a pool page is charged once, so the marginal cost is zero until the page
// exhausts and then steps. The amortised cost per request is above the request size at the counts that
// matter (layers x slots), and the test fails if it is not. Measured table: docs/code-notes/cuda.md#TestSmallAllocPool.
func TestSmallAllocPool(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	const sz, n = 371712, 512
	base, _, _ := dev.Context().MemInfo()
	for range n {
		_ = gpu.NewBufferLenOf[byte](dev, sz)
	}
	final, _, _ := dev.Context().MemInfo()
	per := float64(int64(base)-int64(final)) / n
	t.Logf("%d allocs of %d B -> %.2f MiB total, %.0f B/alloc amortised",
		n, sz, float64(int64(base)-int64(final))/(1<<20), per)
	if per <= float64(sz) {
		t.Errorf("amortised cost %.0f B <= request %d B — sub-granularity allocations are being "+
			"treated as free, which the 512-allocation step measurement refutes", per, sz)
	}
}

// TestSmallAllocPool_sharedAcrossDevices records why a device's free-memory delta can fall below the bytes allocated on it:
// devices in one process share the primary context, and with it the small-allocation pool. A device created while another
// holds a pool page places its own small buffers in that page's free space, which moves free memory by nothing, while a
// buffer of the quantum moves it by exactly its size. Measured table: docs/measurements/sb12-memtests-2026-10-28/.
func TestSmallAllocPool_sharedAcrossDevices(t *testing.T) {
	first, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer first.ReleaseObjects()
	const small, n = 4096, 64
	for range 4 { // an earlier test's small buffers: they open a pool page and leave most of it free
		_ = gpu.NewBufferLenOf[byte](first, small)
	}
	second, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer second.ReleaseObjects()
	free := func() int64 {
		f, _, err := second.Context().MemInfo()
		if err != nil {
			t.Fatalf("MemInfo: %v", err)
		}
		return int64(f)
	}
	base := free()
	for range n {
		_ = gpu.NewBufferLenOf[byte](second, small)
	}
	smallMoved := base - free()
	base = free()
	_ = gpu.NewBufferLenOf[byte](second, gpu.AllocQuantumBytes)
	bigMoved := base - free()
	t.Logf("%d buffers of %d B (%d B) on a second device moved free memory by %d B; one %d-byte buffer moved it by %d B",
		n, small, n*small, smallMoved, gpu.AllocQuantumBytes, bigMoved)
	if bigMoved != gpu.AllocQuantumBytes {
		t.Errorf("a buffer of the quantum moved free memory by %d B, want %d", bigMoved, gpu.AllocQuantumBytes)
	}
	if smallMoved != 0 {
		t.Errorf("the second device's %d B of small buffers moved free memory by %d B, want 0: the pool page the first "+
			"device opened was not shared", n*small, smallMoved)
	}
}
