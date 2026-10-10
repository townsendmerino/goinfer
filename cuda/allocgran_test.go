//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
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
