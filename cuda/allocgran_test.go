//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestAllocGranularity records what this driver actually charges for a device allocation. It exists
// because the expert-cache sizing arithmetic must predict device consumption, and summing requested
// bytes does not.
//
// MEASURED (RTX 2070 SUPER, driver 595.58.03, 2026-08-11) — the numbers, not just the conclusion:
//
//	request         actual/alloc   overhead
//	1 048 576 B ->   1 048 576 B     +0.0%
//	1 048 577 B ->   2 097 152 B   +100.0%
//	1 052 672 B ->   2 097 152 B    +99.2%
//	2 973 696 B ->   4 194 304 B    +41.0%   (int4 weights, one 26B expert)
//	3 490 000 B ->   4 194 304 B    +20.2%
//
// Those four large samples ALL sit just above a power of two, where next-power-of-two and 2 MiB
// granularity predict identically — so they cannot separate the two hypotheses, and reading them as
// "rounds to the next power of two" was a name asserted from data that did not constrain it. The
// discriminating requests are 5 / 6 / 9 MiB:
//
//	5 MiB -> actual  6.00 MiB    nextPow2 says  8    2 MiB-granular says  6
//	6 MiB -> actual  6.00 MiB    nextPow2 says  8    2 MiB-granular says  6
//	9 MiB -> actual 10.00 MiB    nextPow2 says 16    2 MiB-granular says 10
//
// Unanimous for 2 MiB granularity. nextPow2 would over-charge by up to 2x on any buffer that does
// not happen to sit just above a power of two, under-granting slots on a future geometry — a new
// mis-estimate introduced by the fix for the old one.
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
		for i := 0; i < K; i++ {
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

// TestSmallAllocPool records that sub-granularity allocations are NOT free, which a single
// allocation appears to show and cannot: one 371 712-byte request measured ZERO device bytes,
// because the pool page had already been charged.
//
// MEASURED, same box, 371 712 B (f16 scales for one 26B expert):
//
//	after   1 alloc  ->   2.00 MiB total   (the page, charged once)
//	after 2..4       ->   +0.00 MiB        (drawn down, marginal cost zero)
//	after  64        ->  +24.00 MiB
//	after 128..512   ->  +26.00 MiB per 64
//	512 allocs       -> 206.00 MiB total = 421 888 B/alloc amortised
//
// So the marginal cost is zero until the page exhausts and then it steps. 421 888 B amortised
// against a 371 712 B request is the honest figure at the counts that matter (30 layers x N slots).
func TestSmallAllocPool(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	const sz, n = 371712, 512
	base, _, _ := dev.Context().MemInfo()
	for i := 0; i < n; i++ {
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
