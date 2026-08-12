//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestAllocFloor measures how far cuMemAlloc will actually drain the device, against what
// cuMemGetInfo reports as free at that moment.
//
// A10's ordering hypothesis predicted that issuing the largest slot buffers first would let the
// sequence complete. It did not. It got 27 MiB further (failing with 155,385,856 B free instead of
// 182,648,832) and allocated more total bytes, but still failed — and the failing request was
// 4,212,736 B against 155,385,856 B free, a ratio of 36.88. A 4 MiB request refused with 148 MiB
// free is not a contiguity story.
//
// Both failures sit in the same band regardless of request size, which reads as a FLOOR: some
// quantity cuMemGetInfo counts as free that cuMemAlloc will not hand out. This measures it directly,
// with no model and no 26B: drain in shrinking chunks until even a 2 MiB request is refused, then
// report what free says.
func TestAllocFloor(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	read := func() int64 { f, _, _ := dev.Context().MemInfo(); return int64(f) }
	var hold []gpu.Buffer
	alloc := func(n int) (ok bool) {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		hold = append(hold, gpu.NewBufferLenOf[byte](dev, n))
		return true
	}
	start := read()
	// Drain with shrinking request sizes, so the last refusals are of genuinely small blocks. A
	// single size would conflate "this size does not fit" with "nothing fits".
	var lastOK int64
	for size := int64(1) << 30; size >= (1 << 20); size /= 2 {
		for alloc(int(size)) {
			lastOK = read()
		}
		t.Logf("  %6d KiB blocks exhausted; free now %13d B (%.1f MiB)",
			size>>10, read(), float64(read())/(1<<20))
	}
	floor := read()
	t.Logf("start free      %13d B (%.1f MiB)", start, float64(start)/(1<<20))
	t.Logf("free after last SUCCESSFUL alloc %13d B", lastOK)
	t.Logf("FLOOR: free reported when even a 1 MiB request is refused: %d B (%.1f MiB)",
		floor, float64(floor)/(1<<20))
	t.Logf("  for comparison — allocSlots failures on the real 26B:")
	t.Logf("    group-by-group order: refused 67,403,776 B with 182,648,832 B free")
	t.Logf("    largest-first order:  refused  4,212,736 B with 155,385,856 B free")

	if floor <= 0 {
		t.Fatal("free read as zero — the instrument did not run")
	}
	// Not a threshold assertion: the number is the finding. What WOULD be a defect is the drain
	// reaching zero, which would mean there is no floor and the 26B failures need another cause.
	if floor < (1 << 20) {
		t.Logf("  => free drains essentially to zero, so there is NO reserve and the 26B allocation " +
			"failures are not explained by a floor — the ordering/contiguity account must be revisited")
	} else {
		t.Logf("  => %.1f MiB is reported free but cannot be allocated, at ANY request size down to "+
			"1 MiB. That is a reserve, not fragmentation, and it is the quantity allocSlots must "+
			"treat as unavailable.", float64(floor)/(1<<20))
	}

	// THE RELATIONSHIP, pinned. Leftover after allocSlots must clear the floor, and the margin is
	// what guarantees it. Every observation fits:
	//
	//	cap 31 -> leftover 501,415,936  > floor  -> works
	//	cap 33 -> leftover 312,672,256  > floor  -> works
	//	cap 34 -> leftover  61,014,016  < floor  -> fails mid-allocation
	//
	// It also retires a figure A9-MARGIN nearly recommended. A 128 MiB margin (134,217,728) is BELOW
	// this floor; the cap-33 run under it worked only because that cap's leftover happened to be
	// 312 MiB. That was luck, not safety, and the assertion below is what turns the distinction into
	// something a test can see.
	if int64(slotMarginBytes) < floor {
		t.Errorf("slotMarginBytes (%d) is below the measured allocation floor (%d). Free VRAM "+
			"overstates allocatable VRAM by that much, so the cap can be granted at a size whose "+
			"allocations fail PART WAY THROUGH — declining to the staged path after having already "+
			"claimed most of the device", int64(slotMarginBytes), floor)
	}
	t.Logf("margin check: slotMarginBytes %d >= floor %d, clear by %d B (%.1f MiB)",
		int64(slotMarginBytes), floor, int64(slotMarginBytes)-floor,
		float64(int64(slotMarginBytes)-floor)/(1<<20))
	_ = hold
}
