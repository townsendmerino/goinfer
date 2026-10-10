//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestAllocFloor measures how far cuMemAlloc will actually drain the device, against what cuMemGetInfo
// reports as free at that moment. The shortfall is a FLOOR, not contiguity: some quantity cuMemGetInfo
// counts as free that cuMemAlloc will not hand out, at any request size. No model, no 26B: it drains in
// shrinking chunks until even a 2 MiB request is refused, then reports what free says. Record:
// docs/queue-engineering.md, section A10; the figures that led to it: docs/code-notes/cuda.md#TestAllocFloor.
func TestAllocFloor(t *testing.T) {
	drainsDevice(t, "drains the device in shrinking chunks until a 1 MiB request is refused")
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	read := func() int64 { f, _, _ := dev.Context().MemInfo(); return int64(f) }
	var hold []gpu.Buffer
	// FREE THE DRAIN, ON EVERY EXIT PATH. This test allocates until the device refuses a 2 MiB request and
	// holds every buffer in `hold` so the GC cannot reclaim one; without an explicit release an EXHAUSTED
	// device carries into every later test in the package (it made TestAllocGranularity fail with
	// CUDA_ERROR_OUT_OF_MEMORY, which the GPU gate reported as "a CUDA forward moved"). It is a defer rather
	// than a tail block because the foreign-context skip below returns via runtime.Goexit and would jump
	// past a trailing release.
	defer func() {
		for _, b := range hold {
			dev.ReleaseBuf(b)
		}
		hold = nil
	}()
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

	// The floor is pinned to a WINDOW, not a byte: it is a machine property (driver, display stack, device
	// state), and a byte-exact pin on one is brittle. Re-derive on a new box or a driver change, and re-check
	// the demand identity of TestMoERouteDemandThreshold (demand == floor + residual), which is downstream of
	// this number: an unpinned number another gate depends on stops being watched, and when this floor
	// moved unpinned that gate went red accusing the kernel. A smaller floor means more headroom than the cap
	// analysis assumed, and the window's lower edge now falls below zero, so the bottom is guarded by the
	// `floor <= 0` fatal above and "moved" can only mean "grew". Re-derivation record:
	// docs/code-notes/cuda.md#TestAllocFloor.pin.
	const (
		pinnedFloor = 1769472 // measured 2026-08-26, RTX 2070 SUPER, driver 595.91.07 / Nobara 44
		floorWindow = 4 << 20 // the quantity is a driver reserve, not a program's
	)
	// Kept so a REVERSION to an earlier floor is recognisable rather than reading as a move somewhere new.
	previous := []struct {
		val   int64
		label string
	}{
		{54263808, "the 2026-08-21 value (54,263,808), i.e. the pre-re-anchor driver 595.58.03 behaviour"},
		{151191552, "the pre-2026-08-21 value (151,191,552)"},
	}
	if floor < pinnedFloor-floorWindow || floor > pinnedFloor+floorWindow {
		hint := ""
		for _, p := range previous {
			if floor > p.val-floorWindow && floor < p.val+floorWindow {
				hint = " — this is " + p.label + ", so the machine has gone BACK rather than moving somewhere new"
			}
		}
		// The pin is a property of an EXCLUSIVE device: a foreign CUDA context raises this floor, so asserting
		// the exclusive number against a desktop session would report a machine change that has not happened
		// and send the reader toward TestMoERouteDemandThreshold and the A1/A5/A7/A9 pins over a window manager.
		// So SKIP rather than fail, and say why. A skip is not a pass (`gate gpu` lists it as uncovered), whereas
		// a red that everyone learns to expect on a desktop is a gate nobody reads.
		foreign, known := foreignCUDAContexts()
		if !known {
			t.Skipf("floor measured %d B, but cannot determine whether another CUDA context is "+
				"alive (no nvidia-smi). The pin is only meaningful on an exclusive device, and "+
				"guessing that is what this check replaced.", floor)
		}
		if len(foreign) > 0 {
			t.Skipf("floor measured %d B against a pin of %d B, but ANOTHER CUDA CONTEXT IS ALIVE "+
				"— %s. A foreign context RAISES this floor (+16 MiB measured for KDE's compositor "+
				"on 2026-09-01, and the demand identity still closed to the byte against the "+
				"measured floor), so this is not a machine change. Re-run with an exclusive device "+
				"— e.g. from a TTY with the compositor stopped — to test the pin.",
				floor, int64(pinnedFloor), describeForeign(foreign))
		}
		t.Errorf("the device allocation floor moved: measured %d B, pinned %d±%d B%s. This is a "+
			"MACHINE property (driver, display stack, device state), not a property of this repo, so "+
			"the first question is what changed underneath rather than what changed in the tree. "+
			"TestMoERouteDemandThreshold's identity (demand == floor + residual) is DOWNSTREAM of "+
			"this number and will go red too; re-derive it from the new floor rather than editing "+
			"its pin.", floor, int64(pinnedFloor), int64(floorWindow), hint)
	}

	// What WOULD be a defect is the drain reaching zero, which would mean there is no floor and the
	// 26B failures need another cause.
	if floor < (1 << 20) {
		t.Logf("  => free drains essentially to zero, so there is NO reserve and the 26B allocation " +
			"failures are not explained by a floor — the ordering/contiguity account must be revisited")
	} else {
		t.Logf("  => %.1f MiB is reported free but cannot be allocated, at ANY request size down to "+
			"1 MiB. That is a reserve, not fragmentation, and it is the quantity allocSlots must "+
			"treat as unavailable.", float64(floor)/(1<<20))
	}

	// THE RELATIONSHIP, pinned: the leftover after allocSlots must clear the floor, and the margin is what
	// guarantees it. A margin below the floor works only when a cap's leftover happens to be large, which is
	// luck, not safety; the assertion below is what makes the distinction visible.
	if int64(slotMarginBytes) < floor {
		t.Errorf("slotMarginBytes (%d) is below the measured allocation floor (%d). Free VRAM "+
			"overstates allocatable VRAM by that much, so the cap can be granted at a size whose "+
			"allocations fail PART WAY THROUGH — declining to the staged path after having already "+
			"claimed most of the device", int64(slotMarginBytes), floor)
	}
	t.Logf("margin check: slotMarginBytes %d >= floor %d, clear by %d B (%.1f MiB)",
		int64(slotMarginBytes), floor, int64(slotMarginBytes)-floor,
		float64(int64(slotMarginBytes)-floor)/(1<<20))
}
