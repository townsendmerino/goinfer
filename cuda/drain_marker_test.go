//go:build cuda

package cuda

import (
	"os"
	"testing"
)

// drainsDevice is the MARKER for the tier partition, and it is a shared helper rather than a naming
// convention or a hand-kept -run list for one reason: a list of names is a constant that restates a
// property, and it drifts silently. The census-denominator work made that shape visible four times
// over — a check reports its numerator and stays green while its universe shrinks. A `-run
// 'TestAllocFloor|TestA10ReportingGap|…'` in the GPU gate is exactly that constant. This is the
// property itself, in the code that has it.
//
// WHAT IT MARKS. A test that deliberately drives the device to REFUSAL — allocates until even a
// small request fails — or holds it near the floor while a live context keeps working. That is A13's
// only reproducible poisoning stimulus (5/5), and it is a property of the test, not of its name.
//
// IT DOES NOT ONLY LABEL, IT ENFORCES. Deriving a group and trusting everyone to run it is the same
// advisory-comment failure as before, so the marker is also the gate: without GOINFER_DRAIN_GROUP a
// marked test SKIPS. A drainer can therefore never execute inside the main tier even if the shell's
// derivation misses it, and the skip line it prints is what the reconciliation counts.
//
// WHAT THE RECONCILIATION DOES AND DOES NOT CATCH — corrected 2026-08-13, after the first gate run
// caught a case this comment had claimed was impossible. It said "the partition cannot silently drop
// a test into neither half". That is true only of MARKED tests: a derivation miss shows up as a
// main-tier skip with no matching drain-tier run, and the gate fails on the mismatch. It says
// NOTHING about a drainer that never calls this helper at all — that test is invisible to both the
// derivation and the reconciliation, and runs in the main tier as if it were harmless.
//
// That is not hypothetical. The very first gate run after this marker landed went red on
// TestMoERouteDemandThreshold, which balloons the device to as little as 64 MiB free — an unmarked
// drainer, exactly the blind spot. It is marked now. The blind spot is not closed by marking it, so:
// COVERAGE HERE IS BY INSPECTION, and the honest statement is that this helper enforces the
// partition for tests someone remembered to mark, and nothing more.
//
// The child processes spawned by TestA10FloorIsPerProcessOrPerDevice inherit the environment
// (cmd.Env = append(os.Environ(), …)), so they inherit the flag with it.
//
// TO ADD A DRAINER: call this first thing in the test. Nothing else. `gate gpu` derives the group
// by scanning for calls to this function, so there is no second place to update and no list to
// forget.
func drainsDevice(t *testing.T, why string) {
	t.Helper()
	if os.Getenv("GOINFER_DRAIN_GROUP") == "" {
		// The exact token DRAIN-GROUP-SKIP is what `gate gpu` counts to reconcile the split. Keep
		// it, and keep it greppable; the gate fails loudly rather than silently if it disappears.
		t.Skipf("DRAIN-GROUP-SKIP: %s — deferred to the drain tier (GOINFER_DRAIN_GROUP=1), which "+
			"runs in its own process so an exhausted device cannot reach the main tier. See A13.", why)
	}
	t.Logf("DRAIN-GROUP-RUN: %s", why)
}
