//go:build cuda

package cuda

import (
	"os"
	"testing"
)

// drainsDevice is the MARKER for the tier partition. It is a shared helper rather than a naming convention or a
// hand-kept -run list because a list of names is a constant that restates a property, and it drifts silently:
// a check reports its numerator and stays green while its universe shrinks. This is the property itself, in the
// code that has it.
//
// WHAT IT MARKS. A test that deliberately drives the device to REFUSAL (allocates until even a small request
// fails) or holds it near the floor while a live context keeps working. That is A13's only reproducible
// poisoning stimulus, and it is a property of the test, not of its name.
//
// IT DOES NOT ONLY LABEL, IT ENFORCES: without GOINFER_DRAIN_GROUP a marked test SKIPS. A drainer can therefore
// never execute inside the main tier even if the shell's derivation misses it, and the skip line it prints is
// what the reconciliation counts.
//
// WHAT THE RECONCILIATION DOES AND DOES NOT CATCH. A derivation miss on a MARKED test shows up as a main-tier
// skip with no matching drain-tier run, and the gate fails on the mismatch. It says NOTHING about a drainer
// that never calls this helper: that test is invisible to both the derivation and the reconciliation and runs
// in the main tier as if it were harmless (TestMoERouteDemandThreshold was one until it was marked). COVERAGE
// HERE IS BY INSPECTION: this helper enforces the partition for tests someone remembered to mark, and nothing
// more.
//
// The child processes spawned by TestA10FloorIsPerProcessOrPerDevice inherit the environment (cmd.Env =
// append(os.Environ(), …)), so they inherit the flag with it.
//
// TO ADD A DRAINER: call this first thing in the test. Nothing else. `gate gpu` derives the group by scanning
// for calls to this function, so there is no second place to update and no list to forget.
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
