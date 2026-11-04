//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
)

// needsFreshProcess is the second tier-partition MARKER, drainsDevice's sibling (drain_marker_test.go
// explains why a marker and not a -run list). It marks a test whose model sits at the EDGE of the card —
// it fits with tens of MB to spare in a fresh process, and does not after a few hundred other tests have
// run in the same one.
//
// WHY IT EXISTS. The GPU gate's heavy tier runs the whole package in one process, and real-model
// tests whose model sits at the edge of the card fail there with a resident decline or
// CUDA_ERROR_OUT_OF_MEMORY while passing as their own process. This is NOT a leak (A12, docs/QUEUE.md,
// refuted both a leak and parallelism) and the in-process shortfall is still unexplained. The marker
// does not explain it either; it keeps an unexplained tens-of-MB drift from turning a correct test red.
//
// THE SEMANTICS ARE INVERTED FROM drainsDevice ON PURPOSE. A drainer must never run beside anything, so
// it skips unless asked. A fresh-process test is harmless anywhere; it only needs room. So it RUNS unless
// the gate's main tier asks it to defer (GOINFER_ISOLATE_DEFER=1), and an ad-hoc `go test -run` of it
// behaves as it always did. The gate then runs each marked test as its own process and reconciles the
// ISOLATED-SKIP / ISOLATED-RUN tokens both ways, so a test cannot fall into neither half.
//
// TO ADD ONE: call it first thing in the test, with the reason. `gate gpu` finds it by scanning for the
// call. Coverage is by inspection, as with drainsDevice: an unmarked edge-of-card test still runs in the
// main tier.
func needsFreshProcess(t *testing.T, why string) {
	t.Helper()
	if os.Getenv("GOINFER_ISOLATE_DEFER") != "" {
		// ISOLATED-SKIP is the token `gate gpu` counts; keep it exact and greppable.
		t.Skipf("ISOLATED-SKIP: %s — deferred to its own process by the gate's heavy tier", why)
	}
	t.Logf("ISOLATED-RUN: %s", why)
}
