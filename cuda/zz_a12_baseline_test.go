//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestZZ_A12ContextBaseline reports free VRAM at the END of a run, after every preceding test has
// finished and closed. Named zz_ so it sorts last: Go runs tests in declaration order within a
// package, files in sorted order, so this is the only way to observe the state a whole tier leaves
// behind from inside the same process.
//
// A12's surviving mechanism is PER-CONTEXT KERNEL RESERVATIONS: a kernel's first launch reserves
// local-memory backing store sized for occupancy, it is a context property rather than a model one,
// and Close() tears down a model without destroying the context. So a tier that first-launches many
// kernels would end holding memory no model-level Close can return.
//
// THE BOUND IS PRE-REGISTERED, because a confirmed-but-small effect must not be read as the cause.
// A9's census measured the driver sharing ONE backing store sized by the largest kernel — the whole
// census returned a residual identical to moe_route alone, 138,412,032 B — and sequential
// single-stream launch is the precondition, which this tier satisfies (-p 1, one package, no
// t.Parallel).
//
//	step-down <= ~132 MiB     -> mechanism CONFIRMED and INSUFFICIENT. Real, but not the blocker.
//	step-down materially more -> REFUTED; the local-memory account does not explain it.
//	no step-down at all       -> REFUTED outright.
//
// Compare its number against a fresh process: GOINFER_A12_BASELINE=1 with -run on this test alone
// gives the empty-process reading; the same variable across the full tier gives the after reading.
func TestZZ_A12ContextBaseline(t *testing.T) {
	if os.Getenv("GOINFER_A12_BASELINE") == "" {
		t.Skip("set GOINFER_A12_BASELINE=1 — this is A12's end-of-run probe, not a gate")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	free, total, e := ctx.MemInfo()
	if e != nil {
		t.Fatalf("MemInfo: %v", e)
	}
	// Stderr as well as t.Logf: this number is meant to be grepped out of a full-tier run, where
	// t.Logf output is buried under thousands of lines.
	fmt.Fprintf(os.Stderr, "A12BASELINE free=%d total=%d free_MiB=%.1f\n", free, total, float64(free)/(1<<20))
	t.Logf("free at end of run: %.1f MiB of %.1f MiB", float64(free)/(1<<20), float64(total)/(1<<20))
}
