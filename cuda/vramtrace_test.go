//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// TestMain installs an OPT-IN VRAM sampler for A12: does free VRAM decline monotonically across the
// heavy tier (accumulation — something is not freeing), or does it recover after each test with a
// high-water mark above the card (a genuine environment limit)?
//
// Go's testing package exposes no per-test hook, and the four tests that fail in-suite do not share
// a helper — requireHeavyModel covers 14 call sites and none of them. So the boundaries are taken
// from `-v` output and joined to a timestamped sample stream by wall clock.
//
// IT USES cuMemGetInfo, deliberately — the same instrument the whole A-chain used. nvidia-smi would
// have been easier to wire from outside the process and would have been a DIFFERENT instrument: the
// two disagreed by 852,224 B when A10 mixed them (107,806,720 vs 106,954,752), and the reporting-gap
// decomposition only closed with cuMemGetInfo on both sides.
//
// WHAT IT COSTS, stated because it perturbs what it measures: the sampler holds its own context, and
// a context reserves ~106,954,752 B (A10's per-context term). Every reading is therefore offset by
// roughly that much, and the run has that much less to work with than an untraced one. The SHAPE —
// monotonic versus sawtooth — is what this is for, and the shape is unaffected by a constant offset.
// Absolute figures from a traced run are not comparable with an untraced one.
//
// Off unless GOINFER_VRAM_TRACE=1, so no ordinary run pays the context or the polling.
func TestMain(m *testing.M) {
	stop := func() {}
	if os.Getenv("GOINFER_VRAM_TRACE") != "" {
		stop = startVRAMTrace()
	}
	code := m.Run()
	stop()
	os.Exit(code)
}

// startVRAMTrace polls free VRAM and prints one line per sample. Returns a stop function.
//
// Lines are machine-readable and go to STDERR. Not a style choice: `go test` captures the binary's
// stdout and replays it attributed to tests, and anything written outside a test's context — which
// is exactly where a TestMain-owned sampler goroutine lives — is silently dropped. Verified: a
// stderr probe from TestMain appeared, an identical stdout probe did not. The join to test
// boundaries is by wall clock, and both streams are captured together with 2>&1.
func startVRAMTrace() func() {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		fmt.Fprintf(os.Stderr, "VRAMTRACE unavailable: %v\n", err)
		return func() {}
	}
	done := make(chan struct{})
	stopped := make(chan struct{})
	sample := func(now time.Time) {
		free, total, e := dev.Context().MemInfo()
		if e != nil {
			fmt.Fprintf(os.Stderr, "VRAMTRACE %d err=%v\n", now.UnixNano(), e)
			return
		}
		fmt.Fprintf(os.Stderr, "VRAMTRACE %d free=%d total=%d\n", now.UnixNano(), free, total)
	}
	go func() {
		defer close(stopped)
		// 50 ms, and a sample IMMEDIATELY. A 200 ms ticker produced zero samples on a test that
		// completes in 0.04 s — and it completes that fast precisely BECAUSE the tracer's context
		// holds VRAM, so a drain-to-exhaustion test has less to exhaust. The instrument shortened
		// the very interval it was trying to sample.
		sample(time.Now())
		t := time.NewTicker(50 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case now := <-t.C:
				sample(now)
			}
		}
	}()
	return func() { close(done); <-stopped }
}
