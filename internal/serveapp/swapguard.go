package serveapp

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/swapguard"
)

// Both halves of the swap tripwire (docs/tasks/task-never-swap-2026-09.md).
//
// The SERVING half (armSwapGuard, armed once for the process's life): the callback flips the admission gate to refuse new
// requests, lets in-flight generations finish, logs once, and re-opens when swap-used is back within threshold of the baseline
// for 30 s (hysteresis, so it does not flap). It reuses haltGate's chokepoint (right before inf, on every generation route)
// rather than adding another wrapper: swapGuardTripped is a second condition haltGate checks next to the halted pointer
// (halt.go). Unlike halt(), tripping the guard does NOT cancel anything already running; it only refuses new admissions.
//
// The LOAD-TIME half (internal/swapguard's ArmLoad, shared with goinfer-chat; armed and torn down around exactly one
// decoder.Load call): the callback cancels the load's context. The plumbing that stops the load lives in decoder
// (Options.LoadAbort, checked by parallelLayers); this file arms and tears down the watch around loadDecoder's decoder.Load
// call sites and turns a plain ErrLoadAborted into a priced message ("swap grew 0.9 GB during load: resident weights 12.6 GB +
// mapped source 12.1 GB on a 16 GB machine; use the sidecar path / a smaller quant"). Scoped to the GGUF direct-build resident
// path only; see Options.LoadAbort for why the .giw/streaming path is unaffected.
//
// Both halves share swapguard.ThresholdBytes so GOINFER_SWAP_GUARD cannot quote two different numbers for the same env var.

// startSwapGuard arms the swap watch, if enabled, and returns once the goroutine is running (it
// does not block on a first reading). Called once from newServer. GOINFER_SWAP_GUARD sets the
// threshold in MB (default 512); the literal value "off" disables the guard entirely (the watch
// is not started at all — no per-request branch cost, matching StartSwapWatch's own nil-Read
// no-op shape). An unparseable non-"off" value falls back to the default and says so, rather
// than silently guarding at 0 (which would trip on the first tick) or not guarding at all.
func (s *server) startSwapGuard() {
	// Under `go test`, do not arm a REAL background watch against the machine's actual swap-used unless a test explicitly
	// asked for one (GOINFER_SWAP_GUARD set): many tests construct a real *server via newServer for unrelated reasons, and
	// each would start its own goroutine polling `sysctl -n vm.swapusage` every 2s for the rest of the test binary's life,
	// risking a real swap excursion tripping some OTHER test's handler through haltGate for no coverage gain
	// (swapguard_test.go exercises the mechanism through armSwapGuard with a scripted reader). testing.Testing() reports false
	// in a real binary, so production serving is unaffected. An explicit GOINFER_SWAP_GUARD (a number, or "off") always wins.
	if testing.Testing() && os.Getenv("GOINFER_SWAP_GUARD") == "" {
		return
	}
	s.armSwapGuard(decoder.SwapUsedBytes)
}

// armSwapGuard is startSwapGuard with the swap-used probe injectable, so a test can wire the
// whole trip -> haltGate -> resume path against a scripted reader without touching the real OS
// probe (decoder/swapwatch_test.go's own scriptedReader tests the watch's decision logic in
// isolation already; this is the integration seam one level up).
func (s *server) armSwapGuard(read decoder.SwapReader) {
	threshold, thresholdMB, on := swapguard.ThresholdBytes()
	if !on {
		fmt.Fprintln(os.Stderr, "swap guard: disabled (GOINFER_SWAP_GUARD=off)")
		return
	}
	fmt.Fprintf(os.Stderr, "swap guard: armed, threshold +%d MB over baseline\n", thresholdMB)

	// bannerOnce wraps read so the ONE startup banner naming the real baseline uses the watch's own first successful reading,
	// not a separate call to read() before starting the watch: that would burn a syscall the watch's bookkeeping never sees
	// and shift a scripted test reader's sequence by one entry against the watch's own index.
	bannerPrinted := false
	bannerOnce := func() (int64, bool) {
		used, ok := read()
		if !bannerPrinted {
			bannerPrinted = true
			if ok {
				fmt.Fprintf(os.Stderr, "swap guard: baseline %.2f GB swap-used\n", float64(used)/1e9)
			} else {
				fmt.Fprintln(os.Stderr, "swap guard: this platform's swap-used probe reports unknown on the first sample — the guard will never trip until it reports a real reading")
			}
		}
		return used, ok
	}

	s.swapWatch = decoder.StartSwapWatch(context.Background(), decoder.SwapWatchOptions{
		Read:         bannerOnce,
		PollInterval: 2 * time.Second,
		Threshold:    threshold,
		ResumeAfter:  30 * time.Second,
		OnTrip: func(used, delta int64) {
			s.swapGuardTripped.Store(true)
			fmt.Fprintf(os.Stderr, "swap guard: TRIPPED — swap grew %.2f GB over baseline (%.2f GB used now); refusing new requests until it recovers\n", float64(delta)/1e9, float64(used)/1e9)
		},
		OnResume: func() {
			s.swapGuardTripped.Store(false)
			fmt.Fprintln(os.Stderr, "swap guard: recovered — swap-used back within threshold of baseline for 30s; admitting new requests again")
		},
	})
}
