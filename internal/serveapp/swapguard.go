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

// S3 (docs/tasks/task-never-swap-2026-09.md): both halves of the swap tripwire.
//
// The SERVING half (armSwapGuard, armed once for the process's life): "the callback flips the
// admission gate to refuse new requests..., lets in-flight generations finish, logs once, and
// re-opens when swap-used returns to within threshold of the baseline for 30 s (hysteresis, so it
// does not flap)." Reuses haltGate's exact chokepoint (right before inf, on every generation
// route) rather than adding a tenth wrapper alongside the 9 existing srv.haltGate(...) call
// sites: swapGuardTripped is a second condition haltGate checks, next to K2's halted pointer —
// see halt.go. Unlike halt(), which cancels every in-flight generation (K1's cancelAll), tripping
// the swap guard does NOT cancel anything already running, only refuses new admissions — the
// brief's own "lets in-flight generations finish".
//
// The LOAD-TIME half (internal/swapguard's ArmLoad, shared with goinfer-chat; armed and torn down around exactly one decoder.Load
// call): "the callback cancels the load's context... the direct build needs a check between
// layers in parallelLayers" — that plumbing lives in decoder (Options.LoadAbort, checked by
// parallelLayers), this file only arms/tears down the watch around loadDecoder's decoder.Load
// call sites and turns a plain ErrLoadAborted into the priced message the brief asks for ("swap
// grew 0.9 GB during load — resident weights 12.6 GB + mapped source 12.1 GB on a 16 GB machine;
// use the sidecar path / a smaller quant"). Scoped to the GGUF direct-build resident path only —
// see Options.LoadAbort's own doc comment for why the .giw/streaming path is unaffected (the
// abort channel is a genuine no-op there today, not specially handled here).
//
// Both halves share swapguard.ThresholdBytes so GOINFER_SWAP_GUARD can't quote two different
// numbers for the same env var.

// startSwapGuard arms the swap watch, if enabled, and returns once the goroutine is running (it
// does not block on a first reading). Called once from newServer. GOINFER_SWAP_GUARD sets the
// threshold in MB (default 512); the literal value "off" disables the guard entirely (the watch
// is not started at all — no per-request branch cost, matching StartSwapWatch's own nil-Read
// no-op shape). An unparseable non-"off" value falls back to the default and says so, rather
// than silently guarding at 0 (which would trip on the first tick) or not guarding at all.
func (s *server) startSwapGuard() {
	// Under `go test`, do not arm a REAL background watch against the machine's actual
	// swap-used unless a test explicitly asked for one (GOINFER_SWAP_GUARD set). Roughly 17
	// files construct a real *server via newServer for reasons unrelated to this guard; each
	// would otherwise start an independent goroutine polling `sysctl -n vm.swapusage` every 2s
	// for the rest of that test binary's life, on a real, often-loaded development box — a
	// genuine flakiness risk (a real swap excursion mid-suite tripping some OTHER test's
	// handler through haltGate) for zero coverage benefit, since the mechanism itself is
	// already exercised directly by swapguard_test.go's own s.armSwapGuard(scripted) calls.
	// testing.Testing() (Go 1.21+) is the stdlib's own sanctioned way to ask this from
	// production code; it reports false in a real `go build` binary, so production serving is
	// unaffected. An explicit GOINFER_SWAP_GUARD (a number, or "off") always wins either way.
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

	// bannerOnce wraps read so the ONE startup banner naming the real baseline uses the watch's
	// own first successful reading — not a separate out-of-band call to read() before starting
	// the watch, which would (a) burn a real syscall the watch's own bookkeeping never sees and
	// (b) shift a scripted test reader's sequence by one entry against the watch's own index,
	// since StartSwapWatch has no way to know a reading already happened. One extra read has no
	// real-world effect (swap-used is sampled by time, not by call count), but it is exactly the
	// kind of thing that turns a deterministic test flaky for a reason invisible from the test
	// itself — caught here rather than shipped.
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
