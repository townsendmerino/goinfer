package serveapp

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
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
// The LOAD-TIME half (armLoadSwapGuard, armed and torn down around exactly one decoder.Load
// call): "the callback cancels the load's context... the direct build needs a check between
// layers in parallelLayers" — that plumbing lives in decoder (Options.LoadAbort, checked by
// parallelLayers), this file only arms/tears down the watch around loadDecoder's decoder.Load
// call sites and turns a plain ErrLoadAborted into the priced message the brief asks for ("swap
// grew 0.9 GB during load — resident weights 12.6 GB + mapped source 12.1 GB on a 16 GB machine;
// use the sidecar path / a smaller quant"). Scoped to the GGUF direct-build resident path only —
// see Options.LoadAbort's own doc comment for why the .giw/streaming path is unaffected (the
// abort channel is a genuine no-op there today, not specially handled here).
//
// Both halves share swapGuardThresholdBytes so GOINFER_SWAP_GUARD can't quote two different
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
	threshold, thresholdMB, on := swapGuardThresholdBytes()
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

// swapGuardThresholdBytes parses GOINFER_SWAP_GUARD once, shared by both halves of the guard so
// they cannot drift apart on what the same env var means. ok is false for "off" (the byte/MB
// values are meaningless then and must not be used). An unparseable non-"off" value falls back to
// the 512 MB default and says so, rather than silently guarding at 0 (which would trip on the
// first tick) or not guarding at all.
func swapGuardThresholdBytes() (thresholdBytes, thresholdMB int64, ok bool) {
	const defaultMB = 512
	mb := int64(defaultMB)
	switch v := os.Getenv("GOINFER_SWAP_GUARD"); v {
	case "":
		// default
	case "off":
		return 0, 0, false
	default:
		if parsed, err := strconv.ParseInt(v, 10, 64); err == nil && parsed > 0 {
			mb = parsed
		} else {
			fmt.Fprintf(os.Stderr, "swap guard: GOINFER_SWAP_GUARD=%q is neither \"off\" nor a positive integer (MB) — using the %d MB default\n", v, defaultMB)
		}
	}
	return mb * 1024 * 1024, mb, true
}

// startLoadSwapGuard is armLoadSwapGuard with the same `go test` suppression startSwapGuard
// applies to armSwapGuard: the real background watch is not started under `go test` unless a
// test explicitly opts in via GOINFER_SWAP_GUARD (loadDecoder is exercised by many tests that
// have nothing to do with this guard). Call sites use this one; tests call armLoadSwapGuard
// directly with a scripted reader, bypassing the suppression as an explicit opt-in.
func startLoadSwapGuard(loadPath string, opts decoder.Options) (abort <-chan struct{}, wrapErr func(error) error, stop func()) {
	if testing.Testing() && os.Getenv("GOINFER_SWAP_GUARD") == "" {
		return nil, func(err error) error { return err }, func() {}
	}
	return armLoadSwapGuard(loadPath, opts, decoder.SwapUsedBytes)
}

// armLoadSwapGuard arms a watch scoped to exactly one decoder.Load call — S3's load-time half.
// Unlike armSwapGuard (armed once for the process's life, after startup, so its baseline is
// "steady state"), a fresh baseline every load is correct here: the question is "did loading THIS
// model grow swap," not "is the machine's swap elevated in general." ResumeAfter is 0 (no
// OnResume) because a load either finishes or is aborted — there is no "resume mid-load"
// (SwapWatchOptions.ResumeAfter's own doc comment anticipates exactly this consumer).
//
// read is injectable so a test can drive this against a scripted reader instead of the real OS
// probe (decoder/swapwatch_test.go's own scriptedReader tests the watch's decision logic in
// isolation already; this is the integration seam one level up, same split as armSwapGuard/
// startSwapGuard). Returns a nil abort channel — the documented Options.LoadAbort no-op — when
// the guard is off (GOINFER_SWAP_GUARD=off). The returned stop func must be called once the load
// call it guards has returned, by any outcome — it also drains the watch goroutine (SwapWatch.Stop
// blocks until it has exited), so there is never a watch left running past the load it was armed
// for.
func armLoadSwapGuard(loadPath string, opts decoder.Options, read decoder.SwapReader) (abort <-chan struct{}, wrapErr func(error) error, stop func()) {
	noop := func(err error) error { return err }
	threshold, thresholdMB, on := swapGuardThresholdBytes()
	if !on {
		return nil, noop, func() {}
	}

	abortCh := make(chan struct{})
	var mu sync.Mutex
	var tripped bool
	var trippedUsed, trippedDelta int64

	fmt.Fprintf(os.Stderr, "swap guard (load): armed for %s, threshold +%d MB over baseline\n", loadPath, thresholdMB)
	bannerPrinted := false
	bannerOnce := func() (int64, bool) {
		used, ok := read()
		if !bannerPrinted {
			bannerPrinted = true
			if ok {
				fmt.Fprintf(os.Stderr, "swap guard (load): baseline %.2f GB swap-used\n", float64(used)/1e9)
			}
		}
		return used, ok
	}

	watch := decoder.StartSwapWatch(context.Background(), decoder.SwapWatchOptions{
		Read:         bannerOnce,
		PollInterval: 2 * time.Second,
		Threshold:    threshold,
		OnTrip: func(used, delta int64) {
			mu.Lock()
			first := !tripped
			if first {
				tripped, trippedUsed, trippedDelta = true, used, delta
			}
			mu.Unlock()
			if first {
				close(abortCh) // the actual signal parallelLayers' abort select observes
			}
			fmt.Fprintf(os.Stderr, "swap guard (load): TRIPPED — swap grew %.2f GB over baseline (%.2f GB used now); aborting the load of %s\n", float64(delta)/1e9, float64(used)/1e9, loadPath)
		},
	})

	wrap := func(err error) error {
		if err == nil || !errors.Is(err, decoder.ErrLoadAborted) {
			return err
		}
		mu.Lock()
		used, delta := trippedUsed, trippedDelta
		mu.Unlock()
		reason := fmt.Sprintf("swap grew %.2f GB over baseline during load (%.2f GB swap-used now)", float64(delta)/1e9, float64(used)/1e9)
		if desc, derr := decoder.FitDescribe(loadPath, opts); derr == nil {
			return fmt.Errorf("load aborted: %s — %s; use -stream-weights or a smaller quant: %w", reason, desc, err)
		}
		return fmt.Errorf("load aborted: %s: %w", reason, err)
	}

	return abortCh, wrap, watch.Stop
}
