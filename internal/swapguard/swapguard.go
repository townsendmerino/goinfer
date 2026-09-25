// Package swapguard is S3's load-time swap tripwire (docs/tasks/task-never-swap-2026-09.md), shared by every
// entry point that loads a model: goinfer-serve (internal/serveapp, which also runs the SERVING half — the
// admission gate — on top of ThresholdBytes) and goinfer-chat (internal/chatapp, which has no admission gate,
// so it takes only this half: "chat gets the load-time half only", S3 Build item 4).
//
// The load-time half arms a watch around exactly one decoder.Load of a .gguf direct build; a trip closes the
// channel the build observes through Options.LoadAbort (checked between layers by parallelLayers) and
// WrapErr turns the resulting decoder.ErrLoadAborted into a message that names the swap growth and the fit
// guard's own priced terms (decoder.FitDescribe), plus caller-specific advice — the flag names differ per
// command, so the caller supplies them.
package swapguard

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

// ThresholdBytes parses GOINFER_SWAP_GUARD once, shared by both halves of the guard so
// they cannot drift apart on what the same env var means. ok is false for "off" (the byte/MB
// values are meaningless then and must not be used). An unparseable non-"off" value falls back to
// the 512 MB default and says so, rather than silently guarding at 0 (which would trip on the
// first tick) or not guarding at all.
func ThresholdBytes() (thresholdBytes, thresholdMB int64, ok bool) {
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

// StartLoad is ArmLoad with the same `go test` suppression serveapp's startSwapGuard applies to its
// serving half: the real background watch is not started under `go test` unless a
// test explicitly opts in via GOINFER_SWAP_GUARD (model loads are exercised by many tests that
// have nothing to do with this guard). Call sites use this one; tests call ArmLoad directly with a
// scripted reader, bypassing the suppression as an explicit opt-in.
func StartLoad(loadPath string, opts decoder.Options, advice string) (abort <-chan struct{}, wrapErr func(error) error, stop func()) {
	if testing.Testing() && os.Getenv("GOINFER_SWAP_GUARD") == "" {
		return nil, func(err error) error { return err }, func() {}
	}
	return ArmLoad(loadPath, opts, advice, decoder.SwapUsedBytes)
}

// ArmLoad arms a watch scoped to exactly one decoder.Load call — S3's load-time half.
// Unlike serveapp's armSwapGuard (armed once for the process's life, after startup, so its baseline is
// "steady state"), a fresh baseline every load is correct here: the question is "did loading THIS
// model grow swap," not "is the machine's swap elevated in general." ResumeAfter is 0 (no
// OnResume) because a load either finishes or is aborted — there is no "resume mid-load"
// (SwapWatchOptions.ResumeAfter's own doc comment anticipates exactly this consumer).
//
// read is injectable so a test can drive this against a scripted reader instead of the real OS
// probe (decoder/swapwatch_test.go's own scriptedReader tests the watch's decision logic in
// isolation already; this is the integration seam one level up, same split as serveapp's
// armSwapGuard/startSwapGuard). Returns a nil abort channel — the documented Options.LoadAbort no-op — when
// the guard is off (GOINFER_SWAP_GUARD=off). The returned stop func must be called once the load
// call it guards has returned, by any outcome — it also drains the watch goroutine (SwapWatch.Stop
// blocks until it has exited), so there is never a watch left running past the load it was armed
// for.
//
// advice is what the abort message tells the user to do instead, in the calling command's own flag
// names (serve: "use -stream-weights or a smaller quant").
func ArmLoad(loadPath string, opts decoder.Options, advice string, read decoder.SwapReader) (abort <-chan struct{}, wrapErr func(error) error, stop func()) {
	noop := func(err error) error { return err }
	threshold, thresholdMB, on := ThresholdBytes()
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
			return fmt.Errorf("load aborted: %s — %s; %s: %w", reason, desc, advice, err)
		}
		return fmt.Errorf("load aborted: %s; %s: %w", reason, advice, err)
	}

	return abortCh, wrap, watch.Stop
}
