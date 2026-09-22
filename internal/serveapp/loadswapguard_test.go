package serveapp

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// task-never-swap-2026-09.md S3: the LOAD-TIME half's integration seam — armLoadSwapGuard wired
// to a scripted reader (same convention as TestSwapGuard_tripsAndResumesThroughHaltGate, one
// level up from decoder/swapwatch_test.go's pure watch-logic tests), confirming the trip actually
// closes the abort channel a real decoder.Load would be watching via Options.LoadAbort, and that
// wrapErr turns the resulting decoder.ErrLoadAborted into the priced message the brief asks for.

func TestLoadSwapGuard_tripsClosesAbortAndWrapsError(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "500") // 500 MB threshold, small so the scripted deltas below exercise it
	const baseline = 1_000_000_000
	r := &scriptedSwapReader{seq: []int64{
		baseline,
		baseline + 100_000_000, // under threshold
		baseline + 600_000_000, // trip
		baseline + 600_000_000,
	}}

	abort, wrapErr, stop := armLoadSwapGuard("testdata/does-not-need-to-exist.gguf", decoder.Options{}, r.Read)
	defer stop()
	if abort == nil {
		t.Fatal("armed with GOINFER_SWAP_GUARD set, got a nil abort channel")
	}

	select {
	case <-abort:
		t.Fatal("abort closed before any tick landed")
	default:
	}

	r.waitTicks(t, 3) // baseline + one under-threshold + the tripping sample
	select {
	case <-abort:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for the abort channel to close after the trip")
	}

	// Simulates what a real caller sees back from decoder.Load/parallelLayers once the build
	// actually observes the closed channel (decoder/parallellayers_abort_test.go covers that part
	// directly; this test's job is the guard's own wiring, not re-proving parallelLayers).
	wrapped := wrapErr(decoder.ErrLoadAborted)
	if !errors.Is(wrapped, decoder.ErrLoadAborted) {
		t.Fatalf("wrapped error %v lost errors.Is(ErrLoadAborted)", wrapped)
	}
	msg := wrapped.Error()
	if !strings.Contains(msg, "swap grew") {
		t.Errorf("wrapped error %q does not name the swap-growth reason", msg)
	}
	if !strings.Contains(msg, "0.6") { // 600_000_000 / 1e9 GB delta
		t.Errorf("wrapped error %q does not quote the measured delta", msg)
	}
}

func TestLoadSwapGuard_off(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "off")
	calls := 0
	fakeRead := func() (int64, bool) { calls++; return 1 << 40, true } // absurdly high, would trip instantly if read
	abort, wrapErr, stop := armLoadSwapGuard("irrelevant.gguf", decoder.Options{}, fakeRead)
	defer stop()
	if abort != nil {
		t.Fatal("GOINFER_SWAP_GUARD=off must return a nil (no-op) abort channel")
	}
	time.Sleep(20 * time.Millisecond)
	if calls != 0 {
		t.Fatalf("GOINFER_SWAP_GUARD=off must never call the reader, got %d calls", calls)
	}
	someErr := errors.New("unrelated")
	if got := wrapErr(someErr); got != someErr {
		t.Fatalf("off: wrapErr must be a pure passthrough, got %v", got)
	}
}

func TestLoadSwapGuard_wrapErrPassthroughForNonAbortErrors(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "500")
	r := &scriptedSwapReader{seq: []int64{1_000_000_000}}
	_, wrapErr, stop := armLoadSwapGuard("irrelevant.gguf", decoder.Options{}, r.Read)
	defer stop()

	if wrapErr(nil) != nil {
		t.Fatal("wrapErr(nil) must return nil")
	}
	someErr := errors.New("a real build error, not an abort")
	if got := wrapErr(someErr); got != someErr {
		t.Fatalf("a non-abort error must pass through unchanged, got %v", got)
	}
}
