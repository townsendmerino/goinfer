package swapguard

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// task-never-swap-2026-09.md S3: the LOAD-TIME half's integration seam — ArmLoad wired
// to a scripted reader (same convention as serveapp's TestSwapGuard_tripsAndResumesThroughHaltGate, one
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

	abort, wrapErr, stop := ArmLoad("testdata/does-not-need-to-exist.gguf", decoder.Options{}, "use -stream-weights or a smaller quant", r.Read)
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
	if !strings.Contains(msg, "use -stream-weights or a smaller quant") {
		t.Errorf("wrapped error %q does not carry the caller's advice — each command names its own flags", msg)
	}
}

func TestLoadSwapGuard_off(t *testing.T) {
	t.Setenv("GOINFER_SWAP_GUARD", "off")
	calls := 0
	fakeRead := func() (int64, bool) { calls++; return 1 << 40, true } // absurdly high, would trip instantly if read
	abort, wrapErr, stop := ArmLoad("irrelevant.gguf", decoder.Options{}, "advice", fakeRead)
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
	_, wrapErr, stop := ArmLoad("irrelevant.gguf", decoder.Options{}, "advice", r.Read)
	defer stop()

	if wrapErr(nil) != nil {
		t.Fatal("wrapErr(nil) must return nil")
	}
	someErr := errors.New("a real build error, not an abort")
	if got := wrapErr(someErr); got != someErr {
		t.Fatalf("a non-abort error must pass through unchanged, got %v", got)
	}
}

// scriptedSwapReader plays back a fixed sequence and repeats the last entry, counting ticks so a test can
// wait for N samples without a fixed sleep (the same helper serveapp's swapguard_test.go keeps for its
// serving half).
type scriptedSwapReader struct {
	seq   []int64
	i     atomic.Int64
	ticks atomic.Int64
}

func (s *scriptedSwapReader) Read() (int64, bool) {
	i := s.i.Load()
	if i < int64(len(s.seq)-1) {
		s.i.Add(1)
	}
	s.ticks.Add(1)
	v := s.seq[min(i, int64(len(s.seq)-1))]
	if v < 0 {
		return 0, false
	}
	return v, true
}

func (s *scriptedSwapReader) waitTicks(t *testing.T, n int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for s.ticks.Load() < int64(n) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %d ticks (got %d)", n, s.ticks.Load())
		}
		time.Sleep(time.Millisecond)
	}
}
