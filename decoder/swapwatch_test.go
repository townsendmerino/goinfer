package decoder

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// scriptedReader plays back a fixed sequence of (usedBytes, ok) readings, one per call, and
// repeats the LAST entry forever once exhausted (so a test does not need to size the sequence to
// the exact number of ticks the watch happens to take). ticks records how many times Read was
// actually called, for tests that need to know the watch has observed at least N samples.
type scriptedReader struct {
	mu       sync.Mutex
	seq      []int64 // -1 marks an "unavailable" (ok=false) tick
	i        int
	ticks    atomic.Int64
	tickedCh chan struct{} // closed-then-replaced signal: send after recording each tick
}

func newScriptedReader(seq []int64) *scriptedReader {
	return &scriptedReader{seq: seq, tickedCh: make(chan struct{}, 1024)}
}

func (s *scriptedReader) Read() (int64, bool) {
	s.mu.Lock()
	v := s.seq[min(s.i, len(s.seq)-1)]
	if s.i < len(s.seq)-1 {
		s.i++
	}
	s.mu.Unlock()
	s.ticks.Add(1)
	select {
	case s.tickedCh <- struct{}{}:
	default:
	}
	if v < 0 {
		return 0, false
	}
	return v, true
}

// waitTicks blocks until at least n reads have happened, or fails the test after a generous
// timeout — a fast local poll interval keeps this quick without sleeping a fixed duration.
func (s *scriptedReader) waitTicks(t *testing.T, n int) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		if s.ticks.Load() >= int64(n) {
			return
		}
		select {
		case <-s.tickedCh:
		case <-deadline:
			t.Fatalf("timed out waiting for %d ticks (got %d)", n, s.ticks.Load())
		}
	}
}

// TestSwapWatch_baseline confirms the FIRST reading becomes the baseline regardless of its
// absolute value — including a machine already deep in swap from something else (S3's own
// requirement: the guard measures what THIS process adds, not the machine's pre-existing state).
func TestSwapWatch_baseline(t *testing.T) {
	const alreadyDeepInSwap = 14_700_000_000 // the cold-user run's own stale 14.7 GB baseline
	r := newScriptedReader([]int64{alreadyDeepInSwap, alreadyDeepInSwap, alreadyDeepInSwap + 1000})
	var tripped atomic.Bool
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := StartSwapWatch(ctx, SwapWatchOptions{
		Read:         r.Read,
		PollInterval: time.Millisecond,
		Threshold:    512 * 1024 * 1024,
		OnTrip:       func(int64, int64) { tripped.Store(true) },
	})
	defer w.Stop()
	r.waitTicks(t, 5)
	if tripped.Load() {
		t.Fatalf("tripped on a machine that never grew swap past baseline + threshold, just started deep in swap")
	}
}

// TestSwapWatch_ramp confirms OnTrip fires exactly once when swap-used grows past baseline +
// threshold, with the delta reported correctly, and does not fire again on later ticks that stay
// above the threshold (ResumeAfter=0 here, matching the load-time consumer's shape).
func TestSwapWatch_ramp(t *testing.T) {
	const baseline = 2_000_000_000
	const threshold = 500_000_000
	r := newScriptedReader([]int64{
		baseline,
		baseline + 100_000_000, // under threshold: no trip
		baseline + 400_000_000, // still under: no trip
		baseline + 600_000_000, // over: trip
		baseline + 900_000_000, // still over: no SECOND OnTrip
		baseline + 950_000_000,
	})
	var trips atomic.Int64
	var lastUsed, lastDelta atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := StartSwapWatch(ctx, SwapWatchOptions{
		Read:         r.Read,
		PollInterval: time.Millisecond,
		Threshold:    threshold,
		OnTrip: func(used, delta int64) {
			trips.Add(1)
			lastUsed.Store(used)
			lastDelta.Store(delta)
		},
	})
	defer w.Stop()
	r.waitTicks(t, 6)
	if got := trips.Load(); got != 1 {
		t.Fatalf("OnTrip fired %d times, want exactly 1", got)
	}
	if got, want := lastUsed.Load(), int64(baseline+600_000_000); got != want {
		t.Errorf("OnTrip usedBytes = %d, want %d", got, want)
	}
	if got, want := lastDelta.Load(), int64(600_000_000); got != want {
		t.Errorf("OnTrip deltaBytes = %d, want %d", got, want)
	}
}

// TestSwapWatch_hysteresis confirms OnResume fires only after swap-used has stayed within
// threshold of baseline for the FULL ResumeAfter window, continuously — a dip back under
// threshold that does not hold resets the clock rather than firing early (the "does not flap"
// requirement).
func TestSwapWatch_hysteresis(t *testing.T) {
	const baseline = 1_000_000_000
	const threshold = 200_000_000
	r := newScriptedReader([]int64{
		baseline,
		baseline + 300_000_000, // trip
		baseline + 50_000_000,  // back within threshold: hysteresis clock starts
		baseline + 300_000_000, // grew again: clock RESETS, must not resume yet
		baseline + 50_000_000,  // within again: clock restarts
		baseline + 60_000_000,  // still within: clock keeps running
		baseline + 60_000_000,
	})
	var trips, resumes atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := StartSwapWatch(ctx, SwapWatchOptions{
		Read:         r.Read,
		PollInterval: time.Millisecond,
		Threshold:    threshold,
		ResumeAfter:  30 * time.Millisecond, // several poll intervals, so the reset is exercised
		OnTrip:       func(int64, int64) { trips.Add(1) },
		OnResume:     func() { resumes.Add(1) },
	})
	defer w.Stop()
	r.waitTicks(t, 7)
	if got := trips.Load(); got != 1 {
		t.Fatalf("OnTrip fired %d times, want 1", got)
	}
	if resumes.Load() != 0 {
		t.Fatalf("OnResume fired before the reset-then-hold sequence had run its course (resumes=%d)", resumes.Load())
	}
	// Now let it hold within threshold for real and confirm OnResume DOES eventually fire.
	time.Sleep(60 * time.Millisecond)
	if resumes.Load() != 1 {
		t.Fatalf("OnResume did not fire after holding within threshold for longer than ResumeAfter (resumes=%d)", resumes.Load())
	}
}

// TestSwapWatch_unavailableReadingsAreSkipped confirms a reader that returns ok=false is treated
// as "try again next tick", never as a trip and never as advancing the baseline — including when
// unavailability is the FIRST few ticks (baseline capture must wait for a real reading).
func TestSwapWatch_unavailableReadingsAreSkipped(t *testing.T) {
	r := newScriptedReader([]int64{-1, -1, 1_000_000_000, -1, 1_000_000_000 + 100})
	var tripped atomic.Bool
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := StartSwapWatch(ctx, SwapWatchOptions{
		Read:         r.Read,
		PollInterval: time.Millisecond,
		Threshold:    512 * 1024 * 1024,
		OnTrip:       func(int64, int64) { tripped.Store(true) },
	})
	defer w.Stop()
	r.waitTicks(t, 5)
	if tripped.Load() {
		t.Fatal("an unavailable reading must never be treated as a trip or corrupt the baseline")
	}
}

// TestSwapWatch_offSwitch confirms a nil Read (the GOINFER_SWAP_GUARD=off shape) starts and stops
// cleanly and never calls OnTrip, so a caller does not need its own enabled/disabled branch
// around every StartSwapWatch call site.
func TestSwapWatch_offSwitch(t *testing.T) {
	var tripped atomic.Bool
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := StartSwapWatch(ctx, SwapWatchOptions{
		Read:   nil,
		OnTrip: func(int64, int64) { tripped.Store(true) },
	})
	w.Stop() // must return promptly, not hang waiting for a goroutine that was never started
	if tripped.Load() {
		t.Fatal("a nil Read must never trip")
	}
}

// TestSwapWatch_ctxCancelStopsTheGoroutine confirms cancelling the passed ctx stops sampling
// (equivalent to Stop), so a caller that ties the watch's lifetime to a request/load ctx does not
// leak the goroutine.
func TestSwapWatch_ctxCancelStopsTheGoroutine(t *testing.T) {
	r := newScriptedReader([]int64{1, 2, 3})
	ctx, cancel := context.WithCancel(context.Background())
	w := StartSwapWatch(ctx, SwapWatchOptions{Read: r.Read, PollInterval: time.Millisecond})
	r.waitTicks(t, 2)
	cancel()
	done := make(chan struct{})
	go func() { w.Stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Stop() did not return after the passed ctx was cancelled")
	}
}

// TestSwapWatch_keyingOnRSSWouldHaveMissedR11c is the inverting-guard mutation check
// docs/tasks/task-never-swap-2026-09.md S3 asks for, encoded as a test so the mistake it guards
// against cannot come back silently. CLAUDE.md: "A guard that INVERTS under the condition it
// exists for is worse than no guard — it actively reassures." The real R11(c) run
// (docs/measurements/metal-moe-autopager-m26-2026-09-20.md) measured, in the SAME build, both
// figures below: swap-used genuinely grew from 2.3 to 12 GB and NEVER came back down, while the
// test's own RSS reading went "7 MB -> 892 MB -> falling" — because darwin reclaims MTLBuffer
// pages under pressure as fast as they are written, so RSS reports what survived, not what was
// asked for (S0's own reading of this run). Run BOTH trajectories through the exact same watch:
// a swap-keyed watch never resumes (correct — the machine never stopped losing memory); an
// RSS-keyed watch resumes mid-spiral, because its own falling tail reads as recovery. That
// resume-while-still-losing is the inversion, shown directly rather than argued.
func TestSwapWatch_keyingOnRSSWouldHaveMissedR11c(t *testing.T) {
	const threshold = 512 * 1024 * 1024
	const resumeAfter = 2 * time.Millisecond

	// The real R11(c) N=64 run's own reported RSS trajectory, in bytes: 7 MB baseline -> 892 MB
	// at peak -> falling as the OS reclaims (the "RSS 7 MB -> 892 MB after build" the run itself
	// logged, extended with the fall the mechanism implies once the page-out settles).
	rssSeq := []int64{7_000_000, 620_000_000, 892_000_000, 340_000_000, 210_000_000, 180_000_000}
	// The SAME run's real swap-used trajectory: 2.3 -> 12 GB in ~16s during build, and staying
	// there — the machine never actually recovered.
	swapSeq := []int64{2_300_000_000, 4_800_000_000, 7_600_000_000, 9_900_000_000, 12_000_000_000, 12_000_000_000}

	resumedAfter := func(seq []int64) bool {
		r := newScriptedReader(seq)
		var resumed atomic.Bool
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		w := StartSwapWatch(ctx, SwapWatchOptions{
			Read:         r.Read,
			PollInterval: time.Millisecond,
			Threshold:    threshold,
			ResumeAfter:  resumeAfter,
			OnResume:     func() { resumed.Store(true) },
		})
		defer w.Stop()
		r.waitTicks(t, len(seq))
		time.Sleep(20 * time.Millisecond) // let the hysteresis window elapse against the held tail
		return resumed.Load()
	}

	if resumedAfter(swapSeq) {
		t.Fatal("the swap-keyed watch resumed on a trajectory that never returned within threshold of baseline — the machine never actually recovered")
	}
	if !resumedAfter(rssSeq) {
		t.Fatal("expected an RSS-keyed watch to (wrongly) resume while its own falling tail holds flat — this is the inversion the guard must never reproduce for real: RSS reporting 'recovered' during a run where swap (the true signal) never stopped climbing")
	}
}
