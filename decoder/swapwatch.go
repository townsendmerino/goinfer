package decoder

import (
	"context"
	"time"
)

// S3 (docs/tasks/task-never-swap-2026-09.md): the swap tripwire. goinfer notices swap growing
// during its own run and acts — instead of leaving "kill at swap-used baseline + 500 MB" to a
// human watching `free -m`/`sysctl vm.swapusage` in another terminal, which is exactly what the
// cold-user harness and the R11(c) runs have had to do by hand. The external shell script that
// polled `vm.swapusage` every 1s and SIGKILLed on two consecutive >80 MB ticks (or one >300 MB
// jump) cut the peak excursion 5-7x against manual monitoring
// (docs/measurements/metal-moe-autopager-m26-2026-09-20.md's third attempt) — this moves that
// same idea inside the process and gives it a way to act short of SIGKILL.
//
// Keys on SWAP-USED, never RSS: darwin RSS reports what survived reclaim under memory pressure
// (CLAUDE.md's "a guard that INVERTS under the condition it exists for" — measured on this
// exact R11(c) run: RSS read "7 MB -> 892 MB after build" while swap grew from 2.3 to 12 GB
// during that same build, because the MTLBuffer pages were being compressed and evicted as fast
// as they were written). Swap-used only grows when the machine is actually losing memory, which
// is the one figure this guard can trust.

// SwapReader samples the machine's current swap-used bytes. ok is false when the figure could
// not be determined (no swap concept on this platform, or a probe that failed this tick) — the
// watcher treats "unknown" as "do nothing and try again next tick", never as a guess in either
// direction. SwapUsedBytes (memwatch_{darwin,linux,other}.go) is the real probe; tests inject a
// scripted fake.
type SwapReader func() (usedBytes int64, ok bool)

// SwapWatchOptions configures StartSwapWatch. Read is required; every other field defaults.
type SwapWatchOptions struct {
	Read SwapReader

	// PollInterval between samples. Default 2s (S3's registered interval).
	PollInterval time.Duration

	// Threshold is how far swap-used may grow above the FIRST successful sample's baseline
	// before OnTrip fires. Default 512 MiB (S3's registered default). A machine already deep in
	// swap from something else unrelated has a baseline that includes it — this measures only
	// what grows AFTER the watch starts, i.e. what the watched process itself adds, not the
	// machine's pre-existing state.
	Threshold int64

	// ResumeAfter: once tripped, OnResume (if set) fires after swap-used has stayed within
	// Threshold of the baseline for this long CONTINUOUSLY — the hysteresis the serving consumer
	// needs so a swap figure hovering near the line does not flap the admission gate open and
	// shut. 0 disables OnResume entirely (the load-time consumer wants this: a load either
	// finishes or is aborted, there is no "resume mid-load").
	ResumeAfter time.Duration

	// OnTrip is called exactly once per trip (not on every tick while still tripped), with the
	// current swap-used reading and the delta over baseline that caused it.
	OnTrip func(usedBytes, deltaBytes int64)

	// OnResume is called exactly once when ResumeAfter's hysteresis clears following a trip.
	// Never called when ResumeAfter <= 0 or OnResume is nil.
	OnResume func()
}

// SwapWatch is a running background sampler. The zero value is not usable — obtain one from
// StartSwapWatch.
type SwapWatch struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// Stop ends the watch and blocks until its goroutine has actually exited (so a caller that also
// wants to read state the callbacks wrote does not race the last tick). Safe on a nil receiver
// and safe to call more than once.
func (w *SwapWatch) Stop() {
	if w == nil {
		return
	}
	w.cancel()
	<-w.done
}

// StartSwapWatch starts sampling in a background goroutine and returns immediately; ctx ending is
// equivalent to calling Stop. A nil opts.Read makes this a no-op watch that starts and stops
// cleanly but never samples or trips — callers do not need their own "is this enabled" branch
// around every call site; GOINFER_SWAP_GUARD=off (consulted by callers, not here — this is the
// portable primitive, not the env-var policy) is expected to pass a nil Read to get this shape.
func StartSwapWatch(ctx context.Context, opts SwapWatchOptions) *SwapWatch {
	if opts.PollInterval <= 0 {
		opts.PollInterval = 2 * time.Second
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 512 * 1024 * 1024
	}
	watchCtx, cancel := context.WithCancel(ctx)
	w := &SwapWatch{cancel: cancel, done: make(chan struct{})}
	if opts.Read == nil {
		close(w.done)
		return w
	}
	go runSwapWatch(watchCtx, opts, w.done)
	return w
}

func runSwapWatch(ctx context.Context, opts SwapWatchOptions, done chan<- struct{}) {
	defer close(done)
	t := time.NewTicker(opts.PollInterval)
	defer t.Stop()

	var baseline int64
	haveBaseline := false
	tripped := false
	var withinSince time.Time // zero = the hysteresis clock is not currently running

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		used, ok := opts.Read()
		if !ok {
			continue // unknown this tick — try again next tick, never substitute a guess
		}
		if !haveBaseline {
			// The FIRST successful reading, whatever it is, becomes the baseline — including on
			// a machine already deep in swap from something else. Everything below is a delta
			// over THIS, not an absolute figure.
			baseline = used
			haveBaseline = true
			continue
		}
		delta := used - baseline
		within := delta <= opts.Threshold

		if !tripped {
			if !within {
				tripped = true
				if opts.OnTrip != nil {
					opts.OnTrip(used, delta)
				}
			}
			continue
		}

		// Already tripped: track the hysteresis window toward OnResume, if the caller asked for one.
		if opts.ResumeAfter <= 0 || opts.OnResume == nil {
			continue
		}
		if !within {
			withinSince = time.Time{} // still above threshold (or grew again) — reset the clock
			continue
		}
		if withinSince.IsZero() {
			withinSince = time.Now()
			continue
		}
		if time.Since(withinSince) >= opts.ResumeAfter {
			tripped = false
			withinSince = time.Time{}
			opts.OnResume()
		}
	}
}
