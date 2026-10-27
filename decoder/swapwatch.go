package decoder

import (
	"context"
	"time"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// SwapReader samples the machine's current swap-used bytes. ok is false when the figure could
// not be determined (no swap concept on this platform, or a probe that failed this tick) — the
// watcher treats "unknown" as "do nothing and try again next tick", never as a guess in either
// direction. SwapUsedBytes is the real probe; tests inject a scripted fake.
type SwapReader func() (usedBytes int64, ok bool)

// SwapWatchOptions configures StartSwapWatch. Read is required; every other field defaults. The watch keys on swap-used,
// never RSS: darwin RSS reports what survived reclaim under memory pressure, so it can fall while the machine is losing
// memory (docs/code-notes/decoder.md#swapwatch).
type SwapWatchOptions struct {
	Read SwapReader

	// PollInterval between samples. Default 2s.
	PollInterval time.Duration

	// Threshold is how far swap-used may grow above the first successful sample's baseline before OnTrip fires. Default
	// 512 MiB. A machine already deep in swap from something unrelated has a baseline that includes it: this measures only
	// what grows after the watch starts, i.e. what the watched process itself adds.
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
type SwapWatch struct{ w *probe.Watch }

// Rebaseline asks the watch to take its next successful reading as the new baseline and forget a trip: swap-used at that moment is the new
// zero, and if the watch was tripped it calls OnResume (when set) once. A serving watch armed before a model load calls it when the load
// has finished, so swap the load itself put there (pages that stay in swap until touched, and so never "recover") is not charged to serving.
// It returns at once; the change lands at the next tick. A load that grew swap past the threshold is not hidden by this: the load-time watch
// (Options.LoadAbort) aborts it first, and a failed load never reaches here. Safe on a nil receiver and from any goroutine.
func (w *SwapWatch) Rebaseline() {
	if w == nil {
		return
	}
	w.w.Rebaseline()
}

// Stop ends the watch and blocks until its goroutine has actually exited (so a caller that also
// wants to read state the callbacks wrote does not race the last tick). Safe on a nil receiver
// and safe to call more than once.
func (w *SwapWatch) Stop() {
	if w == nil {
		return
	}
	w.w.Stop()
}

// StartSwapWatch starts sampling in a background goroutine and returns immediately; ctx ending is
// equivalent to calling Stop. A nil opts.Read makes this a no-op watch that starts and stops
// cleanly but never samples or trips — callers do not need their own "is this enabled" branch
// around every call site; GOINFER_SWAP_GUARD=off (consulted by callers, not here — this is the
// portable primitive, not the env-var policy) is expected to pass a nil Read to get this shape.
func StartSwapWatch(ctx context.Context, opts SwapWatchOptions) *SwapWatch {
	return &SwapWatch{w: probe.StartWatch(ctx, probe.WatchOptions{
		Read:         opts.Read,
		PollInterval: opts.PollInterval,
		Threshold:    opts.Threshold,
		ResumeAfter:  opts.ResumeAfter,
		OnTrip:       opts.OnTrip,
		OnResume:     opts.OnResume,
	})}
}
